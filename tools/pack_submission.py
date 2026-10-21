#!/usr/bin/env python3
"""Build an upload-ready submission ZIP from a kit run plus the generator code in submission/.

Stages into artifacts/<name>/ (submission/ itself stays code-only), then:
- copies smolvlm2/ (top-level files only) and saves it in bf16, which the benches load anyway;
- copies action_expert/ and rewrites vlm_model_name / tokenizer_name to the judge path;
- checks layout, no images/caches, no remote refs, tensor names vs the demo, no NaN/Inf;
- with generator weights (--sam3/--da3/--sam2) copies them to weights/<name>/ (SAM 3 in bf16, which it runs in anyway;
  the others bit-exact), vendors the DA3 source and missing pure-Python deps (--da3-src, --pydeps) into src/third_party/,
  and points config.yaml's model refs at them, so the generator runs offline from the ZIP;
- checks layout, no images/caches, no remote refs, every config.yaml model ref shipped, tensor names vs the demo,
  no NaN/Inf;
- zips to artifacts/<name>.zip (<= 5 GB).
Then verify it offline in the organizers' image: tools/verify_submission.sh artifacts/<name>.zip

Example:
  python tools/pack_submission.py --name a0_base --from runs/a0/submission \\
      --annotations data/gen/v0/annotations.jsonl --configs configs/vla/a0.yaml \\
      --sam3 <sam3 snapshot> --da3 <DA3-BASE snapshot> --sam2 <sam2.1-hiera-small snapshot> \\
      --da3-src example/depth-anything-3/src --pydeps <site-packages>/addict
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JUDGE_VLM_PATH = "/home/ubuntu22/aij_robotics_judge/submission/smolvlm2"
JUDGE_PATH_KEYS = ("vlm_model_name", "tokenizer_name")
DEMO_MANIFEST = REPO / "vendor" / "demo_submission" / "weights_manifest.json"
DEMO_AE_CONFIG = REPO / "vendor" / "demo_submission" / "action_expert" / "config.json"
MAX_ZIP_BYTES = 5 * 10**9  # "5 GB"; decimal is the stricter reading

# Generator models: CLI flag -> (HF id used in config.yaml, path in the ZIP, convert to bf16)
GEN_WEIGHTS = {
    "sam3": ("facebook/sam3", "weights/sam3", True),
    "da3": ("depth-anything/DA3-BASE", "weights/da3-base", False),
    "sam2": ("facebook/sam2.1-hiera-small", "weights/sam2.1-hiera-small", False),
}
THIRD_PARTY = "src/third_party"  # DA3 package + vendored deps; DepthTeacher puts da3_src on sys.path
DA3_SRC_REF = "example/depth-anything-3/src"
MODEL_REF_KEYS = ("sam3_model", "depth_model", "segmenter", "da3_src", "sam_model", "siglip_model")
VENDOR_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt"}

CODE_FILES = ("README.md", "requirements.txt", "generate_dataset.py", "config.yaml")
# Layout from docs/organizers/task_description.txt
REQUIRED = {
    "": ("README.md", "requirements.txt", "generate_dataset.py", "config.yaml", "annotations.jsonl", "src"),
    "smolvlm2": ("config.json", "model.safetensors", "generation_config.json", "processor_config.json",
                 "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"),
    "action_expert": ("config.json", "model.safetensors", "train_config.json",
                      "policy_preprocessor.json", "policy_postprocessor.json"),
}
REQUIRED_GLOBS = ("action_expert/policy_preprocessor_step_*.safetensors",
                  "action_expert/policy_postprocessor_step_*.safetensors")
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
                      ".mp4", ".avi", ".mov", ".mkv", ".webm", ".parquet", ".arrow", ".pyc"}
FORBIDDEN_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", ".cache"}
REMOTE_MARKERS = ("hf://", "http://", "https://", "huggingface.co")
# LLaMA-Factory writes these next to the final weights in vlm/checkpoints/; not part of the model.
TRAINER_FILES = {"training_loss.png", "trainer_log.jsonl", "trainer_state.json", "training_args.bin",
                 "all_results.json", "train_results.json", "README.md"}


def _is_trainer_dir(name: str) -> bool:
    # Same rule as the kit's _ignore_trainer_checkpoints in run_pipeline.py.
    return name.startswith("checkpoint-") or name.startswith("global_step")


def iter_strings(obj, path=""):
    """Yield (json_path, key, value) for every string in a JSON tree."""
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else ()
    for k, v in items:
        p = f"{path}.{k}" if isinstance(obj, dict) else f"{path}[{k}]"
        if isinstance(v, str):
            yield p, (k if isinstance(obj, dict) else None), v
        else:
            yield from iter_strings(v, p)


def rewrite_keys(obj, keys, value) -> int:
    """Set every string field named in keys to value, in place. Returns how many were set."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                obj[k] = value
                n += 1
            else:
                n += rewrite_keys(v, keys, value)
    elif isinstance(obj, list):
        for v in obj:
            n += rewrite_keys(v, keys, value)
    return n


def set_dtype_bf16(obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("dtype", "torch_dtype") and v == "float32":
                obj[k] = "bfloat16"
            else:
                set_dtype_bf16(v)
    elif isinstance(obj, list):
        for v in obj:
            set_dtype_bf16(v)


def convert_safetensors_bf16(src: Path, dst: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            t = f.get_tensor(k)
            tensors[k] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
    save_file(tensors, dst, metadata={**meta, "format": "pt"})


def copy_vlm(src: Path, dst: Path, bf16: bool) -> None:
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_dir():
            if not _is_trainer_dir(f.name):
                print(f"  note: skipping subdir smolvlm2/{f.name}")
            continue
        if f.name in TRAINER_FILES:
            print(f"  note: skipping trainer file smolvlm2/{f.name}")
            continue
        if bf16 and f.suffix == ".safetensors":
            convert_safetensors_bf16(f, dst / f.name)
        elif bf16 and f.name == "config.json":
            cfg = json.loads(f.read_text())
            set_dtype_bf16(cfg)
            (dst / f.name).write_text(json.dumps(cfg, indent=2) + "\n")
        else:
            shutil.copy2(f, dst / f.name)


def copy_action_expert(src: Path, dst: Path, judge_path: str | None) -> int:
    """Copy the policy dir; rewrite judge-path fields unless judge_path is None. Returns fields rewritten."""
    dst.mkdir(parents=True)
    rewritten = 0
    for f in sorted(src.iterdir()):
        if f.is_dir():
            print(f"  note: skipping subdir action_expert/{f.name}")
            continue
        if f.suffix == ".json" and judge_path is not None:
            cfg = json.loads(f.read_text())
            n = rewrite_keys(cfg, JUDGE_PATH_KEYS, judge_path)
            if n:
                (dst / f.name).write_text(json.dumps(cfg, indent=4) + "\n")
                rewritten += n
                continue
        shutil.copy2(f, dst / f.name)
    ensure_policy_type(dst)
    return rewritten


def ensure_policy_type(ae_dir: Path) -> None:
    """LeRobot checkpoints may omit `type` in config.json; the kit's loader (scaling_lawyer smolvla_model
    _ensure_policy_config_type) then writes it from train_config.json, which fails on a read-only submission.
    Do the same here so the shipped config needs no rewrite."""
    path = ae_dir / "config.json"
    if not path.is_file():
        return
    cfg = json.loads(path.read_text())
    if cfg.get("type"):
        return
    policy_type = "smolvla"
    tc = ae_dir / "train_config.json"
    if tc.is_file():
        nested = json.loads(tc.read_text()).get("policy")
        if isinstance(nested, dict) and nested.get("type"):
            policy_type = str(nested["type"])
    cfg["type"] = policy_type
    path.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"  note: action_expert/config.json had no type; set to {policy_type!r} (as the kit's loader would)")


def copy_gen_weights(src: Path, dst: Path, bf16: bool) -> None:
    """An HF snapshot dir (symlinks followed) into dst; safetensors and config dtype to bf16 if asked."""
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_dir() or f.name.startswith("."):
            continue
        if bf16 and f.suffix == ".safetensors":
            convert_safetensors_bf16(f, dst / f.name)
        elif bf16 and f.name == "config.json":
            cfg = json.loads(f.read_text())
            set_dtype_bf16(cfg)
            (dst / f.name).write_text(json.dumps(cfg, indent=2) + "\n")
        else:
            shutil.copy(f, dst / f.name)


def _vendor_ignore(d, names):
    """Only source and config files of vendored packages (no caches, images, notebooks)."""
    return [n for n in names if n in FORBIDDEN_DIRS
            or (not (Path(d) / n).is_dir() and Path(n).suffix not in VENDOR_SUFFIXES)]


def find_license(pkg_dir: Path) -> Path | None:
    """LICENSE next to a source checkout's package dir, or in a site-packages <name>-*.dist-info."""
    cands = [*pkg_dir.parent.glob("LICENSE*"), *pkg_dir.parent.parent.glob("LICENSE*"),
             *pkg_dir.parent.glob(f"{pkg_dir.name}-*.dist-info/LICENSE*"),
             *pkg_dir.parent.glob(f"{pkg_dir.name}-*.dist-info/licenses/LICENSE*")]
    return next((c for c in cands if c.is_file()), None)


def vendor_package(pkg: Path, third: Path) -> None:
    shutil.copytree(pkg, third / pkg.name, ignore=_vendor_ignore)
    lic = find_license(pkg)
    if lic is not None:
        (third / "LICENSES").mkdir(exist_ok=True)
        shutil.copy(lic, third / "LICENSES" / f"{pkg.name}.LICENSE")
    else:
        print(f"  note: no LICENSE found for {pkg.name}")


def rewrite_config_refs(text: str, ref_map: dict[str, str]) -> tuple[str, int]:
    """Point `key: <HF id>` lines of config.yaml at shipped paths (comments elsewhere kept). Returns (text, n)."""
    pat = re.compile(rf"^(\s*)({'|'.join(MODEL_REF_KEYS)}):\s*(\S+)(.*)$", re.M)
    n = 0

    def sub(m):
        nonlocal n
        indent, key, val, _ = m.groups()
        if val not in ref_map:
            return m.group(0)
        n += 1
        return f"{indent}{key}: {ref_map[val]}  # shipped in the ZIP; {val} in development"
    return pat.sub(sub, text), n


def model_refs(obj, path=""):
    """(json_path, value) of every model reference in a parsed config."""
    for p, key, val in iter_strings(obj, path):
        if key in MODEL_REF_KEYS:
            yield p, val


def check_config_refs(stage_dir: Path) -> list[str]:
    """Every model ref in the staged config.yaml must be a relative path that exists in the ZIP (no HF ids)."""
    import yaml
    cfg = yaml.safe_load((stage_dir / "config.yaml").read_text())
    if not isinstance(cfg, dict):
        return []
    return [f"config.yaml{p} = {v!r} is not shipped in the ZIP (an HF id or missing path; offline run would fail)"
            for p, v in model_refs(cfg) if Path(v).is_absolute() or not (stage_dir / v).exists()]


def stage_generator(args, stage_dir: Path) -> int:
    """Generator weights + vendored code, then config.yaml refs rewritten. Returns refs rewritten."""
    ref_map = {}
    for flag, (hf_id, rel, bf16) in GEN_WEIGHTS.items():
        src = getattr(args, flag)
        if src is None:
            continue
        print(f"  {rel:<27} <- {src}{' (to bf16)' if bf16 else ''}")
        copy_gen_weights(src, stage_dir / rel, bf16)
        ref_map[hf_id] = rel
    third = stage_dir / THIRD_PARTY
    if args.da3_src is not None:
        print(f"  {THIRD_PARTY}/depth_anything_3 <- {args.da3_src}")
        third.mkdir(parents=True, exist_ok=True)
        vendor_package(args.da3_src / "depth_anything_3", third)
        ref_map[DA3_SRC_REF] = THIRD_PARTY
    for pkg in args.pydeps:
        print(f"  {THIRD_PARTY}/{pkg.name} <- {pkg}")
        third.mkdir(parents=True, exist_ok=True)
        vendor_package(pkg, third)
    cfg = stage_dir / "config.yaml"
    text, n = rewrite_config_refs(cfg.read_text(), ref_map)
    cfg.write_text(text)
    return n


def stage(args, stage_dir: Path) -> int:
    sub = args.code_dir
    for name in CODE_FILES:
        shutil.copy2(sub / name, stage_dir / name)
    shutil.copytree(sub / "src", stage_dir / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    args.config_refs_rewritten = stage_generator(args, stage_dir)
    shutil.copy2(args.annotations, stage_dir / "annotations.jsonl")
    if args.configs:
        (stage_dir / "configs").mkdir()
        for c in args.configs:
            shutil.copy2(c, stage_dir / "configs" / c.name)
    print(f"  smolvlm2      <- {args.vlm}{'' if args.keep_fp32 else ' (to bf16)'}")
    copy_vlm(args.vlm, stage_dir / "smolvlm2", bf16=not args.keep_fp32)
    if args.vlm_fill is not None:
        for name in REQUIRED["smolvlm2"]:
            if not (stage_dir / "smolvlm2" / name).exists() and (args.vlm_fill / name).is_file():
                print(f"  note: smolvlm2/{name} missing in {args.vlm}, taken from {args.vlm_fill}")
                shutil.copy(args.vlm_fill / name, stage_dir / "smolvlm2" / name)
    print(f"  action_expert <- {args.action_expert}")
    judge = None if args.no_judge_paths else args.judge_path
    return copy_action_expert(args.action_expert, stage_dir / "action_expert", judge)


def same_layers_as_demo(stage_dir: Path, demo_config: Path | None) -> bool:
    """False when the policy keeps a different number of VLM layers than the demo (num_vlm_layers), so its tensor
    names legitimately differ from the demo's."""
    if demo_config is None or not demo_config.is_file():
        return True
    ours = json.loads((stage_dir / "action_expert" / "config.json").read_text()).get("num_vlm_layers")
    demo = json.loads(demo_config.read_text()).get("num_vlm_layers")
    return ours == demo


def check_tensors(stage_dir: Path, manifest: Path | None,
                  demo_config: Path | None = None) -> tuple[list[str], list[str]]:
    import torch
    from safetensors import safe_open

    errors, warnings = [], []
    expected = None
    if manifest is not None:
        if same_layers_as_demo(stage_dir, demo_config):
            files = json.loads(manifest.read_text())["files"]
            expected = files.get("action_expert/model.safetensors", {}).get("tensors")
        else:
            warnings.append("action_expert num_vlm_layers differs from the demo: tensor-name check vs the demo skipped")

    for f in sorted([*stage_dir.glob("*/*.safetensors"), *stage_dir.glob("weights/*/*.safetensors")]):
        rel = f.relative_to(stage_dir).as_posix()
        with safe_open(f, framework="pt") as sf:
            keys = set(sf.keys())
            for k in sorted(keys):
                t = sf.get_tensor(k)
                if t.is_floating_point() and not torch.isfinite(t).all():
                    errors.append(f"{rel}: non-finite values in {k}")
                if expected and rel == "action_expert/model.safetensors" and k in expected:
                    if list(t.shape) != expected[k]["shape"]:
                        warnings.append(f"{rel}: {k} shape {list(t.shape)} != demo {expected[k]['shape']}")
        if expected and rel == "action_expert/model.safetensors" and keys != set(expected):
            missing, extra = sorted(set(expected) - keys), sorted(keys - set(expected))
            errors.append(f"{rel}: tensor names differ from the demo "
                          f"(missing {len(missing)}: {missing[:3]}, extra {len(extra)}: {extra[:3]})")
    return errors, warnings


def check_staged(stage_dir: Path, judge_path: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the layout and configs of a staged submission."""
    errors, warnings = [], []
    for sub, names in REQUIRED.items():
        for n in names:
            if not (stage_dir / sub / n).exists():
                errors.append(f"missing {Path(sub, n)}")
    for pattern in REQUIRED_GLOBS:
        if not list(stage_dir.glob(pattern)):
            errors.append(f"missing {pattern}")

    for p in sorted(stage_dir.rglob("*")):
        rel = p.relative_to(stage_dir)
        if p.is_dir() and p.name in FORBIDDEN_DIRS:
            errors.append(f"forbidden dir {rel}")
        elif p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file {rel}")

    configs = [*(stage_dir / "action_expert").glob("*.json"), *(stage_dir / "smolvlm2").glob("*config*.json")]
    for f in sorted(configs):
        rel = f.relative_to(stage_dir)
        for jpath, key, val in iter_strings(json.loads(f.read_text())):
            if key in JUDGE_PATH_KEYS:
                if val != judge_path:
                    errors.append(f"{rel}{jpath} = {val!r}, expected {judge_path!r}")
            elif any(m in val for m in REMOTE_MARKERS):
                errors.append(f"{rel}{jpath} = {val!r} looks like a remote reference")
            elif key == "pretrained_path" and val:
                errors.append(f"{rel}{jpath} = {val!r}; must be null")
            elif val.startswith("/"):
                # The demo ships train_config.json with dataset.root / output_dir like this and passes.
                warnings.append(f"{rel}{jpath} = {val!r} (local path, informational)")
    return errors, warnings


def write_zip(stage_dir: Path, zip_path: Path) -> None:
    tmp = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for p in sorted(stage_dir.rglob("*")):
            if p.is_file():
                comp = zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
                zf.write(p, p.relative_to(stage_dir).as_posix(), compress_type=comp)
    tmp.replace(zip_path)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output: artifacts/<name>/ and artifacts/<name>.zip")
    ap.add_argument("--from", dest="from_dir", type=Path,
                    help="kit output dir with smolvlm2/ and action_expert/ (runs/<run>/submission)")
    ap.add_argument("--vlm", type=Path, help="smolvlm2 dir; overrides --from")
    ap.add_argument("--action-expert", type=Path, help="action_expert dir; overrides --from")
    ap.add_argument("--annotations", type=Path, required=True, help="the JSONL the VLM was trained on")
    ap.add_argument("--configs", type=Path, nargs="*", default=[], help="final SFT/VLA yaml files to include")
    ap.add_argument("--keep-fp32", action="store_true", help="copy smolvlm2 weights without bf16 conversion")
    ap.add_argument("--vlm-fill", type=Path,
                    help="take required smolvlm2 files missing from --vlm (e.g. tokenizer.json) from this dir, "
                         "e.g. the final LLaMA-Factory checkpoint-<step>/")
    ap.add_argument("--force", action="store_true", help="overwrite an existing artifacts/<name>")
    ap.add_argument("--out-dir", type=Path, default=REPO / "artifacts")
    ap.add_argument("--code-dir", type=Path, default=REPO / "submission", help=argparse.SUPPRESS)
    ap.add_argument("--judge-path", default=JUDGE_VLM_PATH, help=argparse.SUPPRESS)
    ap.add_argument("--manifest", default=str(DEMO_MANIFEST),
                    help="demo weight manifest for the tensor-name check; '' disables it")
    ap.add_argument("--no-judge-paths", action="store_true",
                    help="testing only: skip the path rewrite and the checks on it (negative control for verify)")
    ap.add_argument("--demo-config", default=str(DEMO_AE_CONFIG), help=argparse.SUPPRESS)
    gen = ap.add_argument_group("generator (offline finalist run)")
    for flag, (hf_id, rel, bf16) in GEN_WEIGHTS.items():
        gen.add_argument(f"--{flag}", type=Path, help=f"local {hf_id} snapshot dir -> {rel}{' (bf16)' if bf16 else ''}")
    gen.add_argument("--da3-src", type=Path, help=f"dir holding the depth_anything_3 package -> {THIRD_PARTY}/")
    gen.add_argument("--pydeps", type=Path, nargs="*", default=[],
                     help=f"pure-Python package dirs missing from the judge image (addict) -> {THIRD_PARTY}/")
    gen.add_argument("--allow-hub-refs", action="store_true",
                     help="testing only: allow config.yaml model refs that are not shipped (HF ids)")
    args = ap.parse_args(argv)

    if args.from_dir is not None:
        args.vlm = args.vlm or args.from_dir / "smolvlm2"
        args.action_expert = args.action_expert or args.from_dir / "action_expert"
    if args.vlm is None or args.action_expert is None:
        ap.error("give --from, or both --vlm and --action-expert")
    for p in (args.vlm, args.action_expert):
        if not p.is_dir():
            ap.error(f"not a directory: {p}")
    if not args.annotations.is_file():
        ap.error(f"not a file: {args.annotations}")
    for c in args.configs:
        if not c.is_file():
            ap.error(f"not a file: {c}")
    for p in [*(getattr(args, f) for f in GEN_WEIGHTS), args.da3_src, *args.pydeps]:
        if p is not None and not p.is_dir():
            ap.error(f"not a directory: {p}")
    if args.da3_src is not None and not (args.da3_src / "depth_anything_3").is_dir():
        ap.error(f"no depth_anything_3 package in {args.da3_src}")
    args.manifest = Path(args.manifest) if args.manifest else None
    args.demo_config = Path(args.demo_config) if args.demo_config else None
    return args


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    stage_dir = args.out_dir / args.name
    zip_path = args.out_dir / f"{args.name}.zip"
    if stage_dir.exists() or zip_path.exists():
        if not args.force:
            print(f"{stage_dir} or {zip_path} exists; pass --force to overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(stage_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    stage_dir.mkdir(parents=True)

    print(f"staging into {stage_dir}")
    rewritten = stage(args, stage_dir)
    errors, warnings = [], []
    if args.no_judge_paths:
        warnings.append("--no-judge-paths: configs keep their training paths; this ZIP must NOT be uploaded")
    elif rewritten == 0:
        errors.append(f"no {JUDGE_PATH_KEYS} fields found in {args.action_expert}/*.json")
    if not args.no_judge_paths:
        e, w = check_staged(stage_dir, args.judge_path)
        errors += e
        warnings += w
    ref_errors = check_config_refs(stage_dir)
    (warnings if args.allow_hub_refs else errors).extend(ref_errors)
    print("checking tensors")
    e, w = check_tensors(stage_dir, args.manifest, args.demo_config)
    errors += e
    warnings += w

    for w in warnings:
        print(f"  note: {w}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        print(f"not zipping; staged dir left at {stage_dir}", file=sys.stderr)
        return 1

    write_zip(stage_dir, zip_path)
    size = zip_path.stat().st_size
    parts = ["smolvlm2", "action_expert", *sorted(p.relative_to(stage_dir).as_posix()
                                                  for p in stage_dir.glob("weights/*") if p.is_dir()), "src"]
    for part in parts:
        print(f"  {part:<28} {dir_size(stage_dir / part) / 1e9:.3f} GB")
    print(f"  judge-path fields rewritten: {rewritten}; config.yaml model refs rewritten: {args.config_refs_rewritten}")
    print(f"wrote {zip_path} ({size / 1e9:.3f} GB, {size} bytes, limit {MAX_ZIP_BYTES})")
    if size > MAX_ZIP_BYTES:
        print(f"  ERROR: zip exceeds {MAX_ZIP_BYTES / 1e9:.0f} GB", file=sys.stderr)
        zip_path.unlink()
        return 1
    print(f"sha256 {sha256(zip_path)}")
    print(f"next (on the GPU server host): tools/verify_submission.sh {zip_path.relative_to(REPO) if zip_path.is_relative_to(REPO) else zip_path} --libero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
