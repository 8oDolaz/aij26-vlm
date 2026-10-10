#!/usr/bin/env python3
"""Build an upload-ready submission ZIP from a kit run plus the generator code in submission/.

Stages into artifacts/<name>/ (submission/ itself stays code-only), then:
- copies smolvlm2/ (top-level files only) and saves it in bf16, which the benches load anyway;
- copies action_expert/ and rewrites vlm_model_name / tokenizer_name to the judge path;
- checks layout, no images/caches, no remote refs, tensor names vs the demo, no NaN/Inf;
- zips to artifacts/<name>.zip (<= 5 GB).
Then verify it offline in the organizers' image: tools/verify_submission.sh artifacts/<name>.zip

Example:
  python tools/pack_submission.py --name a0_base --from runs/a0/submission \\
      --annotations data/gen/v0/annotations.jsonl --configs configs/vla/a0.yaml
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JUDGE_VLM_PATH = "/home/ubuntu22/aij_robotics_judge/submission/smolvlm2"
JUDGE_PATH_KEYS = ("vlm_model_name", "tokenizer_name")
DEMO_MANIFEST = REPO / "vendor" / "demo_submission" / "weights_manifest.json"
MAX_ZIP_BYTES = 5 * 10**9  # "5 GB"; decimal is the stricter reading

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
    return rewritten


def stage(args, stage_dir: Path) -> int:
    sub = args.code_dir
    for name in CODE_FILES:
        shutil.copy2(sub / name, stage_dir / name)
    shutil.copytree(sub / "src", stage_dir / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    shutil.copy2(args.annotations, stage_dir / "annotations.jsonl")
    if args.configs:
        (stage_dir / "configs").mkdir()
        for c in args.configs:
            shutil.copy2(c, stage_dir / "configs" / c.name)
    print(f"  smolvlm2      <- {args.vlm}{'' if args.keep_fp32 else ' (to bf16)'}")
    copy_vlm(args.vlm, stage_dir / "smolvlm2", bf16=not args.keep_fp32)
    print(f"  action_expert <- {args.action_expert}")
    judge = None if args.no_judge_paths else args.judge_path
    return copy_action_expert(args.action_expert, stage_dir / "action_expert", judge)


def check_tensors(stage_dir: Path, manifest: Path | None) -> tuple[list[str], list[str]]:
    import torch
    from safetensors import safe_open

    errors, warnings = [], []
    expected = None
    if manifest is not None:
        files = json.loads(manifest.read_text())["files"]
        expected = files.get("action_expert/model.safetensors", {}).get("tensors")

    for f in sorted(stage_dir.glob("*/*.safetensors")):
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
    ap.add_argument("--force", action="store_true", help="overwrite an existing artifacts/<name>")
    ap.add_argument("--out-dir", type=Path, default=REPO / "artifacts")
    ap.add_argument("--code-dir", type=Path, default=REPO / "submission", help=argparse.SUPPRESS)
    ap.add_argument("--judge-path", default=JUDGE_VLM_PATH, help=argparse.SUPPRESS)
    ap.add_argument("--manifest", default=str(DEMO_MANIFEST),
                    help="demo weight manifest for the tensor-name check; '' disables it")
    ap.add_argument("--no-judge-paths", action="store_true",
                    help="testing only: skip the path rewrite and the checks on it (negative control for verify)")
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
    args.manifest = Path(args.manifest) if args.manifest else None
    return args


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
    print("checking tensors")
    e, w = check_tensors(stage_dir, args.manifest)
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
    for part in ("smolvlm2", "action_expert"):
        print(f"  {part:<14} {dir_size(stage_dir / part) / 1e9:.2f} GB")
    print(f"  judge-path fields rewritten: {rewritten}")
    print(f"wrote {zip_path} ({size / 1e9:.2f} GB)")
    if size > MAX_ZIP_BYTES:
        print(f"  ERROR: zip exceeds {MAX_ZIP_BYTES / 1e9:.0f} GB", file=sys.stderr)
        zip_path.unlink()
        return 1
    print(f"next (on the GPU server host): tools/verify_submission.sh {zip_path.relative_to(REPO) if zip_path.is_relative_to(REPO) else zip_path} --libero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
