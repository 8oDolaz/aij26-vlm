#!/usr/bin/env python3
"""Fold a LoRA (PEFT) SmolVLA checkpoint from tools/vla_peft_shim into a plain SmolVLA policy directory, so the
submission's action_expert has the usual layout (config.json + model.safetensors + processors).

Run inside the dev container:
  python tools/merge_vla_lora.py runs/<run>/submission/action_expert \
      --reference runs/<plain run>/submission/action_expert
The directory is replaced in place; the adapter version is moved to <dir>/../<name>.peft_adapter (outside submission/
is better for packing: pass --keep-adapter to choose where). --reference checks the merged tensor names against a
plainly trained action_expert and fails on any difference.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def tensor_names(path: Path) -> set[str]:
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return set(f.keys())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_dir", type=Path, help="PEFT checkpoint dir (adapter_config.json inside); replaced in place")
    ap.add_argument("--keep-adapter", type=Path, help="where to move the adapter version (default: next to it)")
    ap.add_argument("--reference", type=Path, help="plain action_expert dir whose tensor names must match")
    args = ap.parse_args()

    src = args.policy_dir.resolve()
    if not (src / "adapter_config.json").is_file():
        print(f"{src} has no adapter_config.json: nothing to merge")
        return 0 if (src / "model.safetensors").is_file() else 1

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(str(src))
    print(f"base policy: {peft_cfg.base_model_name_or_path}")
    base = SmolVLAPolicy.from_pretrained(peft_cfg.base_model_name_or_path)
    merged = PeftModel.from_pretrained(base, str(src), config=peft_cfg).merge_and_unload()
    merged.config.use_peft = False

    tmp = src.with_name(src.name + ".merging")
    if tmp.exists():
        shutil.rmtree(tmp)
    merged.save_pretrained(tmp)
    for f in src.iterdir():  # processors, train config: everything that is not the adapter or the weights
        if f.is_file() and not f.name.startswith("adapter_") and not (tmp / f.name).exists():
            shutil.copy2(f, tmp / f.name)
    tc = tmp / "train_config.json"
    if tc.is_file():
        d = json.loads(tc.read_text())
        d["peft"] = None
        if isinstance(d.get("policy"), dict):
            d["policy"]["use_peft"] = False
        tc.write_text(json.dumps(d, indent=4))

    keep = (args.keep_adapter or src.with_name(src.name + ".peft_adapter")).resolve()
    if keep.exists():
        shutil.rmtree(keep)
    shutil.move(str(src), str(keep))
    shutil.move(str(tmp), str(src))
    print(f"merged policy: {src} (adapter kept at {keep})")

    if args.reference:
        got, ref = tensor_names(src / "model.safetensors"), tensor_names(args.reference / "model.safetensors")
        if got != ref:
            print(f"tensor names differ from {args.reference}: {len(got - ref)} extra, {len(ref - got)} missing")
            for k in sorted(got - ref)[:10]:
                print("  extra  ", k)
            for k in sorted(ref - got)[:10]:
                print("  missing", k)
            return 1
        print(f"tensor names match {args.reference} ({len(got)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
