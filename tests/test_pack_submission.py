"""Tests for tools/pack_submission.py on a tiny fake checkpoint. Run: python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pack_submission as ps  # noqa: E402

LOCAL_VLM = "/workspace/runs/x/vlm/merged"


def make_run(root: Path) -> Path:
    """A fake kit output dir: runs/<run>/submission/{smolvlm2,action_expert}."""
    sub = root / "run" / "submission"
    vlm = sub / "smolvlm2"
    vlm.mkdir(parents=True)
    save_file({"w": torch.ones(2, 2), "ids": torch.arange(3)}, vlm / "model.safetensors")
    (vlm / "config.json").write_text(json.dumps({"dtype": "float32", "text_config": {"dtype": "float32"}}))
    for n in ("generation_config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (vlm / n).write_text("{}")
    (vlm / "chat_template.jinja").write_text("{{ messages }}")
    (vlm / "checkpoint-500").mkdir()  # trainer dir, must be dropped

    ae = sub / "action_expert"
    ae.mkdir()
    save_file({"model.a": torch.zeros(4), "model.b": torch.zeros(2, 2)}, ae / "model.safetensors")
    save_file({"mean": torch.zeros(7)}, ae / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    save_file({"mean": torch.zeros(7)}, ae / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    (ae / "config.json").write_text(json.dumps({"type": "smolvla", "vlm_model_name": LOCAL_VLM, "pretrained_path": None}))
    (ae / "train_config.json").write_text(json.dumps(
        {"dataset": {"root": "/data/vla"}, "policy": {"vlm_model_name": LOCAL_VLM}}))
    (ae / "policy_preprocessor.json").write_text(json.dumps(
        {"steps": [{"config": {"tokenizer_name": LOCAL_VLM}}]}))
    (ae / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
    return sub


def make_code_dir(root: Path) -> Path:
    code = root / "code"
    (code / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (code / "src" / "pkg" / "__init__.py").write_text("")
    (code / "src" / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"")
    for n in ps.CODE_FILES:
        (code / n).write_text("x\n")
    return code


def make_manifest(root: Path) -> Path:
    m = root / "manifest.json"
    m.write_text(json.dumps({"files": {"action_expert/model.safetensors": {"tensors": {
        "model.a": {"shape": [4]}, "model.b": {"shape": [2, 2]}}}}}))
    return m


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = make_run(self.root)
        self.ann = self.root / "annotations.jsonl"
        self.ann.write_text('{"id": 0}\n')
        self.base_args = ["--name", "t", "--from", str(self.run_dir), "--annotations", str(self.ann),
                          "--out-dir", str(self.root / "out"), "--code-dir", str(make_code_dir(self.root)),
                          "--manifest", str(make_manifest(self.root))]

    def tearDown(self):
        self.tmp.cleanup()

    def pack(self, *extra):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ps.main([*self.base_args, *extra])
        return code, err.getvalue()

    @property
    def zip_path(self):
        return self.root / "out" / "t.zip"

    def test_happy_path(self):
        code, err = self.pack()
        self.assertEqual(code, 0, err)
        with zipfile.ZipFile(self.zip_path) as zf:
            names = set(zf.namelist())
            for sub, files in ps.REQUIRED.items():
                for f in files:
                    if f != "src":
                        self.assertIn(f"{sub}/{f}".lstrip("/"), names)
            self.assertIn("src/pkg/__init__.py", names)
            self.assertFalse(any("checkpoint-500" in n or "__pycache__" in n for n in names))
            ae = {n: json.loads(zf.read(f"action_expert/{n}")) for n in
                  ("config.json", "train_config.json", "policy_preprocessor.json")}
            vlm_cfg = json.loads(zf.read("smolvlm2/config.json"))
        self.assertEqual(ae["config.json"]["vlm_model_name"], ps.JUDGE_VLM_PATH)
        self.assertEqual(ae["train_config.json"]["policy"]["vlm_model_name"], ps.JUDGE_VLM_PATH)
        self.assertEqual(ae["policy_preprocessor.json"]["steps"][0]["config"]["tokenizer_name"], ps.JUDGE_VLM_PATH)
        self.assertEqual(vlm_cfg["dtype"], "bfloat16")
        self.assertEqual(vlm_cfg["text_config"]["dtype"], "bfloat16")
        with safe_open(self.root / "out" / "t" / "smolvlm2" / "model.safetensors", framework="pt") as f:
            self.assertEqual(f.get_tensor("w").dtype, torch.bfloat16)
            self.assertEqual(f.get_tensor("ids").dtype, torch.int64)  # non-float untouched
        # source checkpoint is not modified
        self.assertEqual(json.loads((self.run_dir / "action_expert" / "config.json").read_text())["vlm_model_name"],
                         LOCAL_VLM)

    def test_keep_fp32(self):
        self.assertEqual(self.pack("--keep-fp32")[0], 0)
        with safe_open(self.root / "out" / "t" / "smolvlm2" / "model.safetensors", framework="pt") as f:
            self.assertEqual(f.get_tensor("w").dtype, torch.float32)

    def assert_rejected(self, needle):
        code, err = self.pack()
        self.assertEqual(code, 1)
        self.assertIn(needle, err)
        self.assertFalse(self.zip_path.exists())

    def test_rejects_image(self):
        (self.run_dir / "smolvlm2" / "frame.jpg").write_bytes(b"\xff")
        self.assert_rejected("forbidden file smolvlm2/frame.jpg")

    def test_rejects_remote_ref(self):
        p = self.run_dir / "action_expert" / "train_config.json"
        cfg = json.loads(p.read_text())
        cfg["dataset"]["root"] = "hf://datasets/HuggingFaceVLA/libero"
        p.write_text(json.dumps(cfg))
        self.assert_rejected("remote reference")

    def test_rejects_missing_tokenizer(self):
        (self.run_dir / "smolvlm2" / "tokenizer.json").unlink()
        self.assert_rejected("missing smolvlm2/tokenizer.json")

    def test_rejects_nan(self):
        save_file({"model.a": torch.tensor([0.0, float("nan"), 0.0, 0.0]), "model.b": torch.zeros(2, 2)},
                  self.run_dir / "action_expert" / "model.safetensors")
        self.assert_rejected("non-finite values in model.a")

    def test_rejects_tensor_name_drift(self):
        save_file({"model.a": torch.zeros(4)}, self.run_dir / "action_expert" / "model.safetensors")
        self.assert_rejected("tensor names differ from the demo")

    def test_rejects_pretrained_path(self):
        p = self.run_dir / "action_expert" / "config.json"
        cfg = json.loads(p.read_text())
        cfg["pretrained_path"] = "/workspace/runs/old"
        p.write_text(json.dumps(cfg))
        self.assert_rejected("must be null")

    def test_no_judge_paths_keeps_local_path(self):
        self.assertEqual(self.pack("--no-judge-paths")[0], 0)
        with zipfile.ZipFile(self.zip_path) as zf:
            self.assertEqual(json.loads(zf.read("action_expert/config.json"))["vlm_model_name"], LOCAL_VLM)

    def test_refuses_overwrite_without_force(self):
        self.assertEqual(self.pack()[0], 0)
        self.assertEqual(self.pack()[0], 2)
        self.assertEqual(self.pack("--force")[0], 0)


if __name__ == "__main__":
    unittest.main()
