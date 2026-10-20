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


GEN_CONFIG = """seed: 0
depth_tier1:
  depth_model: depth-anything/DA3-BASE   # comment
  da3_src: example/depth-anything-3/src
  segmenter: facebook/sam2.1-hiera-small
generators:
  scene_qa:
    sam3_model: facebook/sam3              # gated HF id
    depth_model: depth-anything/DA3-BASE
    da3_src: example/depth-anything-3/src
"""


def make_generator_inputs(root: Path) -> dict[str, Path]:
    """Fake HF snapshots for SAM 3 / DA3 / SAM 2.1, a DA3 source checkout and a site-packages addict."""
    out = {}
    for name in ("sam3", "da3", "sam2"):
        d = root / "hf" / name
        d.mkdir(parents=True)
        save_file({"w": torch.ones(3)}, d / "model.safetensors")
        (d / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
        out[name] = d
    (out["sam3"] / "tokenizer.json").write_text("{}")
    src = root / "da3repo" / "src"
    (src / "depth_anything_3" / "configs").mkdir(parents=True)
    (src / "depth_anything_3" / "__init__.py").write_text("")
    (src / "depth_anything_3" / "configs" / "da3-base.yaml").write_text("a: 1\n")
    (src / "depth_anything_3" / "logo.png").write_bytes(b"\x89")
    (root / "da3repo" / "LICENSE").write_text("Apache-2.0\n")
    out["da3_src"] = src
    site = root / "site"
    (site / "addict").mkdir(parents=True)
    (site / "addict" / "__init__.py").write_text("")
    (site / "addict-2.4.0.dist-info").mkdir()
    (site / "addict-2.4.0.dist-info" / "LICENSE").write_text("MIT\n")
    out["addict"] = site / "addict"
    return out


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = make_run(self.root)
        self.ann = self.root / "annotations.jsonl"
        self.ann.write_text('{"id": 0}\n')
        self.base_args = ["--name", "t", "--from", str(self.run_dir), "--annotations", str(self.ann),
                          "--out-dir", str(self.root / "out"), "--code-dir", str(make_code_dir(self.root)),
                          "--manifest", str(make_manifest(self.root)), "--demo-config", ""]

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

    def test_sets_missing_policy_type(self):
        p = self.run_dir / "action_expert" / "config.json"
        cfg = json.loads(p.read_text())
        del cfg["type"]
        p.write_text(json.dumps(cfg))
        tc = self.run_dir / "action_expert" / "train_config.json"
        tc.write_text(json.dumps({**json.loads(tc.read_text()), "policy": {"type": "smolvla",
                                                                            "vlm_model_name": LOCAL_VLM}}))
        self.assertEqual(self.pack()[0], 0)
        with zipfile.ZipFile(self.zip_path) as zf:
            self.assertEqual(json.loads(zf.read("action_expert/config.json"))["type"], "smolvla")

    def test_vlm_fill_adds_missing_tokenizer(self):
        (self.run_dir / "smolvlm2" / "tokenizer.json").unlink()
        fill = self.run_dir / "smolvlm2" / "checkpoint-500"
        (fill / "tokenizer.json").write_text('{"fill": 1}')
        code, err = self.pack("--vlm-fill", str(fill))
        self.assertEqual(code, 0, err)
        with zipfile.ZipFile(self.zip_path) as zf:
            self.assertEqual(json.loads(zf.read("smolvlm2/tokenizer.json")), {"fill": 1})

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

    def gen_args(self):
        g = make_generator_inputs(self.root)
        (self.root / "code" / "config.yaml").write_text(GEN_CONFIG)
        return ["--sam3", str(g["sam3"]), "--da3", str(g["da3"]), "--sam2", str(g["sam2"]),
                "--da3-src", str(g["da3_src"]), "--pydeps", str(g["addict"])]

    def test_generator_weights_and_config_refs(self):
        import yaml
        code, err = self.pack(*self.gen_args())
        self.assertEqual(code, 0, err)
        stage = self.root / "out" / "t"
        with zipfile.ZipFile(self.zip_path) as zf:
            names = set(zf.namelist())
            cfg = yaml.safe_load(zf.read("config.yaml"))
        for n in ("weights/sam3/model.safetensors", "weights/sam3/tokenizer.json", "weights/da3-base/config.json",
                  "weights/sam2.1-hiera-small/model.safetensors", "src/third_party/depth_anything_3/__init__.py",
                  "src/third_party/depth_anything_3/configs/da3-base.yaml", "src/third_party/addict/__init__.py",
                  "src/third_party/LICENSES/depth_anything_3.LICENSE", "src/third_party/LICENSES/addict.LICENSE"):
            self.assertIn(n, names)
        self.assertNotIn("src/third_party/depth_anything_3/logo.png", names)
        self.assertEqual(cfg["depth_tier1"]["depth_model"], "weights/da3-base")
        self.assertEqual(cfg["depth_tier1"]["segmenter"], "weights/sam2.1-hiera-small")
        self.assertEqual(cfg["depth_tier1"]["da3_src"], "src/third_party")
        self.assertEqual(cfg["generators"]["scene_qa"]["sam3_model"], "weights/sam3")
        with safe_open(stage / "weights" / "sam3" / "model.safetensors", framework="pt") as f:
            self.assertEqual(f.get_tensor("w").dtype, torch.bfloat16)
        with safe_open(stage / "weights" / "da3-base" / "model.safetensors", framework="pt") as f:
            self.assertEqual(f.get_tensor("w").dtype, torch.float32)
        self.assertEqual(json.loads((stage / "weights" / "sam3" / "config.json").read_text())["torch_dtype"],
                         "bfloat16")

    def test_rejects_unshipped_model_ref(self):
        args = self.gen_args()
        i = args.index("--sam3")
        del args[i:i + 2]
        code, err = self.pack(*args)
        self.assertEqual(code, 1)
        self.assertIn("'facebook/sam3' is not shipped", err)
        self.assertFalse(self.zip_path.exists())

    def test_vlm_layer_count_skips_demo_name_check(self):
        demo = self.root / "demo_config.json"
        demo.write_text(json.dumps({"num_vlm_layers": -1}))
        p = self.run_dir / "action_expert" / "config.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), "num_vlm_layers": 16}))
        save_file({"model.a": torch.zeros(4)}, self.run_dir / "action_expert" / "model.safetensors")
        code, err = self.pack("--demo-config", str(demo))
        self.assertEqual(code, 0, err)


if __name__ == "__main__":
    unittest.main()
