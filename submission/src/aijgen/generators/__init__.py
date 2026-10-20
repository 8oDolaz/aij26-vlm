"""All question generators and how to build them from config.yaml."""
from pathlib import Path

from aijgen.generators.base import Episode, Generator, Question
from aijgen.generators.phase_qa import PhaseGenerator
from aijgen.generators.scene_qa import SceneQAGenerator
from aijgen.generators.task_success import TaskSuccessGenerator

# name in config.yaml -> generator class
REGISTRY = {
    PhaseGenerator.name: PhaseGenerator,
    SceneQAGenerator.name: SceneQAGenerator,
    TaskSuccessGenerator.name: TaskSuccessGenerator,
}


def build_generators(config: dict, work_root: Path | None = None) -> list[Generator]:
    """Create every generator that is enabled in config["generators"]. With work_root, each one gets
    cfg["work_dir"] = <work_root>/<name> unless its config sets one."""
    gens = []
    for name, cfg in config["generators"].items():
        if cfg.get("enabled", True):
            if work_root is not None:
                cfg = {"work_dir": str(Path(work_root) / name), **cfg}
            gens.append(REGISTRY[name](cfg))
    return gens
