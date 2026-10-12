"""All question generators and how to build them from config.yaml."""
from aijgen.generators.base import Episode, Generator, Question
from aijgen.generators.phase_qa import PhaseGenerator

# name in config.yaml -> generator class
REGISTRY = {
    PhaseGenerator.name: PhaseGenerator,
}


def build_generators(config: dict) -> list[Generator]:
    """Create every generator that is enabled in config["generators"]."""
    return [REGISTRY[name](cfg)
            for name, cfg in config["generators"].items() if cfg["enabled"]]