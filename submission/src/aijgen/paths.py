"""Model references in config.yaml: Hugging Face ids in development, paths under the submission root in the packed ZIP.

A relative reference that exists under the submission root (the dir holding generate_dataset.py) wins, so the ZIP can
ship the weights next to the code and run offline from any working directory.
"""
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[2]


def resolve(ref: str | None) -> str | None:
    """`ref` as a path under the submission root if one exists there, else unchanged."""
    if not ref or Path(ref).is_absolute():
        return ref
    local = SUBMISSION_DIR / ref
    return str(local) if local.exists() else ref
