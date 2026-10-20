"""Vocabulary, natural language prompts, and negative distractors for zero-shot object detection."""
import re
from pathlib import Path

DEFAULT_VOCAB_FILE = Path(__file__).parent / "bridge_objects.txt"

# Natural language prompt ensembling templates for known robotic manipulation objects
SCENE_OBJECT_PROMPTS: dict[str, list[str]] = {
    # Cookware & Tableware
    "pot": ["pot", "cooking pot", "silver pot", "metal pot"],
    "pan": ["pan", "frying pan", "skillet", "saucepan"],
    "lid": ["lid", "pot lid", "pan lid", "cover"],
    "bowl": ["bowl", "silver bowl", "small bowl", "dish"],
    "plate": ["plate", "dinner plate", "saucer"],
    "cup": ["cup", "mug", "coffee cup"],
    "glass": ["glass", "a drinking glass", "glass cup"],
    "spoon": ["spoon", "blue spoon", "red spoon", "silver spoon"],
    "fork": ["fork", "blue fork", "red fork", "silver fork"],
    "knife": ["knife", "butter knife", "yellow knife"],
    "spatula": ["spatula", "turner", "green spatula"],
    "ladle": ["ladle", "soup ladle", "scoop"],
    "colander": ["colander", "strainer", "sieve"],
    "cooker": ["cooker", "rice cooker", "slow cooker"],

    # Produce & Food
    "eggplant": ["eggplant", "aubergine", "purple eggplant"],
    "pepper": ["pepper", "bell pepper", "green pepper", "capsicum"],
    "banana": ["banana", "yellow banana"],
    "apple": ["apple", "red apple", "green apple"],
    "carrot": ["carrot", "orange carrot"],
    "corn": ["corn", "corn cob", "yellow corn"],
    "potato": ["potato", "brown potato"],
    "tomato": ["tomato", "red tomato"],
    "cucumber": ["cucumber", "green cucumber"],
    "mushroom": ["mushroom", "brown mushroom"],
    "broccoli": ["broccoli", "green broccoli"],
    "strawberry": ["strawberry", "red strawberry"],
    "sushi": ["sushi", "sushi roll"],
    "egg": ["egg", "white egg", "brown egg"],
    "bread": ["bread", "loaf of bread"],
    "croissant": ["croissant", "pastry"],
    "cheese": ["cheese", "block of cheese"],
    "orange": ["orange", "fresh orange", "orange fruit"],

    # Cleaning & Textiles
    "cloth": ["cloth", "blue cloth", "yellow cloth", "orange cloth", "green cloth", "towel", "rag"],
    "towel": ["towel", "blue towel", "yellow towel", "green towel", "cloth"],
    "sponge": ["sponge", "cleaning sponge", "yellow sponge"],
    "napkin": ["napkin", "table napkin", "paper napkin"],
    "rag": ["rag", "cleaning rag", "cloth"],
    "brush": ["brush", "blue brush", "yellow brush", "cleaning brush"],

    # Containers, Fixtures & Appliances
    "can": ["can", "soda can", "tin can", "coke can", "pepsi can", "7up can"],
    "bottle": ["bottle", "plastic bottle", "water bottle"],
    "box": ["box", "cardboard box", "blue box", "yellow box"],
    "basket": ["basket", "white basket", "wire basket", "plastic basket"],
    "container": ["container", "plastic container", "food container"],
    "bag": ["bag", "plastic bag", "chip bag"],
    "chip bag": ["chip bag", "bag of chips", "green chip bag", "brown chip bag"],
    "sink": ["sink", "kitchen sink", "wash basin"],
    "burner": ["burner", "stove burner", "left burner", "right burner"],
    "stove": ["stove", "cooktop", "electric stove"],
    "drawer": ["drawer", "open drawer", "cabinet drawer"],
    "microwave": ["microwave", "microwave oven"],
    "hole": ["hole", "circular hole", "opening", "slot", "cup holder"],

    # Toys, Shapes & Blocks
    "block": ["block", "wooden block", "blue block", "yellow block"],
    "cube": ["cube", "yellow cube", "blue cube", "red cube", "green cube"],
    "triangle": ["triangle", "blue triangle", "green triangle", "yellow triangle", "red triangle"],
    "circle": ["circle", "red circle", "green circle", "yellow circle", "blue circle"],
    "star": ["star", "red star", "green star", "yellow star", "blue star"],
    "hexagon": ["hexagon", "yellow hexagon", "green hexagon", "blue hexagon", "red hexagon"],
    "heart": ["heart", "yellow heart", "red heart", "blue heart", "green heart"],
    "rectangle": ["rectangle", "blue rectangle", "yellow rectangle", "red rectangle"],
    "cylinder": ["cylinder", "wooden cylinder"],
    "ball": ["ball", "toy ball", "small ball"],
    "toy": ["toy", "plastic toy", "wooden toy"],
    "duck": ["duck", "toy duck", "rubber duck", "yellow duck"],
    "figure": ["figure", "toy figure", "plastic figure"],
}

# Negative distractor classes for competitive suppression of robot parts, surfaces, and shadows
DEFAULT_DISTRACTOR_QUERIES: list[str] = [
    "black metal robot gripper",
    "robot gripper fingers",
    "robot hand manipulator",
    "metal robot arm",
    "robot wrist",
    "empty marble countertop",
    "kitchen countertop surface",
    "background wall and cabinet",
    "empty floor",
    "dark shadow",
]


def load_vocabulary_from_file(vocab_path: Path | str | None = None) -> list[str]:
    """Reads unique object names from a text file (ignores comments and empty lines)."""
    path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_FILE
    if not path.exists():
        return list(SCENE_OBJECT_PROMPTS.keys())

    objects = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().lower()
            if not clean or clean.startswith("#"):
                continue
            if clean not in seen:
                seen.add(clean)
                objects.append(clean)
    return objects


def build_scene_queries(
    vocabulary: list[str] | None = None,
    include_distractors: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Builds a flattened query list and a mapping from each query back to its canonical name.

    If include_distractors is True, negative background classes and robot parts are appended and mapped to
    '__distractor__', enabling competitive suppression of false positives.
    """
    vocab = list(vocabulary) if vocabulary is not None else load_vocabulary_from_file()

    query_to_canonical: dict[str, str] = {}
    expanded_queries: list[str] = []

    for obj in vocab:
        c_norm = obj.strip().lower()
        if c_norm in SCENE_OBJECT_PROMPTS:
            variants = SCENE_OBJECT_PROMPTS[c_norm]
        else:
            article = "an" if c_norm.startswith(("a", "e", "i", "o", "u")) else "a"
            variants = [c_norm, f"{article} {c_norm}", f"the {c_norm}"]

        for v in variants:
            if v not in query_to_canonical:
                query_to_canonical[v] = c_norm
                expanded_queries.append(v)

    if include_distractors:
        for dist in DEFAULT_DISTRACTOR_QUERIES:
            if dist not in query_to_canonical:
                query_to_canonical[dist] = "__distractor__"
                expanded_queries.append(dist)

    return expanded_queries, query_to_canonical


def parse_task_objects(task: str) -> list[str]:
    """Extract candidate object names from natural language task prompt."""
    if not task:
        return []
    cleaned = re.sub(
        r"^(?:pick\s+up|take|place|put|move|push|pull|slide|insert|open|close|separate|nudge|add)\s+",
        "",
        task.strip(),
        flags=re.IGNORECASE,
    )
    parts = re.split(r",|\b(?:and|into|onto|to|from|on|in|under|near|beside|between)\b", cleaned, flags=re.IGNORECASE)
    candidates = []
    for p in parts:
        w = re.sub(r"^(?:the|a|an)\s+", "", p.strip(), flags=re.IGNORECASE).strip(" ._\"'")
        if len(w) > 2 and w not in candidates:
            candidates.append(w)
    return candidates
