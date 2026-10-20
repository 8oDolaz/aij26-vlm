"""Closed per-dataset object dictionaries for SAM 3 and the name helpers around them.

Dictionary format (dicts/<dataset>.txt): one object per line; names on one line separated by " / " are the same kind
of object (one group); "name = prompt" asks SAM 3 with `prompt` and labels the box `name`; "#" starts a comment.
Absent lists (dicts/<dataset>_absent.txt): one name per line, objects that should never be in that dataset's frames.
"""
import difflib
from dataclasses import dataclass
from pathlib import Path

DICT_DIR = Path(__file__).with_name("dicts")

# Not objects: surfaces, places and the robot itself.
SURFACES = {"table", "tabletop", "table top", "counter", "surface", "floor", "wall", "place", "position",
            "robot", "gripper", "arm", "robot arm", "robot gripper"}
# Names whose last word is one of these are dropped: too vague to ground ("red object") or unreliable.
DEFAULT_EXCLUDE = ("object", "thing", "item", "stuff", "something", "anything", "piece", "shelf", "objects", "things",
                   "items", "glass", "hole", "lid", "circle")
# A name made only of these is no object ("purple" from "purple piece of cloth"); "orange" is left out: it is a fruit.
COLOR_WORDS = {"red", "blue", "green", "yellow", "purple", "pink", "black", "white", "brown", "grey", "gray", "silver",
               "gold", "golden", "small", "big", "large", "little", "dark", "light", "tiny"}

# Last words whose plural is not the regular one.
PLURALS = {"knife": "knives", "leaf": "leaves", "loaf": "loaves", "shelf": "shelves", "fish": "fish",
           "sushi": "sushi", "bread": "bread", "corn": "corn", "broccoli": "broccoli", "grapes": "grapes",
           "cheese": "cheese", "potato": "potatoes", "tomato": "tomatoes", "mouse": "mice"}


def name_ok(name: str, exclude=DEFAULT_EXCLUDE) -> bool:
    """False for surfaces, excluded heads ("red object", "lid") and colour/size-only names ("purple")."""
    words = name.split()
    return (len(name) >= 3 and name not in SURFACES and words[-1] not in exclude
            and not all(w in COLOR_WORDS for w in words))


@dataclass(frozen=True)
class Entry:
    name: str    # label of the box
    prompt: str  # phrase SAM 3 is asked with
    group: str   # names on one dictionary line share it


def load_dictionary(path: Path, exclude=DEFAULT_EXCLUDE) -> dict[str, Entry]:
    """name -> Entry, in file order."""
    entries = {}
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.split("#", 1)[0].strip().lower()
        for part in filter(None, (p.strip() for p in line.split("/"))):
            name, _, prompt = (s.strip() for s in part.partition("="))
            if name_ok(name, exclude):
                entries[name] = Entry(name, prompt or name, f"{Path(path).stem}:{i}")
    return entries


def load_names(path: Path) -> list[str]:
    """One name per line, comments and blank lines skipped (absent lists)."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line and line not in names:
            names.append(line)
    return names


def map_to_dictionary(name: str, entries: dict[str, Entry]) -> str | None:
    """The dictionary entry an instruction name refers to: the name itself, else the longest entry it ends with
    ("small knife" -> knife), else an entry spelled almost like its last word ("laddle" -> ladle)."""
    if name in entries:
        return name
    words = name.split()
    tails = [e for e in entries if words[-len(e.split()):] == e.split()]
    if tails:
        return max(tails, key=lambda e: len(e.split()))
    close = difflib.get_close_matches(words[-1], [e for e in entries if " " not in e], n=1, cutoff=0.85)
    return close[0] if close else None


def similar(a: str, b: str, entries: dict[str, Entry]) -> bool:
    """Same name, same dictionary line ("pot / pan") or same last word ("coke can", "pepsi can")."""
    return a == b or entries[a].group == entries[b].group or a.split()[-1] == b.split()[-1]


def plural(name: str) -> str:
    """"coke can" -> "coke cans", "knife" -> "knives", "dish rack" -> "dish racks"."""
    *head, last = name.split()
    if last in PLURALS:
        last = PLURALS[last]
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return " ".join([*head, last])
