"""Instruction parsing: object noun phrases (for the SAM 3 probe) and the strict pick / place parse the task-success
questions swap objects in (docs/sam3_vqa_plan.md, Q08 hard negatives).

parse_task only accepts instructions whose outcome can be checked on the image:
  place  "<verb> [the] X on|onto|in|into|inside|on top of [the] Y"   X ends on or in Y
  pick   "pick up|pick|lift|grab [the] X"                           X ends in the gripper
Relational targets ("near", "to the left of") and phrases with extra clauses ("from the drawer", "and ...") are
rejected: whether another object already satisfies them cannot be checked. X and Y must each name exactly one object
of the dataset dictionary.
"""
import re
from dataclasses import dataclass

from aijgen.scene.vocab import DEFAULT_EXCLUDE, Entry, map_to_dictionary, name_ok, similar

# ---------------------------------------------------------------------------------------------------------------------
# Object noun phrases: "take the eggplant and put it between the two right burners" -> ["eggplant", "burner"]
# ---------------------------------------------------------------------------------------------------------------------
VERBS = {
    "pick", "picked", "lift", "lifted", "put", "place", "placed", "set", "move", "moved", "take", "took", "taken",
    "grab", "grabbed", "get", "got", "bring", "brought", "open", "opened", "close", "closed", "fold", "folded",
    "unfold", "unfolded", "push", "pushed", "pull", "pulled", "slide", "slid", "flip", "flipped", "turn", "turned",
    "knock", "knocked", "sweep", "swept", "wipe", "wiped", "stack", "stacked", "remove", "removed", "insert", "inserted",
    "drop", "dropped", "rotate", "rotated", "separate", "separated", "nudge", "nudged", "add", "added", "transfer",
    "transferred", "reposition", "repositioned", "pour", "poured", "hang", "hung", "cover", "covered", "uncover",
    "uncovered"}
PARTICLES = {"up", "down", "over", "out", "off", "back", "away"}
PIECE_OF = re.compile(r"\b(?:piece|pieces|bunch|pair|sheet)\s+of\b")
SPLIT = re.compile(r"[,;.]|\b(?:and|then|into|onto|in|on|to|from|of|at|by|near|beside|between|under|underneath|"
                   r"inside|above|below|along|against|with|toward|towards|across|around|behind|next|so|it|them)\b")
DROP = {"the", "a", "an", "two", "three", "some", "its", "their", "this", "that", "other", "another", "left", "right",
        "top", "bottom", "middle", "center", "centre", "front", "back", "upper", "lower", "side", "edge", "corner",
        "upright", "one", "ones"} | VERBS | PARTICLES
MAX_TASK_WORDS = 20  # longer "instructions" are mostly captions or junk ("Look at the sequence of images...")
IRREGULAR = {"knives": "knife", "shelves": "shelf", "leaves": "leaf", "loaves": "loaf"}


def is_verb_form(word: str) -> bool:
    """folds, pushes, putting, moving: inflections of VERBS that the word list does not spell out."""
    stems = {word.removesuffix("s"), word.removesuffix("es")}
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        stems |= {stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem}
    return bool(stems & VERBS)


def singular(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and re.search(r"(ches|shes|xes|sses)$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def instruction_names(task: str, exclude=DEFAULT_EXCLUDE, max_words: int = 4) -> list[str]:
    """Object noun phrases of an instruction, in order, without verbs, articles, positions, surfaces or `exclude`d
    heads."""
    names = []
    for chunk in SPLIT.split(PIECE_OF.sub(" ", task.lower())):
        words = [w for w in re.sub(r"[^a-z0-9 ]", " ", chunk).split() if w not in DROP and not is_verb_form(w)]
        if not words:
            continue
        words[-1] = singular(words[-1])
        name = " ".join(words[-max_words:])
        if name_ok(name, exclude) and name not in names:
            names.append(name)
    return names


# ---------------------------------------------------------------------------------------------------------------------
# Strict pick / place parse for task-success questions
# ---------------------------------------------------------------------------------------------------------------------
PLACE_VERBS = r"put|place|placed|move|moved|take|transfer|set|pick up|picked up"
PICK_VERBS = r"pick up|picked up|pick|lift|grab"
PREPS = r"on top of|onto|on|into|inside|in"
PHRASE = r"(?:(?:the|a|an)\s+)?[a-z0-9][a-z0-9' -]*?"
PLACE_RE = re.compile(rf"^(?P<verb>{PLACE_VERBS})\s+(?P<obj>{PHRASE})\s+(?:and\s+(?:put|place)\s+it\s+)?"
                      rf"(?P<prep>{PREPS})\s+(?P<tgt>{PHRASE})$")
PICK_RE = re.compile(rf"^(?P<verb>{PICK_VERBS})\s+(?P<obj>{PHRASE})$")
# words that mean a phrase carries an extra clause or a relation; such instructions are not parsed
CLAUSE = {"and", "then", "from", "of", "to", "near", "with", "at", "by", "beside", "between", "under", "behind",
          "next", "left", "right", "front", "it", "them", "onto", "into", "on", "in", "inside", "towards", "toward"}
GERUND = {"put": "putting", "place": "placing", "placed": "placing", "move": "moving", "moved": "moving",
          "take": "taking", "transfer": "transferring", "set": "setting", "pick up": "picking up",
          "picked up": "picking up", "pick": "picking up", "lift": "lifting", "grab": "grabbing"}


@dataclass(frozen=True)
class Task:
    text: str                  # the instruction as parsed: stripped, trailing period removed
    kind: str                  # "place" or "pick"
    verb: str
    obj: str                   # dictionary name of the moved object
    target: str | None         # dictionary name of the place (place tasks)
    span: tuple[int, int]      # character span of the object phrase (with its article) in `text`

    def swapped(self, name: str) -> str:
        """The instruction with the object phrase replaced by "the <name>" (by "<name>" if the original phrase has
        no article: "pick coke can" -> "pick pepsi can")."""
        s, e = self.span
        article = "the " if re.match(r"(?:the|a|an)\s", self.text[s:e].lower()) else ""
        return f"{self.text[:s]}{article}{name}{self.text[e:]}"

    def gerund(self, text: str | None = None) -> str | None:
        """"put the spoon on the towel" -> "putting the spoon on the towel" (for "Is the robot successful in ...?"),
        or None if the verb has no clean -ing form here, or a second verb follows ("pick up X and put it in Y")."""
        text = text or self.text
        g = GERUND.get(self.verb.lower())
        if g is None or re.search(r"\band\b", text.lower()):
            return None
        return g + text[len(self.verb):]


def _one_name(phrase: str, entries: dict[str, Entry]) -> str | None:
    """The single dictionary object a phrase names, or None (no object, several objects, or a clause in it)."""
    words = re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split()
    if not words or CLAUSE & set(words):
        return None
    names = instruction_names(phrase)
    if len(names) != 1:
        return None
    return map_to_dictionary(names[0], entries)


def parse_task(task: str, entries: dict[str, Entry]) -> Task | None:
    """The pick / place structure of an instruction (module docstring), or None."""
    text = task.strip().rstrip(".").strip()
    low = text.lower()
    if not text or len(text.split()) > MAX_TASK_WORDS:
        return None
    m = PLACE_RE.match(low)
    if m:
        obj, tgt = _one_name(m["obj"], entries), _one_name(m["tgt"], entries)
        if obj is None or tgt is None or similar(obj, tgt, entries):
            return None
        return Task(text, "place", text[:m.end("verb")], obj, tgt, m.span("obj"))
    m = PICK_RE.match(low)
    if m:
        obj = _one_name(m["obj"], entries)
        if obj is None:
            return None
        return Task(text, "pick", text[:m.end("verb")], obj, None, m.span("obj"))
    return None
