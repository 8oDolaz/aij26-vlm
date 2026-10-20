"""Prompt formats copied from the benches' own prompts (runs/base_model_eval/tasks/*/samples*.jsonl).

Every multiple-choice answer is a bare letter (docs/vlm_data_priorities.md #0). Option order is shuffled by
`shuffle`, so the correct letter is uniform over each question's own options.
"""
import numpy as np

LETTERS = "ABCDEFGHIJ"


def note(n: int) -> str:
    """lmms-eval's BLINK / EmbSpatial pre-prompt: n=4 -> "... respond with A, B, C, D without ..."."""
    return f"Note: You only need to respond with {', '.join(LETTERS[:n])} without providing any additional information.\n"


def shuffle(correct: str, distractors: list[str], rng: np.random.Generator) -> tuple[list[str], str]:
    """(options in random order, letter of the correct one)."""
    options = [correct, *distractors]
    options = [options[i] for i in rng.permutation(len(options))]
    return options, LETTERS[options.index(correct)]


def embspatial(question: str, options: list[str]) -> str:
    """'Note: ...\\n<question>\\nA) table\\nB) towel ...'"""
    return note(len(options)) + question + "".join(f"\n{LETTERS[i]}) {o}" for i, o in enumerate(options))


def blink(question: str, options: list[str], choices_line: bool = True) -> str:
    """'Note: ...\\n<question>\\nSelect from the following choices.\\n(A) yes\\n(B) no'. Object Localization puts the
    choices after "Select from the following options." inside its own question, so choices_line=False there."""
    lead = "\nSelect from the following choices." if choices_line else ""
    return note(len(options)) + question + lead + "".join(f"\n({LETTERS[i]}) {o}" for i, o in enumerate(options))


ERQA_SUFFIX = (" Please answer directly with only the letter of the correct option and nothing else.\n"
               "Answer with the single letter corresponding to the correct option. Do not explain. "
               "Your answer must be one of A, B, C, or D.")  # identical on all 400 ERQA questions


def erqa(question: str, options: list[str]) -> str:
    """'<question> Choices: A. red dot. B. pink dot. ... Please answer directly ...'"""
    return question + " Choices: " + " ".join(f"{LETTERS[i]}. {o}." for i, o in enumerate(options)) + ERQA_SUFFIX


CROSSPOINT_PRE = ("I am providing you with two images of the same scene from different viewpoints. A red point is marked on "
                  "the first image. You are given multiple points on the second image.")
CROSSPOINT_JUDGE_Q = [  # the three wordings of CrossPoint Correspondence-Judgement (57 / 51 / 48 of its 156 items)
    "The point in the first image correspond to which marker in the second image?",
    "Which point in the second image matches the same 3d point as the red point in the first image?",
    "Which point on the second image corresponds to the point in the first image?",
]


def crosspoint(question: str, options: list[str]) -> str:
    """'I am providing you ... second image. <question>\\nA. Green dot  B. Blue dot  C. Yellow dot'"""
    return f"{CROSSPOINT_PRE} {question}\n" + "  ".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


MSMU_CONCISE ="Answer the question based only on the image. Give a concise answer."
MSMU_NUMERIC = "Answer the question based only on the image. If the answer is numeric, output only the number and unit if necessary."


def msmu(question: str, numeric: bool = False) -> str:
    """'Question: <question>\\nAnswer the question based only on the image. ...' (free-form answer)."""
    return f"Question: {question}\n{MSMU_NUMERIC if numeric else MSMU_CONCISE}"


BLINK_LOCALIZATION = (
    "A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes should touch the "
    "outermost pixels of the object that is being labeled. Given the two bounding boxes on the image, labeled by A and "
    "B, which bounding box more accurately localizes and encloses the {name}? Select from the following options.")
