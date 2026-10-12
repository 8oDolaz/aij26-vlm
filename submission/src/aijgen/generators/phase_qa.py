"""Q1: which action phase is the robot in. One question per phase run."""
from aijgen import phases as ph
from aijgen.generators.base import Generator, Question, pick_frame

# answer option text for each phase
PHASE_TEXT = {
    "pre_grasp":      "Approaching the object with the gripper open",
    "immobilization": "Closing the gripper around the object",
    "contact":        "Holding the object in the closed gripper",
    "detach":         "Opening the gripper to release the object",
    "post_grasp":     "Moving away after releasing the object",
}

QUESTION = 'The robot is tasked to "{task}". What is the robot\'s current action phase?'

'''
номер	имя
0	pre_grasp
1	immobilization
2	contact
3	detach
4	post_grasp
'''

class PhaseGenerator(Generator):
    name = "phase_qa"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.frame_range = tuple(cfg["frame_range"])

        self.labeled_phases = set(cfg["labeled_phases"])
        unknown = self.labeled_phases - set(ph.PHASES)
        if unknown:
            raise ValueError(f"Unknown phases in config: {unknown}")

    def generate(self, ep, rng):
        # no gripper signal, or no real grasp
        if ep.labels is None or not ph.has_phases(ep.labels, {ph.IMMOBILIZATION, ph.CONTACT}):
            return [] 

        question = QUESTION.format(task=ep.task)
        out = []
        for phase, start, end in ph.runs(ep.labels):

            phase = ph.PHASES[phase]
            if phase not in self.labeled_phases:
                continue

            frame = pick_frame(start, end, rng, self.frame_range)

            correct = PHASE_TEXT[phase]
            others = [PHASE_TEXT[name] for name in ph.PHASES if name != phase]
            distractors = [others[i] for i in rng.choice(len(others), size=3, replace=False)]

            text, letter = self.mcq(question, correct, distractors, rng)
            out.append(Question(text, letter, [frame], self.name, {"phase": phase}))
        return out