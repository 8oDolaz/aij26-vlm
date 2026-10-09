import numpy as np


APPROACH = "approach"
CLOSING = "closing"
CONTACT = "contact"
OPENING = "opening"
RETREAT = "retreat"

PHASES = [APPROACH, CLOSING, CONTACT, OPENING, RETREAT]

def label_phases(g: np.ndarray, contact_border: float = 0.3, closing_border: float = 0.9, min_diff: float = 0.1) -> list[str]:
    """Phases from gripper (normalized per episode).

    Phases:
        approach  -- before the grasp, gripper still fully open
        closing   -- before the grasp, gripper already closing
        contact   -- from the first to the last frame where the gripper is
                     near its episode minimum (the robot holds the object)
        opening   -- after the grasp, gripper not fully open yet
        retreat   -- after the grasp, gripper fully open again
    
    Args:
        g: Gripper state per frame, shape (T,)
        contact_border: Normalized aperture at or below which a frame counts
            as contact. 
        closing_border: Normalized aperture below which the gripper is
            considered not fully open.
        min_diff: If max(g) - min(g) is smaller than this, the gripper
            never really closed.

    Returns:
        A list of T phase names, one per frame, each taken from PHASES.

    """
    gmin, gmax = g.min(), g.max()
    if gmax - gmin < min_diff:                       # gripper never really closed
        return [APPROACH] * len(g)

    #normalization
    g_norm = (g - gmin) / (gmax - gmin)             # 0 = most closed, 1 = most open

    contact = g_norm <= contact_border
    first_contact = int(np.argmax(contact))
    last_contact = len(g) - 1 - int(np.argmax(contact[::-1]))

    labels = []
    for t in range(len(g)):
        if first_contact <= t <= last_contact:
            labels.append(CONTACT)
        elif t < first_contact:
            labels.append(CLOSING if g_norm[t] < closing_border else APPROACH)
        else:
            labels.append(OPENING if g_norm[t] < closing_border else RETREAT)
    return labels