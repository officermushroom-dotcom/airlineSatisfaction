"""Gated h-blend with a teacher mix. Reuse: add this notebook as an input, then `from gate_blend import *`."""
import numpy as np
from scipy.stats import rankdata

MAIN_W = np.array([0.1, 0.2, 0.3, 0.4])          # per-member weights (Nina's h-blend)
ORDER_C = np.array([0.07, -0.01, -0.02, -0.04])  # per-position corrections after sorting each row


def pct(v):
    return (rankdata(np.asarray(v, dtype=np.float64), method='average') - 0.5) / len(v)


def hblend(M, main=MAIN_W, corr=ORDER_C, mix=(0.3, 0.7)):
    """Nina's h-blend: sort each row both ways, weight by member + position, mix the two sides."""
    n, sides = len(M), []
    for rev in (False, True):
        order = np.argsort(-M if rev else M, axis=1, kind='stable')
        s = np.zeros(n)
        for j in range(M.shape[1]):
            k = order[:, j]
            s += M[np.arange(n), k] * (main[k] + corr[j])
        sides.append(s.astype(np.float32))
    return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)


def teacher_mix(files, weights):
    w = {k: v for k, v in weights.items() if v}
    return sum(files[k] * v for k, v in w.items()) / sum(w.values())


def gated_blend(members, teacher, weight=0.30, amp=0.02):
    """Kirill's gate: the teacher share moves by +/- amp/2 with the rank of member disagreement."""
    w = weight + amp * (pct(members.std(axis=1)) - 0.5)
    return pct(hblend((1 - w[:, None]) * members + w[:, None] * teacher[:, None]))
