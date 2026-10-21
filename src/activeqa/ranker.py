"""Reference-free learned selector for reformulation candidates.

The ranker is trained offline (``scripts/train_selector.py``) on candidate pools
collected while training on the *train* split. Gold answers are used only to
label that data; every feature consumed at test time is reference-free and
depends on the candidate question, the extracted answer and the span confidences
alone. The collected pools live in ``outputs/selector_dataset.jsonl``.
"""
import math

from .preprocessing import interrogatives, normalize, tokens
from .rewards import token_f1

FEATURE_NAMES = (
    "conf_delta",
    "conf_rank_frac",
    "conf_z",
    "consistency",
    "same_answer",
    "vote_frac",
    "question_similarity",
    "question_len_log_ratio",
    "answer_len_log_ratio",
    "valid",
    "interrogative_preserved",
    "ends_with_question_mark",
)


def _log_ratio(x, y):
    # soft-clip the log ratio to +-1.5 so degenerate tiny/huge rewrites cannot
    # produce an extreme feature that outweighs the confidence evidence.
    return max(-1.5, min(1.5, math.log(max(1, x) / max(1, y))))


def candidate_features(original_question, original_conf,
                       candidate, answer, confidence,
                       pool_answers, pool_confs, valid=True):
    """Reference-free features for one candidate rewrite.

    ``pool_answers``/``pool_confs`` describe the sampled-candidate pool (the
    original question is *not* part of the pool); the candidate itself must be
    included in it. ``conf_rank_frac`` and ``conf_z`` therefore measure where
    this candidate's span confidence sits among the offered choices — exactly
    the evidence a deployable selector can act on.
    """
    confidences = sorted(list(pool_confs) + [original_conf], reverse=True)
    rank = confidences.index(confidence) if confidence in confidences else len(confidences)
    conf_rank_frac = 1.0 - rank / max(1, len(confidences))
    mean_conf = sum(confidences) / len(confidences)
    spread = (sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)) ** 0.5
    conf_z = (confidence - mean_conf) / spread if spread > 1e-6 else 0.0

    original_tokens = tokens(original_question)
    candidate_tokens = tokens(candidate)

    return {
        "conf_delta": confidence - original_conf,
        "conf_rank_frac": conf_rank_frac,
        "conf_z": conf_z,
        "consistency": 0.0,   # filled by caller (needs the original answer)
        "same_answer": 0.0,   # filled by caller
        "vote_frac": 0.0,     # filled by caller
        "question_similarity": token_f1(candidate, [original_question]),
        "question_len_log_ratio": _log_ratio(len(candidate_tokens), len(original_tokens)),
        "answer_len_log_ratio": 0.0,  # filled by caller
        "valid": 1.0 if valid else 0.0,
        "interrogative_preserved": 1.0 if interrogatives(original_question).issubset(
            interrogatives(candidate)) else 0.0,
        "ends_with_question_mark": 1.0 if candidate.rstrip().endswith("?") else 0.0,
    }


def fill_answer_features(features, answer, original_answer, pool_answers):
    """Complete the answer-dependent features in place and return the dict."""
    answer_key = normalize(answer)
    features["consistency"] = token_f1(answer, [original_answer])
    features["same_answer"] = 1.0 if answer_key == normalize(original_answer) else 0.0
    features["vote_frac"] = (sum(1 for a in pool_answers if normalize(a) == answer_key)
                             / max(1, len(pool_answers)))
    features["answer_len_log_ratio"] = _log_ratio(len(tokens(answer)), len(tokens(original_answer)))
    return features


def feature_vector(features, names=FEATURE_NAMES):
    return [float(features.get(name, 0.0)) for name in names]


def dot(weights, vector, intercept):
    return intercept + sum(w * x for w, x in zip(weights, vector))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


class Ranker:
    """Loaded ranker: logistic weights over :data:`FEATURE_NAMES`."""

    def __init__(self, weights):
        self.names = tuple(weights["names"])
        if self.names != FEATURE_NAMES:
            raise ValueError("selector weights do not match the feature schema")
        self.coef = [float(w) for w in weights["coef"]]
        self.intercept = float(weights.get("intercept", 0.0))
        self.threshold = float(weights.get("switch_threshold", 0.5))
        self.meta = weights.get("meta", {})

    @classmethod
    def load(cls, path):
        import json
        with open(path, encoding="utf-8") as stream:
            return cls(json.load(stream))

    def probability(self, vector):
        return sigmoid(dot(self.coef, vector, self.intercept))

    def choose(self, probabilities):
        """Return (index, probability) of the top candidate, or None."""
        if not probabilities:
            return None
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return index, probabilities[index]
