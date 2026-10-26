import sys
sys.path.insert(0,'src')
from activeqa.preprocessing import normalize,tokens,entity_signature,interrogatives
from activeqa.rewards import token_f1,reward
from activeqa.policy_gradient import RunningBaseline
from activeqa.policy_gradient import reinforce_step,reinforce_batch_step,sft_batch_step
from activeqa.data import _load_indicqa_json
from activeqa.qa_model import _context_token_mask
from activeqa.selection import select_qa_score
import json, tempfile, os
import pytest

def test_unicode_normalization_and_tokens():
    assert normalize('  తెలుగు\nప్రశ్న ')=='తెలుగు ప్రశ్న'
    assert tokens('తెలుగు ప్రశ్న')==['తెలుగు','ప్రశ్న']

def test_f1_and_reward():
    assert token_f1('హైదరాబాద్', ['హైదరాబాద్'])==1
    assert reward('x',['x'])>0

def test_entity_signatures():
    assert entity_signature('ABC 2024').issubset(entity_signature('ABC 2024 వచ్చింది'))

def test_baseline_advantage_nonzero_first_sample():
    b=RunningBaseline(.9)
    assert b.advantage(.5)==.5

def test_reinforce_step_updates_trainable_parameter():
    import torch
    policy = torch.nn.Parameter(torch.tensor(0.25))
    optimizer = torch.optim.SGD([policy], lr=0.1)
    old = policy.detach().clone()
    loss, advantage = reinforce_step(policy, 1.0, RunningBaseline(), optimizer,
                                     [policy], clip=1.0)
    assert loss < 0
    assert advantage == 1.0
    assert policy.grad is not None
    assert not torch.equal(old, policy.detach())

def test_context_mask_matches_model_input_device():
    import torch
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        encoded = {"token_type_ids": torch.tensor([[0, 0, 1, 1]])}
        inputs = {"attention_mask": torch.ones((1, 4), device=device)}
        mask = _context_token_mask(encoded, inputs, 0, device)
        assert mask.device == device
        assert mask.tolist() == [False, False, True, True]


def test_majority_vote_prefers_repeated_answer_over_original():
    from activeqa.selection import select_majority_vote

    # One rewrite yields the (hypothetically better) answer that three of the
    # four sampled rewrites agree on; the lone-vote answer stays rejected.
    class FakeQA:
        def answer_with_score(self, question, context):
            answer = {"original": "A", "c1": "B", "c2": "B", "c3": "B",
                      "c4": "C"}[question]
            score = {"original": -1.0, "c1": -2.0, "c2": -2.0, "c3": -2.0,
                     "c4": -0.1}[question]
            return answer, 0.0, score

    chosen = select_majority_vote(["original", "c1", "c2", "c3", "c4"],
                                  FakeQA(), [{"context": "ctx"}])
    assert chosen in ("c1", "c2", "c3")


def test_majority_vote_keeps_original_when_answers_diverge():
    from activeqa.selection import select_majority_vote

    class FakeQA:
        def answer_with_score(self, question, context):
            answer = {"original": "A", "c1": "B", "c2": "C", "c3": "D",
                      "c4": "E"}[question]
            return answer, 0.0, {"original": -1.0}.get(question, -0.1)

    chosen = select_majority_vote(["original", "c1", "c2", "c3", "c4"],
                                  FakeQA(), [{"context": "ctx"}])
    assert chosen == "original"


def test_ranker_features_are_reference_free_and_sane():
    import math
    from activeqa.ranker import (FEATURE_NAMES, Ranker, candidate_features,
                                 feature_vector, fill_answer_features)
    question = "కులశేఖర పాండ్య రాజ్యం ఏ శతాబ్దంలో?"
    pool_answers = ["6వ", "9వ"]
    pool_confs = [-2.0, -1.0]

    feats = candidate_features(
        original_question=question, original_conf=-1.0,
        candidate="పాండ్య రాజ్యం ఏ శతాబ్దం అభివృద్ధి చెందింది?",
        answer="9వ", confidence=-1.0,
        pool_answers=pool_answers, pool_confs=pool_confs)
    fill_answer_features(feats, "9వ", "6వ", pool_answers)
    vector = feature_vector(feats)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)
    assert feats["conf_delta"] == 0.0
    # the candidate owns the best confidence in the pool => top rank
    assert feats["conf_rank_frac"] == 1.0
    assert feats["same_answer"] == 0.0
    assert abs(feats["vote_frac"] - 0.5) < 1e-9
    assert feats["ends_with_question_mark"] == 1.0
    assert 0.0 < feats["question_similarity"] < 1.0


def test_select_ranker_switches_only_above_threshold():
    from activeqa.ranker import FEATURE_NAMES, Ranker
    from activeqa.selection import select_ranker

    class FakeQA:
        def answer_with_score(self, question, context):
            table = {"orig": ("a", -0.6), "weak": ("a", -2.0), "strong": ("b", -0.5)}
            answer, conf = table[question]
            return answer, 0.0, conf

    # Only conf_delta carries weight (positive slope).
    weights = {"names": list(FEATURE_NAMES),
               "coef": [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
               "intercept": 0.0, "switch_threshold": 0.9}
    ranker = Ranker(weights)

    # strong: conf_delta=+0.1 -> p ~= 0.525, below the 0.9 gate -> keep original.
    assert select_ranker(["orig", "strong"], FakeQA(), [{"context": "c"}], ranker) == "orig"
    # Lower the gate and the confident rewrite wins.
    ranker.threshold = 0.5
    assert select_ranker(["orig", "strong"], FakeQA(), [{"context": "c"}], ranker) == "strong"
    # weak rewrite (conf_delta=-1.4) never clears any positive threshold.
    assert select_ranker(["orig", "weak"], FakeQA(), [{"context": "c"}], ranker) == "orig"


def test_train_selector_calibrates_on_synthetic_pools(tmp_path):
    import json
    sys.path.insert(0, "scripts")
    import train_selector

    def build_record(i, better_conf):
        # the candidate answers the gold answer only when its confidence is
        # higher; the ranker must learn conf_delta as the deciding feature.
        return {
            "id": str(i), "epoch": 1,
            "question": "q", "original_answer": "a", "original_confidence": -1.0,
            "original_f1": 0.2,
            "candidates": ["orig", "cand"],
            "validity": [True, True],
            "answers": ["a", "gold" if better_conf else "wrong"],
            "confidences": [-1.0, -0.8 if better_conf else -1.2],
            "f1s": [0.2, 0.9 if better_conf else 0.1],
        }

    records = [build_record(i, better_conf=(i % 2 == 0)) for i in range(60)]
    dataset = tmp_path / "selector_dataset.jsonl"
    with open(dataset, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    import train_selector
    out = tmp_path / "weights.json"
    sys_argv = sys.argv
    sys.argv = ["train_selector.py", "--dataset", str(dataset),
                "--out", str(out), "--splits", "3"]
    try:
        train_selector.main()
    finally:
        sys.argv = sys_argv
    weights = json.load(open(out))
    assert weights["switch_threshold"] > 0
    conf_delta_index = weights["names"].index("conf_delta")
    assert weights["coef"][conf_delta_index] > 0


def test_qa_score_selection_uses_confidence_not_nonempty_answers():
    class FakeQA:
        def answer_with_score(self, question, context):
            return "some answer", 0.0, {"low confidence": -2.0, "high confidence": -0.2}[question]

    chosen = select_qa_score(["low confidence", "high confidence"], FakeQA(),
                             [{"context": "context"}])
    assert chosen == "high confidence"


def test_reformulator_rejects_generation_without_eos():
    from activeqa.reformulator import Reformulator
    reformulator = object.__new__(Reformulator)
    assert not reformulator._valid("ప్రశ్న 2024", "ప్రశ్న 2024", terminated=False)


def test_interrogatives_classify_telugu_question_type():
    assert interrogatives("ఇది ఎక్కడ ఉంది?") == {"where"}
    assert interrogatives("ఇది ఎందుకు ఉంది?") == {"why"}
    assert interrogatives("ఎవరు ముఖ్యమంత్రి?") == {"who"}


def test_reformulator_rejects_question_type_change():
    from activeqa.reformulator import Reformulator
    reformulator = object.__new__(Reformulator)
    assert not reformulator._valid("ఇది ఎక్కడ ఉంది?", "ఇది ఎందుకు ఉంది?")
    assert reformulator._valid("ఇది ఎక్కడ ఉంది?", "ఇది ఎక్కడ ఉంటుంది?")


def test_sft_batch_step_updates_trainable_parameter():
    import torch
    policy = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    optimizer = torch.optim.SGD([policy], lr=0.1)
    old = policy.detach().clone()
    loss = sft_batch_step([torch.log(policy[0]), torch.log(policy[1])],
                          optimizer, [policy], clip=1.0)
    assert loss == loss  # finite
    assert policy.grad is not None
    assert not torch.equal(old, policy.detach())


def test_reinforce_batch_step_updates_trainable_parameter():
    import torch
    policy = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    optimizer = torch.optim.SGD([policy], lr=0.1)
    old = policy.detach().clone()
    loss = reinforce_batch_step([torch.log(policy[0]), torch.log(policy[1])],
                                [1.0, 0.0], optimizer, [policy], clip=1.0)
    assert loss == loss  # finite
    assert policy.grad is not None
    assert not torch.equal(old, policy.detach())


def test_sequence_logprob_batch_masks_uneven_padding():
    import torch
    from types import SimpleNamespace
    from activeqa.reformulator import Reformulator
    reformulator = object.__new__(Reformulator)
    reformulator.device = torch.device("cpu")
    reformulator.tokenizer = SimpleNamespace(pad_token_id=0)
    reformulator._source = lambda question, max_input: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, labels):
            batch, seq = labels.shape
            return SimpleNamespace(logits=torch.zeros((batch, seq, 5)))

    reformulator.model = FakeModel()
    logprobs = reformulator.sequence_logprob_ids_batch(
        "ప్రశ్న", [[1, 2], [1, 2, 3, 4]], max_input=96)
    assert logprobs.shape == (2,)
    expected = torch.tensor([-2.0, -4.0]) * torch.log(torch.tensor(5.0))
    assert torch.allclose(logprobs, expected, atol=1e-6)


def test_evaluation_candidate_choice_does_not_use_gold_answers(tmp_path):
    from activeqa.evaluation import evaluate

    class FakeReformulator:
        def sample_actions(self, *args):
            return [
                {"sampled_text": "high confidence", "candidate": "high confidence", "valid": True},
                {"sampled_text": "low confidence", "candidate": "low confidence", "valid": True},
            ]

    class FakeQA:
        def answer_with_score(self, question, context):
            answer = "gold answer" if question == "low confidence" else "wrong answer"
            score = {"question": -3.0, "high confidence": -0.2,
                     "low confidence": -2.0}[question]
            return answer, 0.0, score

        def answer(self, question, context):
            return self.answer_with_score(question, context)[:2]

    class EmptyRetriever:
        def retrieve(self, *args):
            return []

    config = {"generation": {"candidates_per_question": 2,
                             "max_input_tokens": 96, "max_new_tokens": 64},
              "retrieval": {"top_k": 3, "max_context_chars": 100}}
    rows = [{"id": "1", "question": "question", "context": "supporting context",
             "answers": ["gold answer"]}]
    result = evaluate(rows, FakeReformulator(), FakeQA(), EmptyRetriever(), config,
                      str(tmp_path / "eval.json"))
    with open(tmp_path / "eval.json", encoding="utf-8") as stream:
        saved = json.load(stream)
    row = saved["predictions"][0]
    assert row["qa_selected_question"] == "high confidence"
    assert row["oracle_question"] == "low confidence"
    assert result["context_protocol"].startswith("primary metrics use")


def test_qa_selector_can_abstain_to_original_question():
    class FakeQA:
        def answer_with_score(self, question, context):
            answer = {"original": "Bose", "bad reformulation": "Gandhi"}[question]
            score = {"original": -1.2, "bad reformulation": -0.1}[question]
            return answer, 0.0, score

    assert select_qa_score(["original", "bad reformulation"], FakeQA(),
                           [{"context": "context"}]) == "original"


def test_flatten_squad_style_indicqa_json():
    raw={"data":[{"paragraphs":[{"context":" మద్రాస్ ","qas":[{"id":"te1","question":" ఎక్కడ? ","answers":[{"text":"చెన్నై","answer_start":0}]}]}]}]}
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",delete=False) as f:
        json.dump(raw,f,ensure_ascii=False); path=f.name
    try:
        rows=_load_indicqa_json(path)
        assert rows==[{"id":"te1","context":"మద్రాస్","question":"ఎక్కడ?","answers":["చెన్నై"]}]
    finally: os.unlink(path)


def test_retrieval_can_use_full_disjoint_train_partition(monkeypatch, tmp_path):
    from activeqa.data import load_splits
    import huggingface_hub

    rows = [{"id": str(i), "question": f"ప్రశ్న {i}", "context": f"సందర్భం {i}",
             "answers": [{"text": f"సమాధానం {i}"}]} for i in range(10)]
    payload = {"data": [{"paragraphs": [{"context": row["context"], "qas": [{
        "id": row["id"], "question": row["question"], "answers": row["answers"]
    }]} for row in rows]}]}
    path = tmp_path / "indicqa.te.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))

    train, heldout, retrieval_corpus = load_splits({
        "dataset": "fake", "config": "indicqa.te", "eval_fraction": 0.2,
        "train_examples": 2, "eval_examples": 2,
    }, include_full_train=True)
    assert len(train) == 2
    assert len(heldout) == 2
    assert len(retrieval_corpus) == 8
    assert {row["id"] for row in heldout}.isdisjoint(row["id"] for row in retrieval_corpus)


def test_reformulator_accepts_real_sft_paraphrases():
    """The RL validity gate must not reject genuine SFT output.

    These are raw candidates from the verified SFT run. The old gate required
    55% of all content words to survive and only accepted known==known answer
    types, which rejected 4 of these 5 paraphrases and silently reduced the RL
    update to a no-op.
    """
    from activeqa.reformulator import Reformulator
    reformulator = object.__new__(Reformulator)
    pairs = [
        ("ఎక్కడికక్కడ భక్తులు ఇబ్బంది పడకుండా అన్ని ఏర్పాట్లూ చేసింది ప్రభుత్వం?",
         "భక్తులకు ఎలాంటి ఇబ్బంది కలగకుండా ప్రభుత్వం అన్ని ఏర్పాట్లు చేసింది?"),
        ("దీనిపై ఎందుకు సమాధానం చెప్పరని ప్రశ్నించారు?",
         "ఎందుకు ఈ ప్రశ్నకు సమాధానం చెప్పలేము?"),
        ("పర అంటే ఏమిటో నాకు నిక్కచ్చిగా తెలియదు?",
         "నాకు తెలియనిది ఏంటో నాకు తెలీదు?"),
        ("ఈ విషయంలో జోక్యం చేసుకునే యోచన ఉందా?",
         "ఈ విషయంలో మీరు జోక్యం చేసుకోవాలనుకుంటున్నారా?"),
    ]
    for question, candidate in pairs:
        assert reformulator._valid(question, candidate), candidate


def test_qa_eval_reports_conservative_and_corrective_selection():
    """The two selectors must both be measured, and both must be safe.

    Question 1: the rewrite keeps an answer consistent with the original's
    (partial) answer, so the conservative rule may switch on confidence.

    Question 2: the original answer is wrong, so the only helpful rewrite
    changes the answer. The conservative rule must abstain there; the
    confidence-override rule must take it. A higher-confidence invalid
    candidate must never win in either rule.
    """
    from activeqa.qa_eval import evaluate_qa_pairs

    class FakeReformulator:
        def sample_actions(self, question, candidates, *args, **kwargs):
            table = {
                "q_consistency": [
                    {"sampled_text": "delhi government announced",
                     "candidate": "delhi government announced", "valid": True},
                    {"sampled_text": "q_consistency",
                     "candidate": "q_consistency", "valid": False},
                ],
                "q_corrective": [
                    {"sampled_text": "gold", "candidate": "gold", "valid": True},
                    {"sampled_text": "gold", "candidate": "gold", "valid": False},
                ],
            }
            return table[question]

    class FakeQA:
        def answer_batch(self, questions, context):
            table = {
                # Partial credit, low confidence.
                "q_consistency": ("delhi government", 0.0, -1.0),
                # Consistent successor answer, more confident.
                "delhi government announced": ("delhi government announced", 0.0, -0.4),
                # Wrong answer, low confidence: this is what reformulation is for.
                "q_corrective": ("wrong", 0.0, -1.0),
                "gold": ("gold", 0.0, -0.1),
            }
            return [table.get(question, ("wrong", 0.0, -1.0)) for question in questions]

    summary, records = evaluate_qa_pairs(
        FakeReformulator(), FakeQA(),
        [{"id": "1", "question": "q_consistency", "context": "c",
          "answers": ["delhi government announced"]},
         {"id": "2", "question": "q_corrective", "context": "c",
          "answers": ["gold"]}],
        candidates=2, limit=2, label="test")

    by_id = {record["id"]: record for record in records}
    assert by_id["1"]["switched"] is True
    assert by_id["1"]["selected_question"] == "delhi government announced"
    assert by_id["1"]["selected_f1"] == pytest.approx(1.0)
    assert by_id["1"]["original_f1"] == pytest.approx(0.8)

    # Conservative rule refuses the answer-changing rewrite.
    assert by_id["2"]["switched"] is False
    assert by_id["2"]["selected_question"] == "q_corrective"
    assert by_id["2"]["selected_f1"] == 0.0
    # Corrective rule takes it, and the invalid candidate still loses.
    assert by_id["2"]["corrective_switch"] is True
    assert by_id["2"]["corrective_question"] == "gold"
    assert by_id["2"]["corrective_f1"] == pytest.approx(1.0)

    assert summary["switch_rate"] == pytest.approx(0.5)
    # The corrective rule switches on both questions: Q1 for the same
    # consistent reason as the conservative rule, Q2 via the override.
    assert summary["corrective_switch_rate"] == pytest.approx(1.0)
    assert by_id["1"]["corrective_question"] == "delhi government announced"
    assert summary["original_f1"] == pytest.approx(0.4)
    assert summary["selected_f1"] == pytest.approx(0.5)
    assert summary["corrective_f1"] == pytest.approx(1.0)
    assert summary["improved_rate"] == pytest.approx(0.5)
    assert summary["harmed_rate"] == 0.0
    assert summary["valid_generation_rate"] == pytest.approx(0.5)
    assert summary["distinct_selected_ratio"] == pytest.approx(1.0)


def test_rft_takes_an_update_only_for_qualifying_rewrites():
    import torch
    from activeqa.rft import rft_on_policy

    class FakeRef:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            torch.nn.init.constant_(self.model.weight, 1.0)
            torch.nn.init.constant_(self.model.bias, 0.0)
            self.calls = 0

        def sample_actions_batch(self, questions, candidates, *args, **kwargs):
            self.calls += 1
            return [[{"valid": True, "sampled_text": "rewrite", "candidate": "rewrite",
                      "token_ids": [1, 2]} for _ in range(candidates)]
                    for _ in questions]

        def sequence_logprob_batch(self, pairs, max_input):
            return torch.stack([self.model(torch.ones(1, 1))[0, 0] for _ in pairs])

    class FakeQA:
        def answer_batch(self, questions, context):
            return [("gold", 0.0, -0.5) for _ in questions]

    ref = FakeRef()
    before = ref.model.weight.detach().clone()
    metrics, records = rft_on_policy(
        ref, FakeQA(),
        [{"question": "q", "context": "c", "answers": ["gold"]}],
        candidates=2, batch_questions=1, epochs=1, learning_rate=1e-2,
        generate_batch=1, log=lambda *_: None)
    assert metrics["optimizer_steps"] == 1
    assert metrics["selected_rate"] == 1.0
    assert len(records) == 1
    assert not torch.equal(before, ref.model.weight.detach())


def test_rft_skips_when_no_rewrite_beats_the_original():
    import torch
    from activeqa.rft import rft_on_policy

    class FakeRef:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.calls = 0

        def sample_actions_batch(self, questions, candidates, *args, **kwargs):
            self.calls += 1
            return [[{"valid": True, "sampled_text": "rewrite", "candidate": "rewrite",
                      "token_ids": [1]} for _ in range(candidates)]
                    for _ in questions]

        def sequence_logprob_batch(self, pairs, max_input):
            raise AssertionError("no update should be attempted")

    class WorseQA:
        def answer_batch(self, questions, context):
            # The original answers correctly; every rewrite is wrong.
            return [("gold", 0.0, -0.5)] + [("nonsense", 0.0, -0.5)] * (len(questions) - 1)

    metrics, records = rft_on_policy(
        FakeRef(), WorseQA(),
        [{"question": "q", "context": "c", "answers": ["gold"]}],
        candidates=3, batch_questions=1, epochs=1, log=lambda *_: None)
    assert metrics["optimizer_steps"] == 0
    assert metrics["selected_rate"] == 0.0
    assert records == []
