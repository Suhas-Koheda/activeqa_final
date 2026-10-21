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
