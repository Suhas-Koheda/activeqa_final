import sys
sys.path.insert(0,'src')
from activeqa.preprocessing import normalize,tokens,entity_signature
from activeqa.rewards import token_f1,reward
from activeqa.policy_gradient import RunningBaseline
from activeqa.policy_gradient import reinforce_step
from activeqa.data import _load_indicqa_json
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


def test_flatten_squad_style_indicqa_json():
    raw={"data":[{"paragraphs":[{"context":" మద్రాస్ ","qas":[{"id":"te1","question":" ఎక్కడ? ","answers":[{"text":"చెన్నై","answer_start":0}]}]}]}]}
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",delete=False) as f:
        json.dump(raw,f,ensure_ascii=False); path=f.name
    try:
        rows=_load_indicqa_json(path)
        assert rows==[{"id":"te1","context":"మద్రాస్","question":"ఎక్కడ?","answers":["చెన్నై"]}]
    finally: os.unlink(path)
