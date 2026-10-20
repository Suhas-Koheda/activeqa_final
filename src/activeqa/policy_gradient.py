from __future__ import annotations


class RunningBaseline:
    def __init__(self,momentum=.9): self.momentum=momentum; self.value=None
    def advantage(self,reward):
        old=self.value if self.value is not None else 0.0
        adv=reward-old
        self.value=reward if self.value is None else self.momentum*self.value+(1-self.momentum)*reward
        return adv

def reinforce_step(logprob,reward_value,baseline,optimizer,params,clip=1.0):
    import torch
    adv=baseline.advantage(float(reward_value))
    loss=-logprob*torch.tensor(adv,dtype=logprob.dtype)
    if not torch.isfinite(loss): raise FloatingPointError("non-finite policy loss")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grads=[p.grad for p in params if p.grad is not None]
    if not grads: raise RuntimeError("REINFORCE produced no trainable gradients")
    if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError("non-finite policy gradient")
    torch.nn.utils.clip_grad_norm_(params,clip)
    optimizer.step()
    return float(loss.detach()),adv
