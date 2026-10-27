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


def batch_loss_step(logprobs, optimizer, params, clip=1.0):
    """Teacher-forcing update on selected high-reward sequences.

    Used by rejection-sampling fine-tuning: only sequences that were sampled,
    validated, and scored at least as well as the original question are passed
    here. The loss increases their likelihood, so a batch with no good sample
    is skipped instead of pushing every sampled action down as REINFORCE would.
    """
    import torch
    if not logprobs:
        raise ValueError("At least one target sequence is required")
    loss = -torch.stack(logprobs).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise RuntimeError("policy step produced no trainable gradients")
    if not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("non-finite gradient")
    torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    return float(loss.detach())


def reinforce_batch_step(logprobs, advantages, optimizer, params, clip=1.0):
    """One REINFORCE update over a batch of sampled actions.

    Advantages are expected to be per-question centred (typically against the
    original-question reward). Scaling by their batch standard deviation keeps
    the update magnitude stable without removing the sign of the signal.
    """
    import torch
    if not logprobs:
        raise ValueError("At least one sampled action is required")
    if len(logprobs) != len(advantages):
        raise ValueError("logprobs and advantages must have the same length")
    device = logprobs[0].device
    adv = torch.as_tensor(advantages, dtype=logprobs[0].dtype, device=device)
    if adv.numel() > 1:
        adv = adv / (adv.std(unbiased=False) + 1e-6)
    loss = -(torch.stack(logprobs) * adv).sum() / len(logprobs)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite policy loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise RuntimeError("REINFORCE produced no trainable gradients")
    if not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("non-finite policy gradient")
    torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    return float(loss.detach())
