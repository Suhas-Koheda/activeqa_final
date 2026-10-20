"""Frozen CPU extractive QA with context-only span decoding."""
import inspect
import time

import torch


def _context_token_mask(encoded, inputs, batch_index, device):
    if "token_type_ids" in encoded:
        return encoded["token_type_ids"][batch_index].eq(1).to(device)
    try:
        sequence_ids = encoded.sequence_ids(batch_index)
    except (AttributeError, TypeError):
        sequence_ids = None
    if sequence_ids is None:
        return None
    return torch.tensor([sid == 1 for sid in sequence_ids], device=device)


class TeluguQA:
    def __init__(self, name, cache_dir="models", max_length=384, stride=64, device="cpu"):
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.device = torch.device(device)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            name, cache_dir=cache_dir).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_length = max_length
        self.stride = min(stride, max(0, max_length // 4))
        self.forward_fields = set(inspect.signature(self.model.forward).parameters)

    @torch.inference_mode()
    def answer(self, question, context):
        text, elapsed, _ = self.answer_with_score(question, context)
        return text, elapsed

    @torch.inference_mode()
    def answer_with_score(self, question, context):
        """Return answer text, elapsed seconds, and a context-span confidence score.

        Confidence is the best start/end log-probability sum among legal context
        spans. It is an inference-time ranking signal, not a calibrated probability.
        """
        started = time.perf_counter()
        encoded = self.tok(
            question, context, return_tensors="pt", truncation="only_second",
            max_length=self.max_length, stride=self.stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        inputs = {key: value.to(self.device) for key, value in encoded.items()
                  if key in self.forward_fields and key in {
                      "input_ids", "attention_mask", "token_type_ids"}}
        output = self.model(**inputs)
        best = (float("-inf"), "")
        for batch_index in range(inputs["input_ids"].shape[0]):
            start_logits = output.start_logits[batch_index].clone()
            end_logits = output.end_logits[batch_index].clone()
            # Tokenizer outputs stay on CPU; model inputs/logits may be on CUDA.
            context_mask = _context_token_mask(encoded, inputs, batch_index, self.device)
            if context_mask is None:
                continue
            context_mask &= inputs["attention_mask"][batch_index].bool()
            if not context_mask.any():
                continue
            start_logits[~context_mask] = -1e9
            end_logits[~context_mask] = -1e9
            start_logp = torch.log_softmax(start_logits, dim=-1)
            end_logp = torch.log_softmax(end_logits, dim=-1)
            starts = torch.topk(start_logp, k=min(20, len(start_logp))).indices.tolist()
            ends = torch.topk(end_logp, k=min(20, len(end_logp))).indices.tolist()
            for start in starts:
                for end in ends:
                    if end < start or end - start > 30:
                        continue
                    score = float(start_logp[start] + end_logp[end])
                    if score > best[0]:
                        char_start = int(offsets[batch_index, start, 0])
                        char_end = int(offsets[batch_index, end, 1])
                        best = (score, context[char_start:char_end].strip())
        return best[1], time.perf_counter() - started, best[0]
