"""Frozen CPU extractive QA with context-only span decoding."""
import inspect
import time

import torch


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
            if "token_type_ids" in encoded:
                context_mask = encoded["token_type_ids"][batch_index].eq(1)
            else:
                # Fast tokenizers expose sequence IDs, letting us avoid decoding spans
                # from the question or special tokens.
                try:
                    sequence_ids = encoded.sequence_ids(batch_index)
                except (AttributeError, TypeError):
                    sequence_ids = None
                if sequence_ids is None:
                    continue
                context_mask = torch.tensor([sid == 1 for sid in sequence_ids], device=self.device)
            context_mask &= inputs["attention_mask"][batch_index].bool()
            if not context_mask.any():
                continue
            start_logits[~context_mask] = -1e9
            end_logits[~context_mask] = -1e9
            starts = torch.topk(start_logits, k=min(20, len(start_logits))).indices.tolist()
            ends = torch.topk(end_logits, k=min(20, len(end_logits))).indices.tolist()
            for start in starts:
                for end in ends:
                    if end < start or end - start > 30:
                        continue
                    score = float(start_logits[start] + end_logits[end])
                    if score > best[0]:
                        char_start = int(offsets[batch_index, start, 0])
                        char_end = int(offsets[batch_index, end, 1])
                        best = (score, context[char_start:char_end].strip())
        return best[1], time.perf_counter() - started
