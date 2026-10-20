---
language:
  - te
license: mit
task_categories:
  - question-answering
  - text-generation
pretty_name: ActiveQA Telugu RL Sample
size_categories:
  - 1K<n<10K
---

# ActiveQA Telugu RL Sample

This repository contains a small, reproducible Telugu question-answering sample intended for reinforcement-learning experiments on question reformulation.

## Files

- `train.jsonl`: 804 Telugu question, context, and answer records.
- `dev.jsonl`: 200 corresponding records, split by Wikipedia article from the training set.
- `reformulations_review.jsonl`: 100 OpenCode MiniMax-generated candidate rewrites of training questions. These are draft annotations and require human review; they are not validated semantic-equivalence labels.
- `metadata.json`: source, sampling, filtering, seed, and split information.

The JSONL QA schema uses `id`, `question`, `context`, `answers` (list), `title`, `wiki_id`, `url`, and source provenance fields. The reformulation file additionally contains `candidate`, `generator`, and `needs_manual_review`.

## Source and construction

The QA examples were sampled from the Telugu (`te`) training config of [AI4Bharat Indic-Rag-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite), which is derived from Telugu Wikipedia and whose records attribute QA generation to Meta-Llama-3.3-70B-Instruct. The source card declares MIT. Sampling is deterministic (seed 42); identical QA/context rows are removed; answer strings must occur in the context after Unicode punctuation/spacing normalization; contexts shorter than 100 characters are excluded; and complete `wiki_id` groups are assigned to a single split. The resulting train/dev article overlap is zero.

The generated sample was also checked against the locally cached IndicQA Telugu evaluation contexts; no exact context duplicates were found. At runtime, the project drops any exact context duplicates it encounters between external training data and the selected IndicQA evaluation set.

The candidate rewrites were generated with the free OpenCode model `opencode/minimax-m3.1-flash-free`. Automatic checks cover nonempty output and ASCII-number preservation only. No semantic-preservation claim is made. Review candidates before training on them or reporting their quality.

## Limitations

Both the source QA examples and generated candidates are synthetic/automatically constructed. The answer-in-context filter is not a factuality check. The dataset is suitable for a pilot pipeline and reward-function debugging, not for claiming benchmark improvement. The associated IndicQA test set must remain evaluation-only.
