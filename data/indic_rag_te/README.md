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

A Telugu QA sample for reward-based question reformulation experiments.

## Files

- `train.jsonl`: 804 Telugu question, context, and answer records.
- `dev.jsonl`: 200 QA records split by Wikipedia article from training.
- `reformulations_review.jsonl`: 300 rule-based rewrite candidates (242 train, 58 dev); no language model generated these examples. All are marked for Telugu-speaker review.
- `rule_reformulations/train.jsonl` and `rule_reformulations/dev.jsonl`: split-specific rewrite records.
- `metadata.json`: source, sampling, filtering, and split information.

Reformulations use conservative substitutions for common question phrases such as `ఎక్కడ` / `ఏ ప్రదేశంలో`, `ఎప్పుడు` / `ఏ సమయంలో`, and `ఏ విధంగా` / `ఎలా`. Every rewrite retains the source question, context, answer, article ID, and split. The rewrites are silver labels: semantic equivalence has not been independently verified. Review them with a fluent Telugu speaker before using them as supervised labels.

## QA data source

QA examples were sampled from the Telugu (`te`) training config of [AI4Bharat Indic-Rag-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite), derived from Telugu Wikipedia. The source records attribute QA generation to Meta-Llama-3.3-70B-Instruct. The source card declares MIT. Rows with missing values were removed, answer strings were required to occur in context after Unicode normalization, and train/dev article overlap is zero. These checks do not establish answer correctness.

The IndicQA Telugu test split remains evaluation-only; exact context duplicates are removed at runtime. This is a pilot dataset, not evidence of improved Telugu question answering.
