# Human answer review

Use `scripts/review_answers.py init` to create one review file per reviewer from a `--generate` run. Read each question, reference answer, gold evidence, generated answer, and `generation_context` in the run report. Fill applicable fields with `1` (passes) or `0` (fails); leave fields that do not apply as `null`. Add a note for every failure. A review cannot be scored with missing required judgments.

## Labels

For **answerable** questions:

- `correct`: the answer agrees with the reference and source documents. Equivalent wording is fine; a lucky correct guess can pass here but fail grounding.
- `grounded`: every factual claim is supported by the actual `generation_context`, with no unsupported extra claim.
- `complete`: it includes all essential conditions and parts requested by the question.
- `citation_correct`: every cited filename supports the claim it is attached to; if a factual answer has no source citation, score `0`.
- `abstained`: leave `null`.

For **unanswerable** questions:

- `correct`: it does not assert an unsupported answer.
- `grounded`: it introduces no unsupported factual claims.
- `abstained`: it clearly states that the provided documents lack the answer.
- `complete` and `citation_correct`: leave `null`.

Two reviewers should score independently. Run `score` with both files to list disagreements, discuss them against the source text, and save an adjudicated third review. Report each reviewer and the adjudicated result rather than silently averaging conflicting labels. The bootstrap interval in the score output reflects question sampling only; it does not account for reviewer disagreement.

## Experimental protocol

1. Confirm the indexed corpus contains only the intended evaluation documents. Freeze the `test` split and the reference evidence before tuning.
2. Try changes on `dev` first. Compare retrieval settings with the paired comparison script. Choose one configuration before opening `test` results.
3. Run retrieval-only, generated-answer, and oracle-context evaluations with the same corpus and question set. Review answers without looking at which configuration produced them where practical.
4. Report the number of cases and category breakdown beside every percentage. Include per-case failures, latency distribution, dataset/corpus hashes, and exact model settings.
5. Treat this 24-case set as a starter regression suite. Expand with independently written questions and more documents before claiming general RAG quality. Synthetic paraphrases or LLM-generated questions may be used to discover failures, but should not be the sole held-out evidence.
