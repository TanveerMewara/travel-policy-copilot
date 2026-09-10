# Evaluation

Retrieval and answer generation are measured separately because they fail for
different reasons. A generator cannot recover a fact that retrieval never placed
in its context.

## Retrieval protocol

`data/evaluation_dev.json` contains seven answerable and three unsupported
questions. `data/evaluation_test.json` contains a separate set with the same
counts. The answerability cutoff is selected from development scores only; the
held-out set is used once to estimate generalization.

Run:

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

| Split | Method | Hit@1 | Hit@3 | MRR |
| --- | --- | ---: | ---: | ---: |
| Development | Semantic | 71.4% | 100% | 0.857 |
| Development | Hybrid | 85.7% | 100% | 0.929 |
| Development | Reranked | 100% | 100% | 1.000 |
| Held-out | Semantic | 85.7% | 100% | 0.929 |
| Held-out | Hybrid | 100% | 100% | 1.000 |
| Held-out | Reranked | 100% | 100% | 1.000 |

The semantic cutoff `0.637` scored 70% answerability accuracy on held-out data.
The cross-encoder cutoff `1.696` scored 100% on development and held-out data,
which is why the application gates on the reranker score.

## Answer protocol

`data/evaluation_answers.json` contains six questions spanning policy versions,
numeric boundaries, and non-refundable terms. Run:

```powershell
.\.venv\Scripts\python.exe evaluate_answers.py
```

| Metric | Result |
| --- | ---: |
| Correct retrieved source | 100% |
| Required-fact coverage | 100% |
| Citation coverage | 100% |
| Lexical grounding | 80.9% |

Required-fact coverage checks normalized phrases chosen in advance. Lexical
grounding measures answer-token overlap with evidence; it is a useful diagnostic,
not a semantic faithfulness judge. The test sets are small and fictional. Stronger
evidence would require a larger expert-reviewed corpus, adversarial questions,
repeat runs after every corpus change, and monitoring of real user failures.

