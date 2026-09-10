# Travel Policy Copilot

A local, production-shaped Retrieval-Augmented Generation (RAG) application for
answering questions over versioned travel-policy documents. It demonstrates the
full lifecycle: ingestion, chunking, persistent embeddings, hybrid retrieval,
cross-encoder reranking, grounded generation, citations, abstention, evaluation,
telemetry, tests, a Streamlit interface, and container deployment.

All included policies are fictional examples. They are not Mondee or supplier
policies.

## Why this project stands out

- Combines semantic and lexical keyword signals with Reciprocal Rank Fusion.
- Reranks a small candidate set using a cross-encoder before generation.
- Detects conflicting policy years and asks for clarification instead of guessing.
- Fits an answerability threshold on development data and evaluates it once on a
  separate held-out set.
- Skips the generator for unsupported questions and uses a policy-text fallback
  when the small local model produces a fragile answer.
- Preserves source, section, effective dates, chunk IDs, and PDF page numbers.
- Logs latency and retrieval diagnostics without storing the raw question.

## Measured results

| Split | Method | Hit@1 | Hit@3 | MRR |
| --- | --- | ---: | ---: | ---: |
| Development | Semantic | 71.4% | 100% | 0.857 |
| Development | Hybrid | 85.7% | 100% | 0.929 |
| Development | Reranked | **100%** | **100%** | **1.000** |
| Held-out | Semantic | 85.7% | 100% | 0.929 |
| Held-out | Hybrid | **100%** | **100%** | **1.000** |
| Held-out | Reranked | **100%** | **100%** | **1.000** |

The cross-encoder threshold selected on the development set achieved 100%
answerability accuracy on the held-out set. The six-case generation evaluation
reported 100% correct source selection, 100% required-fact coverage, 100% citation
coverage, and 80.9% lexical grounding. The dataset is intentionally small, so
these figures demonstrate the evaluation method rather than production readiness.

## Run locally

Use Python 3.10 or newer. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_models.py
.\.venv\Scripts\python.exe index_documents.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`prepare_models.py` is the only step that needs network access. Runtime model
loading resolves cached snapshot paths directly and works offline.

Run a CLI query:

```powershell
.\.venv\Scripts\python.exe rag.py "Under the 2026 Harbor Hotel standard policy, what happens if I cancel 36 hours before check-in?"
```

Run the engineering checks:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluate_retrieval.py
.\.venv\Scripts\python.exe evaluate_answers.py
```

Or run it with Docker:

```powershell
docker compose up --build
```

Open `http://localhost:8501`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation methodology](docs/EVALUATION.md)
- [Recruiter demo script](docs/DEMO.md)
- [Step-by-step learning tutorial](docs/TUTORIAL.md)

## Main components

| File | Responsibility |
| --- | --- |
| `document_ingestion.py` | Validates PDFs and extracts overlapping page chunks |
| `index_documents.py` | Builds the persistent vector index and stale-index fingerprint |
| `hybrid_search.py` | Combines semantic and keyword rankings with RRF |
| `reranker.py` | Applies cross-encoder relevance scoring |
| `query_validation.py` | Detects ambiguous policy versions |
| `rag.py` | Runs retrieve, rerank, abstain, augment, generate, and cite |
| `app.py` | Provides the Streamlit UI, filters, upload, and diagnostics |
| `evaluate_retrieval.py` | Measures retrieval and answerability on dev/test splits |
| `evaluate_answers.py` | Measures facts, citations, sources, and lexical grounding |
| `telemetry.py` | Records privacy-conscious local operational events |
