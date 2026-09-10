# Travel Policy Copilot — step-by-step tutorial

## Lesson 1: retrieval before generation

All policies in this repository are fictional learning examples. They are not
Mondee policies or real supplier terms.

This first version uses Python 3.9+ and its standard library. No packages,
API key, paid service, or model download is required.

From this folder, run:

```powershell
python search.py "Harbor Hotel standard cancellation"
```

Or run `python search.py` and type a question when prompted.

### What happens

1. `load_chunks()` reads the Markdown policy files and makes a passage (chunk)
   from each section. It keeps the document title, section, and source path.
2. `tokenize()` lowercases text, extracts words, and removes a few common words.
3. `retrieve()` counts distinct words shared by the question and each passage,
   including its title and section. It returns up to three matches.
4. `main()` prints the evidence and the words responsible for each score.

Try these queries and read the actual passages:

```powershell
python search.py "Harbor Hotel standard cancellation"
python search.py "Harbor Hotel saver cancellation"
python search.py "Pine Hotel standard cancellation"
python search.py "What time is check-in at Harbor Hotel?"
python search.py "Do you offer airport transfers?"
```

For the first query, Harbor Hotel's Standard Rate cancellation section should
rank first. Compare it with Saver Rate: the same hotel has different rules.

### Understand the limitations

This is keyword search, not semantic search and not a complete RAG pipeline.
It cannot reliably connect synonyms such as "money back" and "refund", enforce
which supplier applies, interpret deadlines, or decide if evidence answers a
question. A high score is not proof that a passage is the right policy. Equal
scores retain file/section order; that order does not imply greater relevance.

Try asking `"Harbor Hotel airport transfers"`. It may return unrelated passages
because the hotel name matches. This is a useful failure to observe: finding
similar words is different from finding enough evidence to answer.

### Your exercise

1. Run the first three queries and explain why their top passages differ.
2. Open `data/policies/harbor_standard.md` and change its 48-hour deadline to 72
   hours in both places. Rerun the first query and observe the changed passage.
3. Restore the original 48-hour example when done.
4. Trace the code in this order: `main`, `load_chunks`, `tokenize`, `retrieve`.

Next lesson: use embeddings to search by meaning and compare the results with
this keyword baseline. Later we add answer generation, citations, policy
filters, hybrid retrieval, and evaluation.

## Lesson 2: search by meaning

`semantic_search.py` runs both retrieval methods on the same question. It uses
the pretrained `sentence-transformers/all-MiniLM-L6-v2` embedding model locally
on your CPU. No API key or paid inference service is required. Package setup
and the first model download require internet access. Dependencies include
PyTorch, so installation can take time and substantial disk space.

Use Python 3.10+ for this lesson. In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe semantic_search.py "Can I get my money back?"
```

If `.venv` already exists and dependencies are installed, just run the last
command. Calling its Python directly avoids needing to activate the environment.

### Follow the data

1. Reuse `load_chunks()` from lesson 1; source metadata stays attached.
2. Load a pretrained embedding model (we are not training it).
3. `embed_chunks()` converts each title, section, and passage into a vector
  of 384 numbers. The current 14 passages form an array of shape `(14, 384)`.
4. `semantic_retrieve()` encodes the question using the same model.
5. Compare its vector with each passage vector and return the top three.

An embedding is a learned numerical representation useful for comparing
meaning. Its individual numbers are not labelled facts such as "refund".
`normalize_embeddings=True` makes each vector have length one. The `@`
operation then computes cosine similarities between the passage vectors and
the question vector. Higher scores mean closer vectors, not verified answers.

Vectors remain in memory and are recomputed each run. The model download is
cached in `.cache/models`. With six passages we can compare every vector
directly; persistent vector storage will be a separate lesson.

### Compare and inspect

```powershell
.\.venv\Scripts\python.exe semantic_search.py "Can I get my money back?"
.\.venv\Scripts\python.exe semantic_search.py "When can I arrive at the hotel?"
.\.venv\Scripts\python.exe semantic_search.py "Harbor Hotel saver cancellation"
.\.venv\Scripts\python.exe semantic_search.py "Do you offer airport transfers?"
```

For each query, ask: Is the right topic retrieved? Is it the right hotel and
rate? Does the passage actually answer the question? The first query does not
specify a hotel or rate, so related refund passages cannot establish one
definitive policy for the user.

The airport-transfer query intentionally has no answer in our documents.
Semantic search still returns candidates. We have not implemented answerability
detection or chosen a similarity threshold; a threshold needs evaluation and
is not a guarantee. Likewise, semantic similarity does not enforce hotel or
rate identity. We'll address these limitations in later lessons.

Official references:
- https://www.sbert.net/docs/quickstart.html
- https://www.sbert.net/docs/package_reference/sentence_transformer/model.html

## Lesson 3: the complete basic RAG loop

`rag.py` adds generation to semantic retrieval. It uses `google/flan-t5-small`,
a small instruction-following model that runs locally on CPU. The first run
downloads the model. It is useful for seeing the architecture, but its answers
will be less reliable than those from a larger production model.

Install the one new dependency, then ask a fully specified question:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe rag.py "What happens if I cancel the Harbor Hotel standard rate 36 hours before check-in?"
```

To see the exact prompt—including retrieved evidence—run:

```powershell
.\.venv\Scripts\python.exe rag.py --show-prompt "What happens if I cancel the Harbor Hotel standard rate 36 hours before check-in?"
```

The program now follows the complete RAG sequence:

1. **Retrieve:** semantic search ranks three potentially relevant passages.
2. **Augment:** `build_prompt()` places the highest-ranked passage beside strict
   instructions and the user's question.
3. **Generate:** the local language model reads that prompt and writes an answer.
4. **Inspect:** the program displays every source supplied to the model so you
   can compare the generated answer with the original wording.

The generator does not search the files itself. It sees only the question,
instructions, and passage placed into its prompt. Retrieval therefore controls
what knowledge is available during generation. We display all three retrieval
candidates but give this small generator only the best one: an early test showed
that competing hotel rules distracted it. Production systems can use several
passages with a stronger model, metadata filters, and reranking.

The displayed source list is retrieval provenance: it proves which passages the
model received, but it does not prove which sentence caused its answer. We do
not ask this tiny model to create inline citations because our first test showed
it could output citation symbols without an answer. Reliable sentence-level
citation checking is a later feature; we preserve this failure as a design
lesson instead of presenting unvalidated citation formatting as correctness.

Try these failure cases:

```powershell
.\.venv\Scripts\python.exe rag.py "Can I get my money back?"
.\.venv\Scripts\python.exe rag.py "Does Harbor Hotel offer an airport transfer?"
```

The first question lacks a rate, and the second is unanswered by our policies.
The prompt asks the model to handle those cases, but a small model may ignore
instructions. This is why production RAG needs evaluation and additional
controls rather than trusting fluent output.

For this lesson, passage vectors are recalculated and both models are reloaded
for every command. A later application will load them once at startup and store
the document vectors in a persistent index.

## Lesson 4: evaluate retrieval

`data/evaluation_dev.json` and `data/evaluation_test.json` are labelled sets.
Each answerable question records the source and section that retrieval should
find. Questions whose answers are absent use `null` labels.

Run the evaluation without loading the answer-generation model:

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

The report includes:

- **Hit@1:** how often the expected passage is the first result.
- **Hit@3:** how often it appears anywhere in the first three results.
- **MRR (mean reciprocal rank):** rewards placing the expected passage closer
  to the top. Rank 1 contributes `1`, rank 2 contributes `1/2`, and so on.
- **Answerability accuracy:** how well a similarity threshold separates known
  questions from questions unsupported by the policies.

Retrieval evaluation and answer generation are separate because they fail for
different reasons. If the correct passage is missing, generation never received
the necessary fact. If the correct passage is present but the answer is wrong,
the prompt or generation model is responsible.

This tiny dataset is intentionally an engineering demonstration. Its threshold
is selected and measured on the same examples, so the reported accuracy is
optimistic. A real project needs many reviewed questions plus separate tuning
and held-out test sets.

The early semantic-only experiment selected `0.637`. The completed application
uses the better-calibrated cross-encoder cutoff `1.696` as its default
`--min-score`. When the best reranked passage scores lower, it returns its
insufficient-information response without calling the generator:

```powershell
.\.venv\Scripts\python.exe rag.py "Does Harbor Hotel provide an airport transfer?"
```

You can experiment with another cutoff:

```powershell
.\.venv\Scripts\python.exe rag.py --min-score 2.0 "When can I check in at Pine Hotel?"
```

This guard reduces one observed failure mode. It does not prove that a passage
above the cutoff answers the question, and a cutoff that is too high rejects
valid questions. That tradeoff must be measured on held-out examples.

## Lesson 5: hybrid retrieval

`hybrid_search.py` combines semantic and keyword result lists using Reciprocal
Rank Fusion (RRF):

```text
RRF contribution = 1 / (60 + rank)
```

A passage near the top of both lists receives two contributions. RRF uses rank
positions because a keyword overlap count and cosine similarity are different
measurements and should not be added as though they share one scale.

Inspect a fused ranking:

```powershell
.\.venv\Scripts\python.exe hybrid_search.py "Can I get my money back on the Harbor Hotel saver rate?"
```

Then rerun `evaluate_retrieval.py`. It prints semantic, hybrid, and reranked
metrics side by side. On the development set, hybrid retrieval improves Hit@1
from 71.4% to 85.7%; reranking improves it to 100%.

`rag.py` uses the hybrid ranking to choose candidates. The completed pipeline
reranks them and uses the best cross-encoder score for abstention because an RRF
score is a ranking signal rather than a calibrated answerability signal.

## Lesson 6: persistent indexing and metadata

Each policy file now starts with metadata: a document ID, supplier, rate, and
effective date range. `load_chunks()` attaches that metadata to every section
and creates stable IDs such as `harbor-standard-2026::cancellation`.

Build the index whenever policies are added or changed:

```powershell
.\.venv\Scripts\python.exe index_documents.py
```

It writes three generated files under `data/index/`:

- `vectors.npy` contains normalized 384-number passage vectors.
- `chunks.json` contains text, source locations, IDs, and metadata.
- `manifest.json` records the model, dimensions, creation time, and a source
  fingerprint used to detect stale indexes.

Now search the stored index:

```powershell
.\.venv\Scripts\python.exe persistent_search.py "When can I check in at Pine Hotel?"
```

The embedding model is still needed to encode the new question, but policy
embeddings are loaded from disk. This separates the offline indexing path from
the online query path:

```text
Offline: policy files -> chunks -> passage embeddings -> saved index
Online:  question -> question embedding -> saved-index search -> passages
```

If a policy changes, `load_index()` refuses to use the stale vectors and tells
you to run the indexer again. `data/index/` is ignored by Git because it is a
rebuildable artifact. The RAG command now consumes this persistent index too.

This implementation scans every stored vector with NumPy, which is appropriate
for this small corpus and makes the math visible. At larger scale, the same boundary
can be backed by an approximate-nearest-neighbor vector database.

## Lesson 7: web application

`app.py` turns the pipeline into an interactive Streamlit application. Install
the UI dependency and start it with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application keeps the embedding and generation models cached between
questions, loads stored document vectors, supports exact supplier/rate filters,
shows evidence cards, and exposes semantic, keyword, and RRF diagnostics. The
generation model is loaded lazily only when a question passes the answerability
threshold. Unsupported questions therefore avoid its startup and inference cost.

The sidebar's **Reload index** action clears cached resources after you rebuild
the index. The **Clear conversation** action removes the current chat history.
Chat history is display context only in this version: each question is retrieved
and answered independently, which avoids silently using earlier conversation as
policy evidence.

## Lesson 8: PDF ingestion and page citations

The app sidebar now includes **Add a PDF policy**. Supply a document ID and
policy metadata, upload a text-based PDF, and select **Ingest and rebuild
index**. The PDF and a `.metadata.json` sidecar are stored under `data/policies/`.

During indexing, `load_pdf_chunks()` extracts every page with pypdf, splits each
page into overlapping windows of 180 words with 30 words of overlap, and stores
the page number in each chunk. A PDF chunk ID looks like:

```text
hotel-flex-2026::page-4::chunk-2
```

Keeping chunks within page boundaries gives every retrieved passage an exact
page citation. Overlap preserves some context when a fact crosses a chunk
boundary. The fixed word sizes are starting values that should later be tuned
with retrieval evaluation rather than treated as universal settings.

The uploader validates the PDF header, limits files to 10 MB, rejects duplicate
document IDs, and checks that text can be extracted before saving. Image-only
scans are rejected because they need an OCR stage. After a successful upload,
the vector index is rebuilt and the app's cached resources are refreshed.

The source fingerprint now covers Markdown files, PDFs, and PDF metadata
sidecars. Editing any of them makes the old index stale.

## Lesson 9: cross-encoder reranking

Fast retrieval compares separately created question and passage vectors. The
new `reranker.py` uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to read each
question-passage pair together and give it a new relevance score.

```powershell
.\.venv\Scripts\python.exe reranker.py "What happens if I cancel Harbor Hotel standard rate 36 hours before check-in?"
```

The production-shaped query path is now:

```text
hybrid retrieval -> up to 10 candidates -> cross-encoder -> top 3 -> top 1 to LLM
```

Reranking is more accurate in many difficult searches but costs more than vector
comparison, so it is applied to a small candidate set. Its raw score is not a
probability or confidence percentage.

`evaluate_retrieval.py` reports semantic, hybrid, and reranked metrics on both
development and held-out questions. The expanded results are documented below.

Both the command-line RAG flow and Streamlit app now rerank candidates before the
answerability decision. Unsupported questions do not load or run the generation
model.

## Lesson 10: conflicting versions and held-out evaluation

The corpus now has 14 chunks, including conflicting 2025 and 2026 Harbor and
Pine policies plus two Summit Hotel rates. Dates are included in keyword,
embedding, and reranker inputs so retrieval can distinguish policy versions.

Evaluation is split into `data/evaluation_dev.json` and
`data/evaluation_test.json`. The threshold is selected using development data;
the test questions do not participate in that choice.

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

Current retrieval results:

| Split | Method | Hit@1 | Hit@3 | MRR |
| --- | --- | ---: | ---: | ---: |
| Development | Semantic | 71.4% | 100% | 0.857 |
| Development | Hybrid | 85.7% | 100% | 0.929 |
| Development | Reranked | 100% | 100% | 1.000 |
| Held-out test | Semantic | 85.7% | 100% | 0.929 |
| Held-out test | Hybrid | 100% | 100% | 1.000 |
| Held-out test | Reranked | 100% | 100% | 1.000 |

This is the first dataset where the advanced stages show a measured benefit.
The semantic-score threshold `0.637` scores 100% on development but only 70% on
held-out questions. The cross-encoder threshold `1.696` scores 100% on both sets,
so the completed application uses the cross-encoder signal for abstention.

When otherwise identical policy versions conflict and the question omits a
year, `missing_policy_year()` asks the user to choose among the available years
before reranking or generation. This deterministic metadata check is safer than
asking the language model to guess which policy period applies.
