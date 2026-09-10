"""Streamlit interface for the Travel Policy Copilot learning project."""

from time import perf_counter

import streamlit as st

from document_ingestion import ingest_pdf_bytes
from hybrid_search import hybrid_retrieve
from index_documents import build_index, load_index
from model_runtime import load_embedding_model, load_generator_model
from query_validation import missing_policy_year
from rag import DEFAULT_MIN_SIMILARITY, build_prompt, finalize_answer, generate_answer
from reranker import load_reranker_model, rerank
from telemetry import record_query

st.set_page_config(
    page_title="Travel Policy Copilot",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_retrieval_resources():
    """Load the persistent index and embedding model once per app process."""
    chunks, vectors, manifest = load_index()
    model = load_embedding_model()
    return chunks, vectors, manifest, model


@st.cache_resource(show_spinner=False)
def load_generation_resources():
    """Load the answer model only after the first supported question."""
    return load_generator_model()


@st.cache_resource(show_spinner=False)
def load_reranker_resource():
    """Load the cross-encoder once, after the first answerable question."""
    return load_reranker_model()


def filtered_index(chunks, vectors, supplier, rate):
    """Apply exact metadata filters before vector search."""
    indices = [
        index
        for index, chunk in enumerate(chunks)
        if (supplier == "All" or chunk["metadata"].get("supplier") == supplier)
        and (rate == "All" or chunk["metadata"].get("rate") == rate)
    ]
    return [chunks[index] for index in indices], vectors[indices]


def answer_question(question, chunks, vectors, model, supplier, rate, threshold):
    """Run filtered hybrid retrieval, abstention, and grounded generation."""
    started = perf_counter()
    selected_chunks, selected_vectors = filtered_index(chunks, vectors, supplier, rate)
    if not selected_chunks:
        return {
            "answer": "No indexed policies match the selected filters.",
            "abstained": True,
            "notice": "No policy matched the selected metadata filters.",
            "results": [],
            "answerability_score": 0.0,
            "elapsed": perf_counter() - started,
        }

    candidates = hybrid_retrieve(
        question, selected_chunks, model, selected_vectors, top_k=min(10, len(selected_chunks))
    )
    best_similarity = max(result["semantic_score"] for result in candidates)
    available_years = missing_policy_year(question, candidates)
    if available_years:
        answer = "Which policy year applies? Available years: " + ", ".join(available_years) + "."
        abstained = True
        notice = "Generation was skipped until the policy version is specified."
        results = candidates[:3]
    else:
        results = rerank(question, candidates, load_reranker_resource(), top_k=3)
        if results[0]["reranker_score"] < threshold:
            answer = "I don't have enough information in the provided policies."
            abstained = True
            notice = "Generation was skipped because evidence was insufficient."
        else:
            tokenizer, generator = load_generation_resources()
            prompt = build_prompt(question, results[:1])
            answer = generate_answer(prompt, tokenizer, generator)
            answer = finalize_answer(answer, question, results[0]["text"])
            if answer and not answer.rstrip().endswith("[1]"):
                answer = f"{answer.rstrip()} [1]"
            abstained = False
            notice = ""

    return {
        "answer": answer or "The answer model returned no text.",
        "abstained": abstained,
        "notice": notice,
        "results": results,
        "answerability_score": best_similarity,
        "elapsed": perf_counter() - started,
    }


def show_source(result, number, supplied):
    metadata = result["metadata"]
    label = f"[{number}] {result['title']} · {result['section']}"
    with st.expander(label, expanded=supplied):
        st.write(result["text"])
        page = metadata.get("page_number")
        page_label = f"  |  Page: {page}" if page else ""
        st.caption(
            f"{result['source']}  |  Chunk: {result['chunk_id']}{page_label}  |  "
            f"Effective: {metadata.get('effective_from', '-')} to "
            f"{metadata.get('effective_to', '-')}"
        )
        if supplied:
            st.success("Supplied to the answer model")


st.title("✈️ Travel Policy Copilot")
st.caption("Grounded answers from fictional policy documents · local learning project")

try:
    with st.spinner("Loading the vector index and embedding model..."):
        all_chunks, all_vectors, index_manifest, embedding_model = load_retrieval_resources()
except SystemExit as error:
    st.error(str(error))
    st.code(r".\.venv\Scripts\python.exe index_documents.py", language="powershell")
    st.stop()

suppliers = sorted({chunk["metadata"].get("supplier", "Unknown") for chunk in all_chunks})
rates = sorted({chunk["metadata"].get("rate", "Unknown") for chunk in all_chunks})

with st.sidebar:
    st.header("Search controls")
    supplier_filter = st.selectbox("Supplier", ["All", *suppliers])
    rate_filter = st.selectbox("Rate", ["All", *rates])
    threshold = st.slider(
        "Minimum reranker relevance",
        min_value=-5.0,
        max_value=8.0,
        value=float(DEFAULT_MIN_SIMILARITY),
        step=0.1,
        help="Below this development-set cutoff, the app does not call the answer model.",
    )
    show_diagnostics = st.toggle("Show retrieval diagnostics", value=True)
    st.divider()
    st.subheader("Index status")
    st.metric("Indexed passages", index_manifest["chunk_count"])
    st.caption(f"Embedding model: `{index_manifest['embedding_model']}`")
    st.caption(f"Built: {index_manifest['created_at_utc']}")
    if st.button("Reload index", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    with st.expander("Add a PDF policy"):
        uploaded_pdf = st.file_uploader("Policy PDF", type=["pdf"])
        upload_id = st.text_input("Document ID", placeholder="hotel-flex-2026")
        upload_title = st.text_input("Document title")
        upload_supplier = st.text_input("Supplier")
        upload_rate = st.text_input("Rate")
        upload_from = st.date_input("Effective from")
        upload_to = st.date_input("Effective to")
        if st.button("Ingest and rebuild index", use_container_width=True):
            if uploaded_pdf is None:
                st.error("Choose a PDF first.")
            else:
                try:
                    ingest_pdf_bytes(
                        uploaded_pdf.getvalue(),
                        {
                            "document_id": upload_id,
                            "title": upload_title,
                            "supplier": upload_supplier,
                            "rate": upload_rate,
                            "effective_from": upload_from.isoformat(),
                            "effective_to": upload_to.isoformat(),
                        },
                    )
                    with st.spinner("Extracting pages and rebuilding the index..."):
                        build_index(embedding_model)
                    st.cache_resource.clear()
                    st.success("PDF indexed successfully.")
                    st.rerun()
                except (ValueError, FileExistsError) as error:
                    st.error(str(error))

st.info("Demo data is fictional and must not be treated as real supplier policy.")

example_columns = st.columns(3)
example_prompt = None
with example_columns[0]:
    if st.button("Harbor cancellation", use_container_width=True):
        example_prompt = (
            "What happens if I cancel the 2026 Harbor Hotel standard rate 36 hours before check-in?"
        )
with example_columns[1]:
    if st.button("Pine check-in", use_container_width=True):
        example_prompt = "When can I check in at Pine Hotel?"
with example_columns[2]:
    if st.button("Missing information", use_container_width=True):
        example_prompt = "Does Harbor Hotel provide an airport transfer?"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if message.get("abstained"):
                st.warning(message.get("notice", "Answer generation was skipped."))
            for number, result in enumerate(message.get("results", []), 1):
                show_source(result, number, supplied=(number == 1 and not message["abstained"]))
            if show_diagnostics and message.get("results"):
                with st.expander("Retrieval diagnostics"):
                    st.dataframe(
                        [
                            {
                                "rank": number,
                                "chunk": result["chunk_id"],
                                "semantic similarity": round(result["semantic_score"], 3),
                                "semantic rank": result["semantic_rank"],
                                "keyword rank": result["keyword_rank"],
                                "RRF score": round(result["score"], 5),
                                "reranker score": round(result.get("reranker_score", 0), 3),
                            }
                            for number, result in enumerate(message["results"], 1)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(f"Pipeline time: {message['elapsed']:.2f} seconds")

typed_prompt = st.chat_input("Ask about cancellation, changes, or check-in...")
question = example_prompt or typed_prompt
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and preparing an answer..."):
            response = answer_question(
                question,
                all_chunks,
                all_vectors,
                embedding_model,
                supplier_filter,
                rate_filter,
                threshold,
            )
            record_query(question, response, supplier=supplier_filter, rate=rate_filter)
        st.write(response["answer"])
        if response["abstained"]:
            st.warning(response.get("notice", "Answer generation was skipped."))
        for number, result in enumerate(response["results"], 1):
            show_source(result, number, supplied=(number == 1 and not response["abstained"]))
        if show_diagnostics and response["results"]:
            with st.expander("Retrieval diagnostics"):
                st.json(
                    {
                        "best_semantic_similarity": round(response["answerability_score"], 3),
                        "reranker_score": round(response["results"][0].get("reranker_score", 0), 3),
                        "threshold": threshold,
                        "pipeline_seconds": round(response["elapsed"], 2),
                    }
                )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            **response,
        }
    )
