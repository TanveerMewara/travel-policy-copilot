# Five-minute demo

Start the application with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Ask: `Under the 2026 Harbor Hotel standard policy, what happens if I cancel
   36 hours before check-in?` Show the answer, `[1]` citation, evidence card,
   effective dates, and retrieval diagnostics.
2. Ask the same question without `2026`. Show that the application finds
   conflicting 2025 and 2026 policies and asks which year applies.
3. Ask: `Does Harbor Hotel provide an airport transfer?` Show that the
   cross-encoder score falls below the calibrated cutoff and generation is skipped.
4. Use the supplier and rate filters to demonstrate metadata filtering before
   vector search.
5. Open the PDF uploader and explain that uploaded text PDFs retain page-level
   citations, overlap at chunk boundaries, rebuild the index, and invalidate stale
   content fingerprints.
6. Run `evaluate_retrieval.py` and point out the improvement from semantic Hit@1
   to reranked Hit@1 on both data splits.

Useful recruiter explanation:

> I built every important RAG stage separately so I could measure it. Hybrid
> retrieval improves recall, the cross-encoder improves ordering, metadata handles
> policy versions, and the calibrated gate prevents unsupported questions from
> reaching the LLM. I also kept retrieval evaluation separate from answer
> evaluation, because a fluent answer cannot compensate for missing evidence.

