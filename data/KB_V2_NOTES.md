# Knowledge Base v2 — Editorial and provenance notes

**Status:** Educational prototype corpus; not clinician-reviewed, not a medical device, not for diagnosis or patient care.

## Package contents

- `data/knowledge/*.md`: 15 Markdown documents ready for the current document loader (12 medical + 3 synthetic clinic policies).
- `kb_v2_manifest.csv`: source identifiers, content type, and date on which the sources were consulted.

## Source policy

The medical summaries are original English paraphrases based on the **MedlinePlus health-topic Summary** sections and the **MedlinePlus Medical Tests** pages, not copied third-party encyclopedia or linked content. Source: MedlinePlus, National Library of Medicine. Relevant US public-domain guidance and preferred attribution: https://medlineplus.gov/about/using/usingcontent/ . This does not imply that MedlinePlus reviewed, authored, or endorsed this edited corpus.

Each clinical Markdown file preserves the original public webpage URL. The three Demo Health Clinic policies are **entirely fictional**. They include fabricated business rules to test retrieval; they are not operating instructions for a real clinic.

## Compatibility notes

1. The existing `load_documents()` parser reads four required front-matter fields: `document_id`, `title`, `source`, and `source_url`. Each `.md` follows that contract.
2. The current `chunk_document()` treats each non-empty `##` section as one chunk. All documents use this convention. There are no extra `##` headings in the body besides intended chunk sections.
3. The synthetic policies use **URN identifiers** in `source_url`, e.g. `urn:medical-chatbot-sim:demo-health-clinic:appointment-policy`. These are identifiers, **not clickable web pages**. The current Pydantic schema types `source_url` as `str`, so the format is parseable. A future architecture change should distinguish `source_uri` from a real HTTP URL or allow an optional URL for synthetic sources.
4. `fever.md` in this package updates the existing file (notably by clarifying that ~37 °C is normally around average body temperature). Review the diff before overwriting it.
5. Source content and structure were curated for a retrieval experiment. Do not tune text to match evaluation queries. Before creating a gold query set, freeze this corpus and record a Git commit SHA.
6. These files are **not** a retrieval benchmark or proof of medical correctness. They require manual clinical/source review before use outside a classroom experiment. In particular, recommendations and eligibility related to medicines and vaccines can change.

## How to use

Copy **only** the 15 Markdown files from `data/knowledge/` into the repo's `data/knowledge/` directory. Keep the manifest and these notes outside that directory; the loader treats every `.md` in the folder as a knowledge document.

Suggested next steps: inspect the 15 documents; run existing document-loading and chunking tests; audit the generated sections and chunk lengths; then freeze the corpus and design a query set with independently adjudicated `relevant_chunk_ids` and answerability labels.
