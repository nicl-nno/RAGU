# Module: ragu.search_engine

## Role in RAGU Pipeline

`ragu.search_engine` is the query layer. It consumes an already built `KnowledgeGraph`, retrieves context with one or more strategies, and optionally asks an LLM to synthesize an answer.

Pipeline position:

```text
KnowledgeGraph + query -> retrieval context -> LLM answer
```

## Overview

The module separates retrieval strategy from indexing. All engines share `BaseEngine`, the async `search()` for retrieval-only calls, the async `query()` for retrieval plus complete answer generation, and `stream_query()` for retrieval plus streamed text generation where supported. For many queries at once, every engine also exposes async `batch_search()` / `batch_query()`, which run retrieval concurrently and — for the leaf engines (Local, Naive, Global) — route answer generation through a single `llm.batch_chat_completion` call. `search()`, `query()`, `batch_search()`, and `batch_query()` are coroutines (await them); `stream_query()` returns an async iterator. The `a_search` / `a_query` names are kept as backward-compatible async aliases that delegate to `search` / `query`.

## Key Components

### EvidenceSearchEngine

Opt-in graph and paragraph retrieval with one evidence-grounded follow-up round.
It unions local graph chunks with direct dense paragraph hits, ranks the union
by cosine similarity, and asks the configured LLM for at most three searches
based on the initial paragraphs. Reciprocal-rank fusion combines the original
ranking and follow-up hits. Source paragraphs precede graph descriptions in the
final context, which obeys `max_context_length` during answer generation.

```python
from ragu import EvidenceSearchEngine, EvidenceSearchParams

engine = EvidenceSearchEngine(
    llm=llm, knowledge_graph=knowledge_graph, embedder=embedder,
    max_context_length=8000,
)
result = await engine.query("Your multi-hop question", EvidenceSearchParams())
print(result.response)
print(result.retrieval.metrics["followup_queries"])
```

The constructor accepts the same model, storage and tokenizer options as
`LocalSearchEngine`. `EvidenceSearchParams` inherits `LocalParams`, with
`neighbor_chunk_limit=10` for this engine only. Additional defaults are
`dense_top_k=20`, `followup_rounds=1`, `followup_top_k=10`,
`planner_seed_chunks=6`, `planner_context_tokens=3000`, and `rrf_k=60.0`.
Set `followup_rounds=0` to disable planning. Other round counts are rejected.

With planning enabled, even `search()` makes one LLM call per question;
`query()` adds the final answer call. Planner evidence has its own token budget.
`search()` returns untruncated `EvidenceSearchRetrieve` so callers can inspect
all candidates; `query()` and `stream_query()` enforce the final context cap.
The final answer uses the existing `local_search` prompt. Customize either
`local_search` or `evidence_followup` with `engine.update_prompt(...)`; the
planner schema must expose at most three string `queries`.

Retrieval metrics include direct paragraph IDs (`dense_chunks`), normalized
`followup_queries`, newly retrieved IDs (`followup_chunks`) and `planner_calls`,
alongside local graph metrics. IDs are deduplicated without modifying the graph.
The original `LocalSearchEngine` behavior is unchanged.

For a structured evidence-first answer, select the optional preset on this
engine instance (the final generation still uses one LLM call):

```python
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

engine.update_prompt("local_search", DEFAULT_PROMPT_TEMPLATES["evidence_answer"])
result = await engine.query("Your multi-hop question")
print(result.response.answer)
print(result.response.evidence)  # Up to five grounded title/fact records.
```

This preset returns a Pydantic object in `response`. Normal short-answer prompts
return a string. Structured output is supported by `query()` / `batch_query()`;
use a plain-text answer prompt with `stream_query()`. Custom answer prompts may
use `query`, `context`, `language` and a JSON `payload` containing question and
bounded context. Generated facts are model outputs, not independent validation.

To recheck only abstentions, keep the structured preset above and opt in per query:

```python
result = await engine.query(
    "Your multi-hop question", EvidenceSearchParams(review_unknown_answers=True),
)
print(result.response.answer)
print(result.payload["answer_review"])
```

This adds one model call only when the first answer is `UNKNOWN` (ignoring case
and surrounding whitespace). The review sees the same bounded source context
and the first answer's evidence; substantive first answers are preserved.
`payload["answer_review"]` records whether review ran and, when it did, the
initial answer and evidence. Both answer prompts must expose structured
`answer` and `evidence` fields. Customize `evidence_review` per instance with
`update_prompt`; review is supported by `query()` and `batch_query()`, not
`stream_query()`. The default is `review_unknown_answers=False`. Recovery can
still produce unsupported answers; review is not independent verification.

To search for missing evidence after an abstention, use one of these mutually
exclusive configurations with the structured evidence answer preset:

| Configuration | Additional work after UNKNOWN |
|---|---|
| `EvidenceSearchParams(review_unknown_answers=True)` | One review of the original context |
| `EvidenceSearchParams(recovery_mode="direct")` | One search planner, then one review |
| `EvidenceSearchParams(recovery_mode="after_review")` | Review first; if still UNKNOWN, one search planner and one final review |

`recovery_mode` defaults to `"off"`; combining a non-off mode with
`review_unknown_answers=True` raises `ValueError`. Both search modes preserve
substantive initial answers. The cascade also preserves substantive review
answers. Its planner and final review receive the **original** proposal, even
after an unsuccessful first review.

The `evidence_recovery` planner proposes at most two queries. The existing
embedder and chunk index return five hits per query (the default NanoVDB cosine
threshold is 0.2). New whole paragraphs are interleaved by rank, excluding
content already present, with a 2000-token prefix budget. This prefix precedes
the original context; the combined text uses the engine's configured total
limit. Search recovery requires a tiktoken tokenizer; the evaluated configuration
uses `gpt-4o`/o200k and `max_context_length=8000`.

After search recovery, `str(result.retrieval)` is the actual bounded final
context via `EvidenceSearchRetrieve.rendered_context`. Its chunks include the
original candidates and additions; candidates may extend beyond the rendered
token boundary. The original retrieval is not modified.
`retrieval.metrics` remains the initial search's metrics; its counts do not
describe the added search or the final context. Use the recovery payload and
`rendered_context` for final-context provenance and any subsequent evaluation.
`result.payload["answer_recovery"]` records mode, applied status, initial proposal,
queries, hit IDs, added IDs, and initial/final context when applied.
Customize `evidence_recovery` and `evidence_review` with `update_prompt`.
These modes support `query` and `batch_query`, and reject streaming.

For one further bounded attempt after the cascade still returns UNKNOWN, use
`EvidenceSearchParams(recovery_mode="after_review", second_recovery_policy="repeat")`.
The default second policy is `"off"`; `"repeat"` requires `"after_review"`.
The first recovery uses the original proposal, as above. Only the second recovery
uses the latest UNKNOWN answer/evidence and the latest bounded context. It repeats
the same planner, top-five retrieval and 2000-token addition budget once, for at
most two further model calls. Substantive answers stop the cascade immediately.
The final retrieval renders the last context. Recovery payload `rounds` records
each pass's proposal, queries, hit/addition IDs and before/after contexts; top-level
`initial` and `initial_context` retain the initial recovery inputs, while the
top-level query and hit fields describe the last pass.
`second_recovery_policy="deeper"` uses twenty hits per query **only in the second
recovery pass**. The first pass still uses five. Its planner, paragraph ordering,
2000-token prefix budget and total context limit are identical to `"repeat"`;
this mode expands the candidate pool, not the final context budget. Each recovery
payload round records its `top_k`.

### BaseEngine

Abstract base for query engines.

- Purpose: stores LLM and context truncation settings.
- Important methods: `search`, `query`, `stream_query`.
- Batch methods (all async): `batch_search`, `batch_query`.
  They take a `list[str]` of queries and return a result list **aligned by position** with the
  input. They are fail-fast: the first failing query aborts the whole batch and raises.
- Backward-compatible async aliases: `a_search`, `a_query` delegate to `search` / `query`.


### SearchEngineRetrieve

Base dataclass for retrieval-only results.

- Purpose: carries `query`, engine-specific `result`, and `metrics`.
- Important method: `to_text()`.

```python
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(
    query="What is RAGU?",
    result=NaiveSearchResult(),
    metrics={"chunks": []},
)

print(retrieval.to_text())
```

### SearchEngineResponse

Generated answer container.

- Purpose: carries `query`, `response`, `retrieval`, and optional `payload`.

```python
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(query="What is RAGU?", result=NaiveSearchResult())
response = SearchEngineResponse(
    query="What is RAGU?",
    response="RAGU is a GraphRAG engine.",
    retrieval=retrieval,
)

print(str(response))
```

### SearchEngineStreamEvent

Incremental generated-answer event returned by `stream_query()`.

- Purpose: carries `query`, the already retrieved `retrieval` context, generated
  text `delta`, and optional `payload`.
- Streaming is text-only; use `query()` for structured Pydantic responses.

```python
from ragu import SearchEngineStreamEvent

async for event in engine.stream_query("What is RAGU?"):
    assert isinstance(event, SearchEngineStreamEvent)
    print(event.delta, end="")
```

### NaiveSearchEngine

Chunk-vector RAG.

- Purpose: retrieve chunk vectors directly, optionally rerank, then answer.
- Best for: document QA when graph extraction is not needed.
- Uses: `GraphRetriever.query_chunks`.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

### LocalSearchEngine

Graph-neighborhood RAG.

- Purpose: retrieve relevant entities, then collect related relations, chunks, and community summaries.
- Best for: entity-centric questions and local factual neighborhoods.
- Uses: entity vector DB, graph edges, chunk KV, community summary KV.

```python
from ragu import KnowledgeGraph, LocalSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

#### Optional one-hop chunk expansion

Local search normally returns source chunks of the seed entities only. To also
retrieve evidence from their immediate neighbors, enable bounded expansion:

```python
from ragu.search_engine.local_search import LocalParams

params = LocalParams(neighbor_chunk_limit=10, neighbor_chunk_candidates=50)
result = await engine.search("Where is the company founded by Ada headquartered?", params)
print(result.metrics["neighbor_chunks"])
```

`engine` above is a `LocalSearchEngine`. The same parameters work with `query`,
`batch_search`, `batch_query`, and `stream_query`.

- `neighbor_chunk_limit=0` (default) preserves seed-only behavior. Positive values
  cap the additional chunks retained **after** the existing chunk reranker.
- `neighbor_chunk_candidates=50` caps additional unique chunk IDs fetched
  **before** reranking; it must be at least the output limit. Missing stored
  chunks consume candidate slots but are omitted from results.
- Neighbors are selected via incoming or outgoing edges of this query's final
  seed entities. Traversal stops after one edge; seed entities remain unchanged.
- Candidate order follows seed rank, descending relation strength, then relation
  ID. Chunks are taken round-robin across unique neighbors, skipping duplicates
  and seed chunks. Without a reranker, seed chunks come first, followed by this
  deterministic candidate order. With a reranker, both groups are ranked jointly;
  the extra-chunk cap never removes a seed chunk.
- `metrics["neighbor_chunks"]` lists retained additional chunk IDs when enabled.
  `documents_id` also includes documents of retained chunks. These describe
  retrieval, before generation flags and token truncation are applied.

This uses the already prefetched neighbor metadata and one batched chunk read per
query. The candidate cap bounds extra chunk reads/reranking, not the existing
graph adjacency prefetch. No new model or index rebuild is required. Tune these
limits on a held-out evaluation set: expansion may add noise, and the usual
context token limit still applies. `use_chunks=False` excludes all chunks from
answer generation even when expansion is enabled.

### GlobalSearchEngine

Community-summary RAG.

- Purpose: evaluate all community summaries for query relevance and synthesize a global answer.
- Best for: broad questions that require corpus-level themes.
- Requires: community summaries built by `KnowledgeGraph`.

```python
from ragu import GlobalSearchEngine, KnowledgeGraph
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = GlobalSearchEngine(llm=llm, knowledge_graph=graph)

print(engine.language)
```

### MixSearchEngine

Ensemble engine.

- Purpose: run child engines and synthesize combined context or combined answers.
- Important parameter: `ensemble_responses`.

```python
from ragu import BuilderArguments, KnowledgeGraph, MixSearchEngine, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
naive = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
mix = MixSearchEngine(llm=llm, engines=[naive])

print(len(mix.engines))
```

### QueryPlanEngine

Wraps any search engine and decomposes a complex query into dependent subqueries: each subquery is executed via the wrapped engine, intermediate answers are fed into subsequent subqueries, and the partial answers are synthesized into a final response. Useful for multi-hop questions that no single retrieval strategy can answer directly.

`stream_query()` is supported when the wrapped engine supports it. Dependency
subqueries are executed normally so their complete answers can be used for
rewriting; only the final sink subquery answer is streamed.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine, QueryPlanEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
planner = QueryPlanEngine(engine)

print(planner.get_prompt("query_decomposition").description)
```

## Data Flow

Input: query string, `KnowledgeGraph`, dense embedder, optional sparse embedder and reranker.

Output:

- `SearchEngineRetrieve` from `search`
- `SearchEngineResponse` from `query`
- `SearchEngineStreamEvent` deltas from `stream_query`

Used by: applications that need GraphRAG answers, retrieval diagnostics, or mixed retrieval strategies.

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(["Python is a programming language."])

    engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    retrieval = await engine.search("What is Python?")
    print(retrieval.result.chunks[0].content)


asyncio.run(main())
```

### Example 2 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, GlobalSearchEngine, KnowledgeGraph, LocalSearchEngine, MixSearchEngine
from ragu.graph.types import CommunitySummary, Entity, Relation
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        builder_settings=BuilderArguments(make_community_summary=False),
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        subject_id=guido.id,
        object_id=python.id,
        subject_name=guido.entity_name,
        object_name=python.entity_name,
        relation_type="CREATED",
        description="Guido van Rossum created Python.",
        source_chunk_id=["chunk-1"],
    )
    await graph.upsert_entities([python, guido])
    await graph.upsert_relations([relation])
    await graph.upsert_summaries([CommunitySummary(id="com-1", summary="Python has a creator.")])

    local = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    global_ = GlobalSearchEngine(llm=llm, knowledge_graph=graph)
    mix = MixSearchEngine(llm=llm, engines=[local, global_])

    response = await mix.query("Summarize the main people and organizations.")
    print(response.response)

    async for event in mix.stream_query("Summarize the main people and organizations."):
        print(event.delta, end="")


asyncio.run(main())
```

### Example 3 - Override a search engine instruction

Every search engine inherits `RaguGenerativeModule`, so `get_prompt` /
`update_prompt` work out of the box:

```python
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
engine.get_prompt("naive_search")            # inspect current instruction
engine.update_prompt("naive_search", my_instruction)  # override for this instance only
```

For a full custom-message example see
[`ragu/common/prompts/README.md`](../common/prompts/README.md). Do **not** edit
`DEFAULT_PROMPT_TEMPLATES` directly; `update_prompt` scopes the change to a
single engine instance.

## Integration Points

- LLMs: every `query()` delegates to `batch_query()`; leaf `batch_query()` implementations render one prompt per query and issue a single `llm.batch_chat_completion`. `stream_query()` renders one final-answer prompt and calls `llm.stream_chat_completion`.
- Embedders: local and naive search encode query text through `GraphRetriever`.
- Sparse embedders: local and naive search can issue hybrid dense+sparse vector queries when storage supports sparse vectors.
- Qdrant: hybrid search is handled by vector storage, including Qdrant prefetch/fusion in `QdrantVectorDBStorage`.
- Other modules: search engines read graph artifacts through `KnowledgeGraph.index`.

## Configuration

Shared engine parameters:

- `language`: prompt language, defaulting to `Settings.language`.

Engine token limits are configured centrally via `Settings.llm_context_token_limit`,
`Settings.tokenizer_llm_backend`, and `Settings.tokenizer_llm_name`.

For per-instance control (e.g. several engines with different LLMs / context
windows in the same process), every engine also accepts these constructor
parameters, each defaulting to `None`:

- `max_context_length` (`int | None`) — falls back to `Settings.llm_context_token_limit`.
- `tokenizer_backend` (`Literal["tiktoken", "local"] | None`) — falls back to `Settings.tokenizer_llm_backend`.
- `tokenizer_model` (`str | None`) — falls back to `Settings.tokenizer_llm_name`.

```python
engine = LocalSearchEngine(
    llm=llm, knowledge_graph=kg, embedder=emb,
    max_context_length=16_000,
    tokenizer_model="gpt-4o",
)
```

> Note: an override on `MixSearchEngine` affects **only** its own final-context
> truncation and is **not** propagated to the child engines — each child keeps
> the tokenizer configuration it was constructed with.

Retrieval parameters:

- `top_k`: initial result count for local and naive search.
- `rerank_top_k`: final chunk count for `NaiveSearchEngine` when a reranker is configured.
- `use_summary` and `use_chunks`: toggles for `LocalSearchEngine.query`.
- `allow_partial_failures`: controls `MixSearchEngine` child-engine failures.

## Dependencies

Internal:

- `ragu.graph.KnowledgeGraph`
- `ragu.graph.GraphRetriever`
- `ragu.models`
- `ragu.common.prompts`
- `ragu.utils.token_truncation`

External:

- `jinja2`
- `pydantic`
- `typing_extensions`

## Notes / Pitfalls

- `GlobalSearchEngine` needs community summaries; build with `make_community_summary=True` or call `reindex_community()`.
- `LocalSearchEngine` starts from entity vector search, so it needs entity vectors, not just chunk vectors.
- `NaiveSearchEngine` works with `build_only_vector_context=True`.
- `MixSearchEngine` requires at least one child engine.
- `search()` is useful for debugging retrieval before involving the LLM.

#### Optional acceptance guard

`EvidenceSearchParams(recovery_mode="after_review", second_recovery_policy="deeper", recovery_guard="chain")` verifies a recovered answer before returning it. `recovery_guard` defaults to `"off"`; alternatives are `"independent"` (answer-blind reconstruction) and `"intersection"` (both checks, two calls). It also works with `review_unknown_answers=True`. Only an initial literal `UNKNOWN` followed by a substantive answer triggers verification. Acceptance preserves the exact candidate; rejection restores the original answer, evidence and retrieval. Existing substantive answers are never checked by this option.

Each check requires a complete chain, verbatim context quotes and conservative answer agreement. This is a fallible LLM check, not a guarantee of correctness. Transport errors propagate. `response.payload["answer_guard"]` records initial/candidate answers, candidate context, proofs, rejection reasons and exact messages; it may contain source text. Defaults use the same LLM as answer generation. Customize registered `evidence_guard_chain` and `evidence_guard_independent` prompts with `update_prompt`, preserving the registered Proof schema. No gold labels enter these calls.


The opt-in `recovery_guard="monotone"` option preserves strict accepts and uses a source-span check plus an independent full-context challenge on strict rejections. See [configuration, audit and evaluation status](../../docs/en/monotone_recovery.md).
