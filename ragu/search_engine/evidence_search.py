"""Graph and paragraph retrieval with one evidence-grounded follow-up round."""
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from typing_extensions import override

from ragu.chunker.types import Chunk
from ragu.common.prompts import ChatMessages, DEFAULT_PROMPT_TEMPLATES
from ragu.common.prompts.messages import render
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.search_engine.base_engine import SearchEngineResponse, SearchEngineStreamEvent
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine, LocalSearchRetrieve
from ragu.utils.token_truncation import TokenTruncation


@dataclass
class EvidenceSearchParams(LocalParams):
    """Parameters for graph/dense union and optional evidence-grounded retrieval.

    :param dense_top_k: Direct paragraph hits added to graph candidates.
    :param followup_rounds: Zero disables planning; one performs a bounded round.
    :param followup_top_k: Paragraph hits per planned query (at most three queries).
    :param planner_seed_chunks: Initial direct hits shown to the planner.
    :param planner_context_tokens: Separate budget for planner evidence.
    :param rrf_k: Positive reciprocal-rank fusion offset.
    """
    neighbor_chunk_limit: int = 10
    dense_top_k: int = 20
    followup_rounds: int = 1
    followup_top_k: int = 10
    planner_seed_chunks: int = 6
    planner_context_tokens: int = 3000
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ('dense_top_k', 'followup_top_k', 'planner_seed_chunks', 'planner_context_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.followup_rounds, bool) or not isinstance(self.followup_rounds, int) or self.followup_rounds not in (0, 1):
            raise ValueError('followup_rounds must be zero or one')
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, (int, float)) or not np.isfinite(self.rrf_k) or self.rrf_k <= 0:
            raise ValueError('rrf_k must be finite and positive')


@dataclass(slots=True)
class EvidenceSearchRetrieve(LocalSearchRetrieve):
    """Local graph results with source paragraphs rendered before graph descriptions."""

    @override
    def to_text(self) -> str:
        graph = LocalSearchRetrieve(query=self.query, result=replace(self.result, chunks=[])).to_text().rstrip()
        sections = []
        if self.result.chunks:
            sections.append('**Chunks**\n' + '\n'.join(c.content for c in self.result.chunks))
        if graph:
            sections.append(graph)
        return '\n\n'.join(sections)


class EvidenceSearchEngine(LocalSearchEngine):
    """Combine graph and dense paragraph candidates, then retrieve missing links.

    This opt-in engine preserves LocalSearchEngine's answer prompt interface.
    Search itself may call the LLM once per question to plan at most three
    evidence-grounded searches. Set ``followup_rounds=0`` for retrieval without
    planning. Source paragraphs lead the final bounded context; graph facts
    fill the remaining space. Existing local-search defaults are unchanged.
    """

    def __init__(
        self, llm: LLM, knowledge_graph: KnowledgeGraph, embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None, language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal['tiktoken', 'local'] | None = None,
        tokenizer_model: str | None = None,
    ) -> None:
        """Initialize with the same storage/model/tokenizer options as local search."""
        super().__init__(llm, knowledge_graph, embedder, sparse_embedder, reranker,
                         language, max_context_length, tokenizer_backend, tokenizer_model)
        self.update_prompt('evidence_followup', DEFAULT_PROMPT_TEMPLATES['evidence_followup'])
        self._planner_truncations: dict[int, TokenTruncation] = {}

    async def _rank_chunks(self, query_vector: list[float], chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        points = await self.knowledge_graph.index.chunks_vector_db.get_points_by_ids([c.id for c in chunks])
        missing = [i for i, point in enumerate(points) if point is None or point.dense_embedding is None]
        replacements = await self.retriever.embedder.batch_embed_text([chunks[i].content for i in missing]) if missing else []
        fallback = dict(zip(missing, replacements))
        vectors = np.asarray([fallback[i] if i in fallback else point.dense_embedding for i, point in enumerate(points)], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        scores = np.divide(vectors @ query, denominator, out=np.zeros(len(chunks), dtype=np.float32), where=denominator != 0)
        return [chunks[int(i)] for i in np.argsort(-scores, kind='stable')]

    async def _expand_evidence(self, query: str, local: LocalSearchRetrieve, direct: list[Chunk],
                               query_vector: list[float], params: EvidenceSearchParams) -> EvidenceSearchRetrieve:
        # Preserve graph candidate insertion order for deterministic score ties.
        union = {chunk.id: chunk for chunk in local.result.chunks}
        for chunk in direct:
            union.setdefault(chunk.id, chunk)
        ordered = await self._rank_chunks(query_vector, list(union.values()))
        planned: list[str] = []
        added: list[str] = []
        if params.followup_rounds:
            truncation = self._planner_truncations.get(params.planner_context_tokens)
            if truncation is None:
                truncation = TokenTruncation(self.truncation.model_id, self.truncation.tokenizer_type, params.planner_context_tokens)
                self._planner_truncations[params.planner_context_tokens] = truncation
            evidence = truncation('\n\n'.join(c.content for c in direct[:params.planner_seed_chunks]))
            instruction = self.get_prompt('evidence_followup')
            messages = render(instruction.messages, payload=json.dumps({'question': query, 'evidence': evidence}, ensure_ascii=False))[0]
            plan = await self.llm.chat_completion(messages.to_openai(), output_schema=instruction.pydantic_model)
            if not hasattr(plan, 'queries') or not isinstance(plan.queries, list) or len(plan.queries) > 3 or any(not isinstance(q, str) for q in plan.queries):
                raise ValueError('Evidence planner must return at most three string queries')
            planned = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))
            fused = {chunk.id: 1 / (params.rrf_k + rank) for rank, chunk in enumerate(ordered, 1)}
            followups = await self.retriever.query_chunks(planned, top_k=params.followup_top_k) if planned else []
            for hits in followups:
                for rank, chunk in enumerate(hits.chunks, 1):
                    if chunk.id not in union:
                        union[chunk.id] = chunk
                        added.append(chunk.id)
                    fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (params.rrf_k + rank)
            ordered = sorted(union.values(), key=lambda c: -fused[c.id])
        return EvidenceSearchRetrieve(
            query=query, result=replace(local.result, chunks=ordered,
                documents_id=sorted(set(local.result.documents_id) | {c.doc_id for c in ordered})),
            metrics={**local.metrics, 'dense_chunks': [c.id for c in direct],
                     'followup_queries': planned, 'followup_chunks': added,
                     'planner_calls': params.followup_rounds},
        )

    @override
    async def batch_search(self, queries: list[str], params: EvidenceSearchParams | None = None) -> list[EvidenceSearchRetrieve]:
        """Retrieve source evidence; planning uses only the question and initial texts."""
        if not queries:
            return []
        params = params or EvidenceSearchParams()
        local, direct, vectors = await asyncio.gather(
            super().batch_search(queries, params),
            self.retriever.query_chunks(queries, top_k=params.dense_top_k),
            self.retriever.embedder.batch_embed_text(queries),
        )
        return list(await asyncio.gather(*[
            self._expand_evidence(query, context, hits.chunks, vector, params)
            for query, context, hits, vector in zip(queries, local, direct, vectors)
        ]))

    @override
    def _render_answer_messages(
        self, query: str, context: LocalSearchRetrieve,
        use_summary: bool, use_chunks: bool,
    ) -> ChatMessages:
        """Expose ordinary prompt fields plus a JSON payload for structured answers."""
        if not use_summary:
            context.result.summaries = []
        if not use_chunks:
            context.result.chunks = []
        text = self.truncation(str(context))
        instruction = self.get_prompt('local_search')
        return render(instruction.messages, query=query, context=text, language=self.language,
                      payload=json.dumps({'question': query, 'context': text}, ensure_ascii=False))[0]

    @override
    async def batch_query(self, queries: list[str], params: EvidenceSearchParams | None = None) -> list[SearchEngineResponse]:
        """Answer from the expanded, token-bounded context using the local answer prompt."""
        return await super().batch_query(queries, params or EvidenceSearchParams())

    @override
    async def stream_query(self, query: str, params: EvidenceSearchParams | None = None) -> AsyncIterator[SearchEngineStreamEvent]:
        """Finish evidence planning/retrieval before streaming the final answer."""
        if self.get_prompt('local_search').pydantic_model is not str:
            raise ValueError('stream_query requires a plain-text answer prompt; use query for structured answers')
        async for event in super().stream_query(query, params or EvidenceSearchParams()):
            yield event
