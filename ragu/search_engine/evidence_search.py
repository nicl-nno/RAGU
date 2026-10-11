"""Graph and paragraph retrieval with one evidence-grounded follow-up round."""
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from pydantic import BaseModel
from typing_extensions import override

from ragu.chunker.types import Chunk
from ragu.common.prompts import ChatMessages, DEFAULT_PROMPT_TEMPLATES
from ragu.common.prompts.messages import render
from ragu.common.prompts._grounding_guard import verify as _verify_guard, Proof as _GuardProof
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
    :param review_unknown_answers: Recheck structured UNKNOWN answers once against the same context.
    :param recovery_mode: Search after an initial abstention (direct), after review (after_review), or disable recovery (off).
    :param recovery_guard: Optional source-only acceptance check for recovered UNKNOWN answers.
    :param second_recovery_policy: Repeat recovery once on remaining cascade abstentions, or disable with off.
    """
    neighbor_chunk_limit: int = 10
    dense_top_k: int = 20
    followup_rounds: int = 1
    followup_top_k: int = 10
    planner_seed_chunks: int = 6
    planner_context_tokens: int = 3000
    rrf_k: float = 60.0
    review_unknown_answers: bool = False
    recovery_mode: Literal['off', 'direct', 'after_review'] = 'off'
    second_recovery_policy: Literal['off', 'repeat', 'deeper'] = 'off'
    recovery_guard: Literal['off', 'chain', 'independent', 'intersection'] = 'off'

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.recovery_guard not in ('off', 'chain', 'independent', 'intersection'):
            raise ValueError('recovery_guard must be off, chain, independent, or intersection')
        if self.recovery_guard != 'off' and self.recovery_mode == 'off' and not self.review_unknown_answers:
            raise ValueError('Recovery guard requires recovery or answer review')
        if self.second_recovery_policy not in ('off', 'repeat', 'deeper'):
            raise ValueError('second_recovery_policy must be off, repeat, or deeper')
        if self.second_recovery_policy != 'off' and self.recovery_mode != 'after_review':
            raise ValueError('Second recovery requires recovery_mode=after_review')
        if self.recovery_mode not in ('off', 'direct', 'after_review'):
            raise ValueError('recovery_mode must be off, direct, or after_review')
        if self.recovery_mode != 'off' and self.review_unknown_answers:
            raise ValueError('Use recovery_mode or review_unknown_answers, not both')
        if not isinstance(self.review_unknown_answers, bool):
            raise ValueError('review_unknown_answers must be a boolean')
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

    rendered_context: str | None = None

    @override
    def to_text(self) -> str:
        if self.rendered_context is not None:
            return self.rendered_context
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
        self.update_prompt('evidence_review', DEFAULT_PROMPT_TEMPLATES['evidence_review'])
        self.update_prompt('evidence_recovery', DEFAULT_PROMPT_TEMPLATES['evidence_recovery'])
        for name in ('evidence_guard_chain', 'evidence_guard_independent'):
            self.update_prompt(name, DEFAULT_PROMPT_TEMPLATES[name])
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
        if not queries:
            return []
        params = params or EvidenceSearchParams()
        recovering = params.recovery_mode != 'off'
        review_first = params.review_unknown_answers or params.recovery_mode == 'after_review'
        if recovering and self.truncation.tokenizer_type != 'tiktoken':
            raise ValueError('Answer recovery requires a tiktoken tokenizer')
        if params.review_unknown_answers or recovering:
            for name in ('local_search', 'evidence_review'):
                fields = getattr(self.get_prompt(name).pydantic_model, 'model_fields', {})
                if not {'answer', 'evidence'} <= fields.keys():
                    raise ValueError('Answer review requires structured answer and evidence fields in both prompts')
        results = await super().batch_query(queries, params)
        if not params.review_unknown_answers and not recovering:
            return results
        originals = {id(result): (result.response, result.retrieval) for result in results}
        selected = []
        conversations = []
        for result in results:
            if not isinstance(result.response, BaseModel):
                raise ValueError('Answer review requires structured model responses')
            dumped = result.response.model_dump()
            proposal = {'answer': dumped['answer'], 'evidence': dumped['evidence']}
            if not isinstance(proposal['answer'], str) or not isinstance(proposal['evidence'], list):
                raise ValueError('Answer review requires a string answer and a list of evidence')
            result.payload['answer_review'] = {'applied': False}
            if proposal['answer'].strip().upper() == 'UNKNOWN':
                selected.append((result, proposal))
                conversations.append(self._render_review_messages(result.query, result.retrieval, proposal).to_openai())
        if selected and review_first:
            reviewed = await self.llm.batch_chat_completion(
                conversations, output_schema=self.get_prompt('evidence_review').pydantic_model,
                desc='Evidence answer review',
            )
            if len(reviewed) != len(selected):
                raise ValueError('Answer review returned an unexpected number of responses')
            for (result, proposal), answer in zip(selected, reviewed):
                result.response = answer
                result.payload['answer_review'] = {'applied': True, 'initial': proposal}
        if recovering:
            for result in results:
                result.payload['answer_recovery'] = {'mode': params.recovery_mode, 'applied': False}
            pending = []
            for result, proposal in selected:
                if not hasattr(result.response, 'answer') or not isinstance(result.response.answer, str):
                    raise ValueError('Answer recovery requires a structured string answer')
                if result.response.answer.strip().upper() == 'UNKNOWN':
                    pending.append(self._recover_answer(result, proposal))
            await asyncio.gather(*pending)
            if params.second_recovery_policy != 'off':
                pending = []
                for result in results:
                    if result.response.answer.strip().upper() == 'UNKNOWN':
                        latest = result.response.model_dump()
                        proposal = {key: latest[key] for key in ('answer', 'evidence')}
                        pending.append(self._recover_answer(result, proposal,
                            top_k=20 if params.second_recovery_policy == 'deeper' else 5))
                await asyncio.gather(*pending)
        if params.recovery_guard != 'off':
            for result in results:
                result.payload['answer_guard'] = {'mode': params.recovery_guard, 'applied': False}
            await asyncio.gather(*[
                self._guard_recovered_answer(result, *originals[id(result)], params.recovery_guard)
                for result, _ in selected
                if isinstance(getattr(result.response, 'answer', None), str)
                and result.response.answer.strip().upper() != 'UNKNOWN'
                and result.response.answer.strip()
            ])
        return results

    async def _guard_recovered_answer(self, result: SearchEngineResponse, initial_response: BaseModel,
                                      initial_retrieval: LocalSearchRetrieve, mode: str) -> None:
        """Accept an unchanged recovered answer or restore the exact initial abstention."""
        engine = self
        class PromptAdapter:
            def __init__(self, stage):
                self.stage = stage
                self.messages = None

            async def chat_completion(self, messages, output_schema):
                instruction = engine.get_prompt(self.stage)
                if instruction.pydantic_model is not _GuardProof:
                    raise ValueError('Recovery guard requires the registered Proof schema')
                self.messages = render(instruction.messages, payload=messages[1]['content'])[0].to_openai()
                return await engine.llm.chat_completion(self.messages, output_schema=output_schema)

        proposal = result.response.answer
        context = self.truncation(str(result.retrieval))
        checks = {}
        modes = ('chain', 'independent') if mode == 'intersection' else (mode,)
        for check in modes:
            stage = 'evidence_guard_' + check
            adapter = PromptAdapter(stage)
            decision = await _verify_guard(result.query, context, proposal, adapter, mode=check)
            decision['messages'] = adapter.messages or decision['messages']
            decision['stage'] = stage
            checks[check] = decision
        accepted = all(check['accepted'] for check in checks.values())
        result.payload['answer_guard'] = {'mode': mode, 'applied': True, 'accepted': accepted,
            'initial': initial_response.model_dump(), 'candidate': result.response.model_dump(),
            'candidate_context': context, 'checks': checks}
        if not accepted:
            result.response = initial_response
            result.retrieval = initial_retrieval

    async def _recover_answer(self, result: SearchEngineResponse, proposal: dict, *, top_k: int = 5) -> None:
        """Search and review from the original proposal without mutating its retrieval."""
        old = result.retrieval
        text = self.truncation(str(old))
        instruction = self.get_prompt('evidence_recovery')
        messages = render(instruction.messages, payload=json.dumps(
            {'question': result.query, 'context': text, 'previous_answer': proposal},
            ensure_ascii=False,
        ))[0]
        plan = await self.llm.chat_completion(messages.to_openai(), output_schema=instruction.pydantic_model)
        if not hasattr(plan, 'queries') or not isinstance(plan.queries, list) or len(plan.queries) > 2 or any(not isinstance(q, str) for q in plan.queries):
            raise ValueError('Recovery planner must return at most two string queries')
        queries = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))
        hits = await self.retriever.query_chunks(queries, top_k=top_k) if queries else []
        context, additions = self._pack_recovery_context(text, [list(hit.chunks) for hit in hits])
        merged = {c.id: c for c in old.result.chunks}
        merged.update({c.id: c for c in additions})
        retrieval = replace(old, result=replace(old.result, chunks=list(merged.values()),
            documents_id=sorted(set(old.result.documents_id) | {c.doc_id for c in additions})),
            metrics=dict(old.metrics), rendered_context=context)
        answer = await self.llm.chat_completion(
            self._render_review_messages(result.query, retrieval, proposal).to_openai(),
            output_schema=self.get_prompt('evidence_review').pydantic_model,
        )
        result.retrieval = retrieval
        result.response = answer
        stage = {'proposal': proposal, 'queries': queries, 'top_k': top_k,
                 'hit_ids': [[c.id for c in hit.chunks] for hit in hits],
                 'added_ids': [c.id for c in additions], 'initial_context': text,
                 'final_context': context}
        previous = result.payload['answer_recovery']
        result.payload['answer_recovery'] = {
            **result.payload['answer_recovery'], 'applied': True,
            'initial': previous.get('initial', proposal), 'queries': queries,
            'hit_ids': [[c.id for c in hit.chunks] for hit in hits],
            'added_ids': [c.id for c in additions], 'initial_context': previous.get('initial_context', text),
            'final_context': context,
            'rounds': [*previous.get('rounds', []), stage],
        }

    def _pack_recovery_context(self, old_context: str, hit_lists: list[list[Chunk]]) -> tuple[str, list[Chunk]]:
        """Prefix whole new paragraphs in rank round-robin order within a fixed budget."""
        encoder = self.truncation.encoder
        selected: list[Chunk] = []
        seen: set[str] = set()
        for rank in range(max((len(hits) for hits in hit_lists), default=0)):
            for hits in hit_lists:
                if rank >= len(hits):
                    continue
                chunk = hits[rank]
                if chunk.id in seen or chunk.content in old_context:
                    continue
                seen.add(chunk.id)
                prefix = '**Additional source paragraphs**\n' + '\n\n'.join(c.content for c in [*selected, chunk]) + '\n\n'
                if len(encoder.encode(prefix)) <= min(2000, self.truncation.max_tokens):
                    selected.append(chunk)
        if not selected:
            return old_context, []
        prefix = '**Additional source paragraphs**\n' + '\n\n'.join(c.content for c in selected) + '\n\n'
        return self.truncation(prefix + old_context), selected

    def _render_review_messages(self, query: str, context: LocalSearchRetrieve, proposal: dict) -> ChatMessages:
        """Review the same truncated source text, without exposing evaluation data."""
        return render(self.get_prompt('evidence_review').messages, payload=json.dumps(
            {'question': query, 'context': self.truncation(str(context)), 'proposed_answer': proposal},
            ensure_ascii=False,
        ))[0]

    @override
    async def stream_query(self, query: str, params: EvidenceSearchParams | None = None) -> AsyncIterator[SearchEngineStreamEvent]:
        """Finish evidence planning/retrieval before streaming the final answer."""
        params = params or EvidenceSearchParams()
        if params.review_unknown_answers or params.recovery_mode != 'off':
            raise ValueError('stream_query does not support answer review; use query instead')
        if self.get_prompt('local_search').pydantic_model is not str:
            raise ValueError('stream_query requires a plain-text answer prompt; use query for structured answers')
        async for event in super().stream_query(query, params):
            yield event
