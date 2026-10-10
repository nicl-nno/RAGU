from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
from ragu.chunker.types import Chunk
from ragu.search_engine.local_search import LocalParams, LocalSearchRetrieve, LocalSearchResult


def make_engine(vectors, plan=None):
    storage = SimpleNamespace(get_points_by_ids=AsyncMock(
        side_effect=lambda ids: [SimpleNamespace(dense_embedding=vectors[i]) if i in vectors else None for i in ids]))
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value=SimpleNamespace(queries=plan or [])))
    embedder = SimpleNamespace(batch_embed_text=AsyncMock(return_value=[[0., 1.]]))
    engine = EvidenceSearchEngine(llm, SimpleNamespace(index=SimpleNamespace(chunks_vector_db=storage)),
                                  embedder, max_context_length=80,
                                  tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
    return engine


@pytest.mark.parametrize('kwargs', [
    {'followup_rounds': 1.0}, {'followup_rounds': True}, {'followup_rounds': 2},
    {'dense_top_k': 0}, {'planner_context_tokens': -1}, {'followup_top_k': False},
    {'rrf_k': float('nan')}, {'rrf_k': float('inf')}, {'rrf_k': 0},
])
def test_rejects_unbounded_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        EvidenceSearchParams(**kwargs)


def test_opt_in_defaults_leave_local_search_unchanged():
    assert EvidenceSearchParams().neighbor_chunk_limit == 10
    assert LocalParams().neighbor_chunk_limit == 0


@pytest.mark.asyncio
async def test_fusion_rewards_shared_evidence_and_deduplicates_queries():
    a, b, c = [Chunk(t, 0, t) for t in ('initial anchor', 'shared bridge', 'new answer')]
    engine = make_engine({a.id: [1., 0.], b.id: [0.8, 0.6]}, [' follow link ', 'follow link', ' '])
    engine.retriever.query_chunks = AsyncMock(return_value=[SimpleNamespace(chunks=[b, c])])
    local = LocalSearchRetrieve('question', LocalSearchResult(chunks=[a], documents_id=['initial anchor']))
    result = await engine._expand_evidence('question', local, [a, b], [1., 0.], EvidenceSearchParams())
    assert [x.id for x in result.result.chunks] == [b.id, a.id, c.id]
    assert result.metrics['followup_queries'] == ['follow link']
    assert result.metrics['followup_chunks'] == [c.id]
    assert result.result.documents_id == ['initial anchor', 'new answer', 'shared bridge']
    engine.retriever.query_chunks.assert_awaited_once_with(['follow link'], top_k=10)
    assert local.result.chunks == [a]


@pytest.mark.asyncio
async def test_no_planning_mode_falls_back_for_missing_vectors_and_keeps_ties_stable():
    a, b, c = [Chunk(t, 0, 'doc') for t in ('a', 'b', 'c')]
    engine = make_engine({a.id: [1., 0.], b.id: [1., 0.]})
    local = LocalSearchRetrieve('q', LocalSearchResult(chunks=[b, a]))
    result = await engine._expand_evidence('q', local, [a, c], [1., 0.], EvidenceSearchParams(followup_rounds=0))
    assert result.result.chunks == [b, a, c]
    engine.llm.chat_completion.assert_not_awaited()
    engine.retriever.embedder.batch_embed_text.assert_awaited_once_with(['c'])


@pytest.mark.asyncio
async def test_empty_plan_avoids_followup_retrieval_and_bounds_planner_context():
    a = Chunk('paragraph ' * 300, 0, 'doc')
    engine = make_engine({a.id: [1., 0.]})
    engine.retriever.query_chunks = AsyncMock()
    local = LocalSearchRetrieve('q', LocalSearchResult(chunks=[a]))
    result = await engine._expand_evidence('q', local, [a], [1., 0.], EvidenceSearchParams(planner_context_tokens=12))
    engine.retriever.query_chunks.assert_not_awaited()
    import json
    message = engine.llm.chat_completion.call_args.args[0][0]['content']
    payload = json.loads(message.split('\n', 1)[1])
    assert len(engine.truncation.encoder.encode(payload['evidence'])) <= 12
    assert result.metrics['planner_calls'] == 1
    assert str(result).startswith('**Chunks**\n')
    assert len(engine.truncation.encoder.encode(engine.truncation(str(result)))) <= 80


@pytest.mark.asyncio
async def test_invalid_custom_plan_fails_explicitly():
    engine = make_engine({}, ['one', 'two', 'three', 'four'])
    with pytest.raises(ValueError, match='at most three'):
        await engine._expand_evidence('q', LocalSearchRetrieve('q', LocalSearchResult()), [], [1., 0.], EvidenceSearchParams())


@pytest.mark.asyncio
async def test_empty_batch_performs_no_inference():
    engine = make_engine({})
    assert await engine.batch_search([]) == []
    assert await engine.batch_query([]) == []
    engine.llm.chat_completion.assert_not_awaited()
