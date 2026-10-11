"""Exercise optional recovery routing, final context, and bounded retrieval."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES
from ragu.chunker.types import Chunk
from ragu.search_engine.local_search import LocalSearchResult


def make_engine():
    llm = SimpleNamespace(chat_completion=AsyncMock(), batch_chat_completion=AsyncMock())
    engine = EvidenceSearchEngine(llm, SimpleNamespace(index=SimpleNamespace()), SimpleNamespace(),
        max_context_length=8000, tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
    engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
    return engine


def answer(value):
    return DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model(answer=value, evidence=[])


@pytest.mark.parametrize('mode', ['bad', True, None])
def test_invalid_mode(mode):
    with pytest.raises(ValueError):
        EvidenceSearchParams(recovery_mode=mode)


def test_duplicate_recovery_settings_rejected():
    with pytest.raises(ValueError):
        EvidenceSearchParams(recovery_mode='after_review', review_unknown_answers=True)


@pytest.mark.parametrize('kwargs', [
    {'second_recovery_policy': 'unbounded'},
    {'second_recovery_policy': 'repeat'},
    {'second_recovery_policy': 'repeat', 'recovery_mode': 'direct'},
])
def test_second_recovery_rejects_unvalidated_modes(kwargs):
    with pytest.raises(ValueError):
        EvidenceSearchParams(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('policy,second_top_k', [('repeat', 5), ('deeper', 20)])
async def test_second_recovery_uses_latest_proposal_context_and_preserves_initial(policy, second_top_k):
    engine = make_engine()
    old = EvidenceSearchRetrieve('q', LocalSearchResult(), rendered_context='Initial context')
    engine.batch_search = AsyncMock(return_value=[old])
    engine.llm.batch_chat_completion.side_effect = [[answer('UNKNOWN')], [answer('UNKNOWN')]]
    plan = DEFAULT_PROMPT_TEMPLATES['evidence_recovery'].pydantic_model(queries=['where'])
    latest = DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model(
        answer='UNKNOWN', evidence=[{'title': 'Stage one', 'fact': 'Newly retrieved fact'}])
    engine.llm.chat_completion.side_effect = [plan, latest, plan, answer('recovered')]
    first, second = Chunk('First addition', 0, 'one'), Chunk('Second addition', 0, 'two')
    engine.retriever.query_chunks = AsyncMock(side_effect=[
        [SimpleNamespace(chunks=[first])], [SimpleNamespace(chunks=[second])]])
    result = await engine.query('q', EvidenceSearchParams(recovery_mode='after_review', second_recovery_policy=policy))
    calls = engine.llm.chat_completion.call_args_list
    payload = json.loads(calls[2].args[0][0]['content'].split('\n', 1)[1])
    assert payload['previous_answer'] == latest.model_dump()
    assert 'First addition' in payload['context']
    assert str(result.retrieval).startswith('**Additional source paragraphs**\nSecond addition')
    assert len(result.payload['answer_recovery']['rounds']) == 2
    assert result.payload['answer_recovery']['initial'] == {'answer': 'UNKNOWN', 'evidence': []}
    assert result.payload['answer_recovery']['initial_context'] == 'Initial context'
    assert str(old) == 'Initial context'
    assert engine.llm.chat_completion.await_count == 4
    assert [call.kwargs['top_k'] for call in engine.retriever.query_chunks.call_args_list] == [5, second_top_k]
    assert [stage['top_k'] for stage in result.payload['answer_recovery']['rounds']] == [5, second_top_k]


@pytest.mark.asyncio
async def test_second_recovery_stops_after_substantive_first_recovery():
    engine = make_engine()
    engine.batch_search = AsyncMock(return_value=[EvidenceSearchRetrieve('q', LocalSearchResult())])
    engine.llm.batch_chat_completion.side_effect = [[answer('UNKNOWN')], [answer('UNKNOWN')]]
    engine.llm.chat_completion.side_effect = [
        DEFAULT_PROMPT_TEMPLATES['evidence_recovery'].pydantic_model(queries=[]), answer('known')]
    result = await engine.query('q', EvidenceSearchParams(recovery_mode='after_review', second_recovery_policy='repeat'))
    assert result.response.answer == 'known'
    assert engine.llm.chat_completion.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['direct', 'after_review'])
async def test_mixed_routing_uses_original_proposal_and_new_retrieval(mode):
    engine = make_engine()
    old = [EvidenceSearchRetrieve(q, LocalSearchResult(chunks=[Chunk('Original facts', 0, 'old')])) for q in ['known', 'unknown']]
    engine.batch_search = AsyncMock(return_value=old)
    original_text = str(old[1])
    initial = [answer('known'), answer('UNKNOWN')]
    reviewed = answer('UNKNOWN')
    reviewed.evidence = [DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model.model_fields['evidence'].annotation.__args__[0](title='review', fact='new review fact')]
    engine.llm.batch_chat_completion.side_effect = [initial, [reviewed]] if mode == 'after_review' else [initial]
    plan = DEFAULT_PROMPT_TEMPLATES['evidence_recovery'].pydantic_model(queries=['where'])
    engine.llm.chat_completion.side_effect = [plan, answer('recovered')]
    new = Chunk('New source\nAnswer fact.', 0, 'new')
    engine.retriever.query_chunks = AsyncMock(return_value=[SimpleNamespace(chunks=[new])])
    result = await engine.batch_query(['known', 'unknown'], EvidenceSearchParams(recovery_mode=mode))
    assert [r.response.answer for r in result] == ['known', 'recovered']
    assert result[0].response is initial[0]
    assert result[1].retrieval is not old[1]
    assert str(old[1]) == original_text
    assert str(result[1].retrieval).startswith('**Additional source paragraphs**\nNew source')
    assert result[1].payload['answer_recovery']['added_ids'] == [new.id]
    payload = json.loads(engine.llm.chat_completion.call_args_list[0].args[0][0]['content'].split('\n', 1)[1])
    assert payload['previous_answer'] == {'answer': 'UNKNOWN', 'evidence': []}
    final = json.loads(engine.llm.chat_completion.call_args_list[1].args[0][0]['content'].split('\n', 1)[1])
    assert final['context'] == str(result[1].retrieval)
    assert engine.llm.chat_completion.await_count == 2
    assert engine.llm.batch_chat_completion.await_count == (2 if mode == 'after_review' else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('queries', [[], ['missing']])
async def test_empty_plan_or_no_hits_still_reviews_original_context(queries):
    engine = make_engine()
    old = EvidenceSearchRetrieve('q', LocalSearchResult(), rendered_context='Original')
    engine.batch_search = AsyncMock(return_value=[old])
    engine.llm.batch_chat_completion.return_value = [answer('UNKNOWN')]
    engine.llm.chat_completion.side_effect = [DEFAULT_PROMPT_TEMPLATES['evidence_recovery'].pydantic_model(queries=queries), answer('UNKNOWN')]
    engine.retriever.query_chunks = AsyncMock(return_value=[SimpleNamespace(chunks=[])] if queries else [])
    result = await engine.query('q', EvidenceSearchParams(recovery_mode='direct'))
    assert str(result.retrieval) == 'Original'
    assert result.payload['answer_recovery']['added_ids'] == []
    assert engine.retriever.query_chunks.await_count == bool(queries)


def test_packing_round_robin_whole_additions_and_budget():
    engine = make_engine()
    a, b = Chunk('A', 0, 'a'), Chunk('B', 0, 'b')
    oversized = Chunk('word ' * 3000, 0, 'huge')
    context, additions = engine._pack_recovery_context('old ' * 9000, [[a, b], [b, oversized]])
    assert [c.id for c in additions] == [a.id, b.id]
    assert len(engine.truncation.encoder.encode(context)) <= 8000
    assert engine._pack_recovery_context('A', [[a]]) == ('A', [])


@pytest.mark.asyncio
async def test_cascade_stops_when_review_succeeds_and_stream_rejected():
    engine = make_engine()
    engine.batch_search = AsyncMock(return_value=[EvidenceSearchRetrieve('q', LocalSearchResult())])
    engine.llm.batch_chat_completion.side_effect = [[answer('UNKNOWN')], [answer('known')]]
    result = await engine.query('q', EvidenceSearchParams(recovery_mode='after_review'))
    assert result.response.answer == 'known'
    engine.llm.chat_completion.assert_not_awaited()
    with pytest.raises(ValueError, match='does not support'):
        async for _ in engine.stream_query('q', EvidenceSearchParams(recovery_mode='direct')):
            pass
