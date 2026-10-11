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
    {'review_unknown_answers': 1}, {'review_unknown_answers': 'true'},
])
def test_rejects_unbounded_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        EvidenceSearchParams(**kwargs)


def test_opt_in_defaults_leave_local_search_unchanged():
    assert EvidenceSearchParams().neighbor_chunk_limit == 10
    assert EvidenceSearchParams().review_unknown_answers is False
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


@pytest.mark.asyncio
async def test_structured_answer_payload_is_bounded_and_returned_without_extra_call():
    import json
    from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

    engine = make_engine({})
    chunk = Chunk('Source title\n' + 'facts ' * 100, 0, 'doc')
    retrieval = EvidenceSearchRetrieve('quoted "question"', LocalSearchResult(chunks=[chunk]))
    engine.batch_search = AsyncMock(return_value=[retrieval])
    instruction = DEFAULT_PROMPT_TEMPLATES['evidence_answer']
    engine.update_prompt('local_search', instruction)
    answer = instruction.pydantic_model.model_validate({'evidence': [{'title': 'Source title', 'fact': 'facts'}], 'answer': 'answer'})
    engine.llm.batch_chat_completion = AsyncMock(return_value=[answer])
    result = await engine.query('quoted "question"')
    assert result.response.answer == 'answer'
    messages = engine.llm.batch_chat_completion.call_args.args[0][0]
    payload = json.loads(messages[0]['content'].split('\n', 1)[1])
    assert payload['question'] == 'quoted "question"'
    assert len(engine.truncation.encoder.encode(payload['context'])) <= 80
    engine.llm.batch_chat_completion.assert_awaited_once()
    engine.llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_structured_stream_rejected_before_retrieval():
    from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

    engine = make_engine({})
    engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
    engine.batch_search = AsyncMock()
    with pytest.raises(ValueError, match='plain-text'):
        async for _ in engine.stream_query('q'):
            pass
    engine.batch_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_only_abstentions_preserves_batch_alignment_and_bounded_sources():
    import json
    from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

    engine = make_engine({})
    instruction = DEFAULT_PROMPT_TEMPLATES['evidence_answer']
    review_instruction = DEFAULT_PROMPT_TEMPLATES['evidence_review']
    engine.update_prompt('local_search', instruction)
    first = [instruction.pydantic_model.model_validate({
        'evidence': [{'title': 'A', 'fact': 'initial fact'}], 'answer': value,
    }) for value in [' unknown ', 'supported answer', 'UNKNOWN']]
    reviewed = [review_instruction.pydantic_model.model_validate({
        'evidence': [{'title': 'A', 'fact': 'reviewed fact'}], 'answer': value,
    }) for value in ['recovered', 'UNKNOWN']]
    contexts = [EvidenceSearchRetrieve(q, LocalSearchResult(chunks=[
        Chunk('A\n' + 'source fact ' * 200, 0, 'doc')])) for q in ['q1', 'q2', 'q3']]
    engine.batch_search = AsyncMock(return_value=contexts)
    engine.llm.batch_chat_completion = AsyncMock(side_effect=[first, reviewed])

    results = await engine.batch_query(['q1', 'q2', 'q3'], EvidenceSearchParams(review_unknown_answers=True))
    assert [r.response.answer for r in results] == ['recovered', 'supported answer', 'UNKNOWN']
    assert results[1].response is first[1]
    assert first[0].answer == ' unknown '
    assert results[0].payload['answer_review']['initial']['answer'] == ' unknown '
    assert results[1].payload['answer_review'] == {'applied': False}
    engine.batch_search.assert_awaited_once()
    calls = engine.llm.batch_chat_completion.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs['output_schema'] is review_instruction.pydantic_model
    payloads = [json.loads(messages[0]['content'].split('\n', 1)[1]) for messages in calls[1].args[0]]
    assert [p['question'] for p in payloads] == ['q1', 'q3']
    for payload, original_messages in zip(payloads, [calls[0].args[0][0], calls[0].args[0][2]]):
        original = json.loads(original_messages[0]['content'].split('\n', 1)[1])
        assert payload['context'] == original['context']
        assert len(engine.truncation.encoder.encode(payload['context'])) <= 80
        assert payload['proposed_answer']['evidence'] == [{'title': 'A', 'fact': 'initial fact'}]


@pytest.mark.asyncio
async def test_review_skips_extra_inference_when_every_answer_is_substantive():
    from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES

    engine = make_engine({})
    instruction = DEFAULT_PROMPT_TEMPLATES['evidence_answer']
    engine.update_prompt('local_search', instruction)
    answer = instruction.pydantic_model.model_validate({'evidence': [], 'answer': 'known'})
    engine.batch_search = AsyncMock(return_value=[EvidenceSearchRetrieve('q', LocalSearchResult())])
    engine.llm.batch_chat_completion = AsyncMock(return_value=[answer])
    result = await engine.query('q', EvidenceSearchParams(review_unknown_answers=True))
    assert result.response is answer
    engine.llm.batch_chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_rejects_plain_prompt_and_stream_before_retrieval():
    engine = make_engine({})
    engine.batch_search = AsyncMock()
    params = EvidenceSearchParams(review_unknown_answers=True)
    with pytest.raises(ValueError, match='structured'):
        await engine.query('q', params)
    with pytest.raises(ValueError, match='does not support answer review'):
        async for _ in engine.stream_query('q', params):
            pass
    engine.batch_search.assert_not_awaited()
