"""Public answer pipeline routing and frozen guard prompt parity."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
from ragu.chunker.types import Chunk
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES
from ragu.common.prompts._grounding_guard import Proof, verify
from ragu.search_engine.local_search import LocalSearchResult


def answer(value):
    return DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model(answer=value, evidence=[])


def proof(accepted=True):
    return Proof(constraints=['person birthplace'], steps=[dict(claim='person birthplace',
        quotes=['Alice was born in Paris.'], supported=True, constraint_indices=[0])],
        missing_links=[] if accepted else ['identity missing'], ambiguous=False,
        uses_external_knowledge=False, complete=accepted, answer='Paris', rationale='audit')


def engine_for(initial, review):
    llm = SimpleNamespace(chat_completion=AsyncMock(), batch_chat_completion=AsyncMock(side_effect=[[initial], [review]]))
    engine = EvidenceSearchEngine(llm, SimpleNamespace(index=SimpleNamespace()), SimpleNamespace(),
        max_context_length=8000, tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
    engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
    retrieval = EvidenceSearchRetrieve('Where was Alice born?', LocalSearchResult(), rendered_context='Alice was born in Paris.')
    engine.batch_search = AsyncMock(return_value=[retrieval])
    return engine, retrieval


@pytest.mark.parametrize('value', ['unknown', True, None])
def test_invalid_mode(value):
    with pytest.raises(ValueError):
        EvidenceSearchParams(recovery_guard=value)


def test_guard_requires_recovery_or_review():
    with pytest.raises(ValueError):
        EvidenceSearchParams(recovery_guard='chain')


async def test_initial_known_is_never_checked():
    initial = answer('Paris')
    engine, retrieval = engine_for(initial, answer('ignored'))
    result = await engine.query('Where was Alice born?', EvidenceSearchParams(
        recovery_mode='after_review', second_recovery_policy='deeper', recovery_guard='intersection'))
    assert result.response is initial and result.retrieval is retrieval
    assert result.payload['answer_guard']['applied'] is False
    engine.llm.chat_completion.assert_not_awaited()
    assert engine.llm.batch_chat_completion.await_count == 1


@pytest.mark.parametrize('mode,calls', [('chain', 1), ('independent', 1), ('intersection', 2)])
@pytest.mark.parametrize('accepted', [True, False])
async def test_review_recovery_keeps_exact_candidate_or_restores_initial(mode, calls, accepted):
    initial, candidate = answer(' UNKNOWN '), answer('Paris')
    engine, old = engine_for(initial, candidate)
    engine.llm.chat_completion.return_value = proof(accepted)
    result = await engine.query('Where was Alice born?', EvidenceSearchParams(
        review_unknown_answers=True, recovery_guard=mode))
    assert result.response is (candidate if accepted else initial)
    assert result.retrieval is old
    assert result.payload['answer_guard']['accepted'] is accepted
    assert engine.llm.chat_completion.await_count == calls
    for check, audit in result.payload['answer_guard']['checks'].items():
        frozen = SimpleNamespace(chat_completion=AsyncMock(return_value=proof(accepted)))
        expected = await verify(result.query, str(old), 'Paris', frozen, mode=check)
        assert audit['messages'] == expected['messages']
        assert audit['proof'] == expected['proof']
        assert audit['stage'] == 'evidence_guard_' + check
        assert frozen.chat_completion.call_args.kwargs['output_schema'].model_json_schema() == Proof.model_json_schema()


async def test_reject_restores_original_retrieval_and_preserves_recovery_audit():
    initial = DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model.model_validate(
        {'answer': 'UNKNOWN', 'evidence': [{'title': 'Original', 'fact': 'Incomplete evidence'}]})
    engine, old = engine_for(initial, answer('UNKNOWN'))
    plan = DEFAULT_PROMPT_TEMPLATES['evidence_recovery'].pydantic_model(queries=['Alice birthplace'])
    engine.retriever.query_chunks = AsyncMock(return_value=[SimpleNamespace(
        chunks=[Chunk('New source with another Alice.', 0, 'new')])])
    engine.llm.chat_completion.side_effect = [plan, answer('Paris'), proof(False)]
    result = await engine.query('Where was Alice born?', EvidenceSearchParams(
        recovery_mode='after_review', recovery_guard='chain'))
    assert result.response is initial and result.retrieval is old
    assert result.payload['answer_recovery']['applied'] is True
    assert result.payload['answer_guard']['candidate']['answer'] == 'Paris'
    assert result.response.evidence[0].fact == 'Incomplete evidence'
    assert 'New source' in result.payload['answer_guard']['candidate_context']
    assert 'New source' not in str(result.retrieval)
    assert result.payload['answer_guard']['checks']['chain']['reasons'] == ['incomplete_chain', 'missing_links']


async def test_unknown_after_review_skips_guard():
    engine, _ = engine_for(answer('UNKNOWN'), answer('UNKNOWN'))
    result = await engine.query('Where was Alice born?', EvidenceSearchParams(
        review_unknown_answers=True, recovery_guard='chain'))
    assert result.payload['answer_guard']['applied'] is False
    engine.llm.chat_completion.assert_not_awaited()


async def test_transport_failure_propagates():
    engine, _ = engine_for(answer('UNKNOWN'), answer('Paris'))
    engine.llm.chat_completion.side_effect = RuntimeError('backend unavailable')
    with pytest.raises(RuntimeError, match='backend unavailable'):
        await engine.query('Where was Alice born?', EvidenceSearchParams(review_unknown_answers=True, recovery_guard='chain'))
