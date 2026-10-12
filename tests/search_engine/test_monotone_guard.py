"""Production monotone routing, frozen prompts and exact restoration."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES, ChatMessages
from ragu.common.prompts.messages import SystemMessage, UserMessage
from ragu.common.prompts._monotone_guard import Proof, SpanProof, ChallengeProof, verify
from ragu.search_engine.local_search import LocalSearchResult

QUESTION = 'Who founded Acme?'
CONTEXT = 'Ada founded Acme. Bob founded Beta.'
STRICT = dict(constraints=['Acme founder'], steps=[dict(claim='Ada founded Acme',
    quotes=['Ada founded Acme.'], supported=True, constraint_indices=[0])],
    missing_links=[], ambiguous=False, uses_external_knowledge=False, complete=True,
    answer='Ada', rationale='explicit')
REJECT = {**STRICT, 'complete': False, 'missing_links': ['bridge'], 'answer': 'UNKNOWN'}
SPAN = dict(verdict='supported', source_ids=['S00000'], missing_links=[], explanation='explicit')
CHALLENGE = dict(verdict='supported', missing_links=[], conflict_quotes=[], rationale='explicit')


def answer(text, fact):
    return DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model.model_validate(
        {'answer': text, 'evidence': [{'title': 'Source', 'fact': fact}]})


def make_engine(outputs, initial_text='UNKNOWN'):
    initial = answer(initial_text, 'Initial evidence')
    candidate = answer('Ada', 'Recovered evidence')
    old = EvidenceSearchRetrieve(QUESTION, LocalSearchResult(), rendered_context='Original incomplete context.')
    new = EvidenceSearchRetrieve(QUESTION, LocalSearchResult(), rendered_context=CONTEXT)
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=[initial]),
                          chat_completion=AsyncMock(side_effect=outputs))
    engine = EvidenceSearchEngine(llm, SimpleNamespace(index=SimpleNamespace()), SimpleNamespace(),
        max_context_length=8000, tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
    engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
    engine.batch_search = AsyncMock(return_value=[old])
    async def recover(result, proposal, **kwargs):
        result.response = candidate
        result.retrieval = new
    engine._recover_answer = AsyncMock(side_effect=recover)
    return engine, initial, candidate, old, new


def test_frozen_helper_byte_parity():
    root = Path(__file__).resolve().parents[2]
    assert (root/'experiments/monotone_guard.py').read_bytes() == (root/'ragu/common/prompts/_monotone_guard.py').read_bytes()


@pytest.mark.parametrize('outputs,accepted', [
    ([STRICT], True),
    ([REJECT, {**SPAN, 'verdict': 'insufficient'}], False),
    ([REJECT, SPAN, CHALLENGE], True),
    ([REJECT, SPAN, {**CHALLENGE, 'verdict': 'insufficient', 'missing_links': ['bridge']}], False),
])
async def test_exact_frozen_audit_and_answer_evidence_context_restoration(outputs, accepted):
    engine, initial, candidate, old, new = make_engine(outputs)
    expected = await verify(QUESTION, CONTEXT, 'Ada', SimpleNamespace(chat_completion=AsyncMock(side_effect=outputs)))
    result = await engine.query(QUESTION, EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone'))
    assert result.response is (candidate if accepted else initial)
    assert result.retrieval is (new if accepted else old)
    assert result.response.model_dump() == (candidate if accepted else initial).model_dump()
    audit = result.payload['answer_guard']
    assert audit['accepted'] is accepted
    assert audit['checks']['monotone'] == expected
    assert audit['candidate_context'] == CONTEXT
    assert engine.llm.chat_completion.await_count == len(outputs)


async def test_initial_substantive_answer_and_evidence_untouched():
    engine, initial, _, old, _ = make_engine([], initial_text='Ada')
    result = await engine.query(QUESTION, EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone'))
    assert result.response is initial and result.retrieval is old
    assert not result.payload['answer_guard']['applied']
    engine.llm.chat_completion.assert_not_awaited()
    engine._recover_answer.assert_not_awaited()


@pytest.mark.parametrize('stage', ['strict', 'span', 'challenge'])
async def test_custom_stage_prompt_is_used_and_all_nested_audits_are_actual(stage):
    engine, *_ = make_engine([REJECT, SPAN, CHALLENGE])
    name = 'evidence_guard_monotone_' + stage
    instruction = engine.get_prompt(name)
    messages = ChatMessages.from_messages([
        SystemMessage('Custom auditor: ' + stage),
        UserMessage('{{ payload }}'),
    ])
    engine.update_prompt(name, replace(instruction, messages=messages))
    result = await engine.query(QUESTION, EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone'))
    decision = result.payload['answer_guard']['checks']['monotone']
    offset = ['strict', 'span', 'challenge'].index(stage)
    actual = engine.llm.chat_completion.call_args_list[offset].args[0]
    assert actual[0]['content'] == 'Custom auditor: ' + stage
    assert decision['messages'][offset] == actual
    assert decision['calls'][offset]['messages'] == actual
    if stage == 'strict':
        assert decision['strict']['messages'] == actual
    else:
        assert decision['fallback']['messages'][offset-1] == actual
        if stage == 'span':
            assert decision['fallback']['span']['messages'] == actual


async def test_incompatible_schema_override_fails_before_inference():
    engine, *_ = make_engine([])
    name = 'evidence_guard_monotone_strict'
    engine.update_prompt(name, replace(engine.get_prompt(name), pydantic_model=SpanProof))
    with pytest.raises(ValueError, match='registered stage schema'):
        await engine.query(QUESTION, EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone'))
    engine.llm.chat_completion.assert_not_awaited()


async def test_streaming_does_not_emit_unverified_candidate():
    engine, *_ = make_engine([])
    with pytest.raises(ValueError, match='stream_query'):
        async for _ in engine.stream_query(QUESTION, EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone')):
            pytest.fail('Unexpected event')


def test_default_guard_unchanged_and_mode_requires_recovery():
    assert EvidenceSearchParams().recovery_guard == 'off'
    with pytest.raises(ValueError, match='requires recovery'):
        EvidenceSearchParams(recovery_guard='monotone')
