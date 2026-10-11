"""Offline checks of provenance/coverage rejection and blind prompt isolation."""
import copy
import json

import pytest

from experiments.grounding_guard import normalize_answer, verify


CONTEXT = "Ada was born in Elm. Elm is in Norland."
GOOD = {
    "constraints": ["Ada's birthplace", "Country containing that birthplace"],
    "steps": [
        {"claim": "Ada's birthplace is Elm", "quotes": ["Ada was born in Elm."], "supported": True, "constraint_indices": [0]},
        {"claim": "Elm is in Norland", "quotes": ["Elm is in Norland."], "supported": True, "constraint_indices": [1]},
    ],
    "missing_links": [], "ambiguous": False, "uses_external_knowledge": False,
    "complete": True, "answer": "Norland", "rationale": "Both links explicit.",
}


class Stub:
    def __init__(self, proof):
        self.proof = proof
        self.calls = []

    async def chat_completion(self, messages, output_schema):
        self.calls.append(messages)
        return self.proof


async def test_complete_chain_accepted_and_serializable():
    out = await verify("In what country was Ada born?", CONTEXT, "Norland", Stub(GOOD))
    assert out["accepted"]
    json.dumps(out)


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["steps"].pop(0), "uncovered_constraints"),
    (lambda p: p["steps"][0].update(quotes=["Ada was born in Oak."]), "invalid_exact_quote:0"),
    (lambda p: p["steps"][0].update(quotes=[""]), "invalid_exact_quote:0"),
    (lambda p: p["steps"][0].update(quotes=[]), "invalid_exact_quote:0"),
    (lambda p: p["steps"][0].update(supported=False), "unsupported_step:0"),
    (lambda p: p.update(ambiguous=True), "ambiguous_identity_or_evidence"),
    (lambda p: p.update(missing_links=["Ada may refer to another person"]), "missing_links"),
    (lambda p: p.update(uses_external_knowledge=True), "external_knowledge"),
    (lambda p: p.update(complete=False), "incomplete_chain"),
    (lambda p: p.update(steps=[]), "missing_steps"),
    (lambda p: p["steps"][0].update(constraint_indices=[-1]), "invalid_constraint_reference:0"),
    (lambda p: p.update(answer="UNKNOWN"), "unknown_or_empty_answer"),
    (lambda p: p.update(answer="South Norland"), "answer_disagreement"),
])
async def test_fail_closed(mutation, reason):
    proof = copy.deepcopy(GOOD)
    mutation(proof)
    out = await verify("In what country was Ada born?", CONTEXT, "Norland", Stub(proof))
    assert not out["accepted"]
    assert reason in out["reasons"]


@pytest.mark.parametrize("raw", [None, {}, {**GOOD, "complete": "false"}, [GOOD]])
async def test_malformed_response_rejected(raw):
    out = await verify("Country?", CONTEXT, "Norland", Stub(raw))
    assert out["reasons"] == ["malformed_output"]


async def test_blind_mode_never_sends_candidate_and_requires_agreement():
    llm = Stub(GOOD)
    out = await verify("Country?", CONTEXT, "SECRET_PROPOSAL", llm, "independent")
    assert "SECRET_PROPOSAL" not in json.dumps(llm.calls)
    assert set(json.loads(llm.calls[0][1]["content"])) == {"question", "context"}
    assert out["reasons"] == ["answer_disagreement"]


async def test_unknown_does_not_call_model():
    llm = Stub(GOOD)
    out = await verify("Country?", CONTEXT, "UNKNOWN", llm)
    assert not out["accepted"] and not llm.calls


def test_normalization_is_exact_not_fuzzy():
    assert normalize_answer("  The Norland! ") == "norland"
    assert normalize_answer("South Norland") != normalize_answer("Norland")
    assert normalize_answer("1891") != normalize_answer("1892")
    assert normalize_answer("USA") != normalize_answer("United States")


async def test_transport_errors_are_not_misreported_as_rejection():
    class Broken:
        async def chat_completion(self, *args, **kwargs):
            raise TimeoutError("retry me")
    with pytest.raises(TimeoutError):
        await verify("Country?", CONTEXT, "Norland", Broken())
