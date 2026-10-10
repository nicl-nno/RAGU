"""Regression cases for retrieving evidence across a graph edge, without an LLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine
from ragu.storage.types import EmbeddingHit


def make_case(reverse=False):
    texts = [
        "Ada founded Acme.",
        "Acme is headquartered in Porto.",
        "Acme makes furniture.",
        "OtherCo is headquartered in Lima.",
    ]
    chunks = [Chunk(content=text, chunk_order_idx=i, doc_id=f"doc-{i}") for i, text in enumerate(texts)]
    ada = Entity("Ada", "PERSON", "Founder", [chunks[0].id], documents_id=["doc-0"])
    acme = Entity("Acme", "ORGANIZATION", "Company", [c.id for c in chunks[:3]])
    other = Entity("OtherCo", "ORGANIZATION", "Unrelated company", [chunks[3].id])
    subject, obj = (acme, ada) if reverse else (ada, acme)
    edge = Relation(subject.id, obj.id, subject.entity_name, obj.entity_name, "FOUNDED", "Ada founded Acme")
    nodes = {e.id: e for e in (ada, acme, other)}
    payloads = {
        c.id: {"content": c.content, "chunk_order_idx": c.chunk_order_idx, "doc_id": c.doc_id}
        for c in chunks
    }
    get_chunks = AsyncMock(side_effect=lambda ids: [payloads.get(i) for i in ids])
    kg = SimpleNamespace(index=SimpleNamespace(
        graph_backend=SimpleNamespace(get_all_edges_for_nodes=AsyncMock(
            side_effect=lambda ids: [[edge] if i in (ada.id, acme.id) else [] for i in ids]
        )),
        get_nodes=AsyncMock(side_effect=lambda ids: [nodes.get(i) for i in ids]),
        chunks_kv_storage=SimpleNamespace(get_by_ids=get_chunks),
    ))
    engine = LocalSearchEngine(llm=SimpleNamespace(), knowledge_graph=kg, embedder=SimpleNamespace())
    engine.retriever.query_entities = AsyncMock(return_value=[([ada], [EmbeddingHit(id=ada.id, distance=1.0)])])
    return engine, chunks, ada, acme, other, edge, get_chunks


@pytest.mark.parametrize("reverse", [False, True])
async def test_expansion_recovers_second_supporting_document(reverse):
    engine, chunks, *_, get_chunks = make_case(reverse)
    baseline = await engine.search("Where is the company founded by Ada headquartered?")
    expanded = await engine.search("Where is the company founded by Ada headquartered?", LocalParams(
        neighbor_chunk_limit=1, neighbor_chunk_candidates=1,
    ))
    assert [c.id for c in baseline.result.chunks] == [chunks[0].id]
    assert [c.id for c in expanded.result.chunks] == [chunks[0].id, chunks[1].id]
    assert expanded.metrics["neighbor_chunks"] == [chunks[1].id]
    assert expanded.result.documents_id == ["doc-0", "doc-1"]
    assert get_chunks.await_args.args[0] == [chunks[0].id, chunks[1].id]
    assert "Porto" in expanded.to_text()
    assert "Porto" not in baseline.to_text()


async def test_reranking_selects_neighbor_before_output_limit_and_retains_seed():
    engine, chunks, ada, acme, other, edge, get_chunks = make_case()
    # The answer chunk is second among neighbor candidates before reranking.
    acme.source_chunk_id = [chunks[0].id, chunks[2].id, chunks[1].id]
    engine.reranker = SimpleNamespace(score=AsyncMock(side_effect=lambda query, texts: sorted(
        enumerate([float("Porto" in text) for text in texts]), key=lambda item: -item[1],
    )))
    result = await engine.search("Where is Acme headquartered?", LocalParams(
        neighbor_chunk_limit=1, neighbor_chunk_candidates=2,
    ))
    assert [c.id for c in result.result.chunks] == [chunks[1].id, chunks[0].id]
    assert result.metrics["neighbor_chunks"] == [chunks[1].id]
    assert len(get_chunks.await_args.args[0]) == 3


async def test_batched_queries_do_not_share_neighbor_evidence():
    engine, chunks, ada, acme, other, edge, get_chunks = make_case()
    engine.retriever.query_entities.return_value = [
        ([ada], [EmbeddingHit(id=ada.id, distance=1.0)]),
        ([other], [EmbeddingHit(id=other.id, distance=1.0)]),
    ]
    results = await engine.batch_search(["Ada?", "OtherCo?"], LocalParams(neighbor_chunk_limit=2))
    assert {c.id for c in results[0].result.chunks} == {c.id for c in chunks[:3]}
    assert [c.id for c in results[1].result.chunks] == [chunks[3].id]
    assert results[1].metrics["neighbor_chunks"] == []


async def test_missing_neighbors_chunks_and_duplicate_edges_are_safe():
    engine, chunks, ada, acme, other, edge, get_chunks = make_case()
    acme.source_chunk_id = [chunks[0].id, "missing", chunks[1].id, chunks[1].id]
    dangling = Relation(ada.id, "missing-node", "Ada", "Missing", "KNOWS", "Unknown")
    loop = Relation(ada.id, ada.id, "Ada", "Ada", "SELF", "Self edge")
    engine.knowledge_graph.index.graph_backend.get_all_edges_for_nodes.side_effect = None
    engine.knowledge_graph.index.graph_backend.get_all_edges_for_nodes.return_value = [[edge, edge, dangling, loop]]
    result = await engine.search("Ada?", LocalParams(neighbor_chunk_limit=3))
    assert [c.id for c in result.result.chunks] == [chunks[0].id, chunks[1].id]
    assert get_chunks.await_args.args[0] == [chunks[0].id, "missing", chunks[1].id]


async def test_candidate_budget_is_shared_round_robin_across_neighbors():
    engine, chunks, ada, acme, other, edge, get_chunks = make_case()
    second_edge = Relation(ada.id, other.id, "Ada", "OtherCo", "KNOWS", "Knows", relation_strength=0.5)
    engine.knowledge_graph.index.graph_backend.get_all_edges_for_nodes.side_effect = None
    engine.knowledge_graph.index.graph_backend.get_all_edges_for_nodes.return_value = [[second_edge, edge]]
    result = await engine.search("Ada?", LocalParams(neighbor_chunk_limit=2, neighbor_chunk_candidates=2))
    assert [c.id for c in result.result.chunks] == [chunks[0].id, chunks[1].id, chunks[3].id]
    assert len(get_chunks.await_args.args[0]) == 3


async def test_zero_limit_preserves_seed_only():
    engine, chunks, *_, get_chunks = make_case()
    result = await engine.search("Ada?", LocalParams(neighbor_chunk_limit=0, neighbor_chunk_candidates=0))
    assert [c.id for c in result.result.chunks] == [chunks[0].id]
    assert "neighbor_chunks" not in result.metrics
    get_chunks.assert_awaited_once_with([chunks[0].id])


@pytest.mark.parametrize("use_chunks", [False, True])
async def test_expanded_evidence_reaches_generation_only_when_chunks_enabled(use_chunks):
    engine, chunks, *_, get_chunks = make_case()
    engine.llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["test answer"]))
    response = await engine.query("Where is Ada's company headquartered?", LocalParams(
        neighbor_chunk_limit=1, neighbor_chunk_candidates=1, use_chunks=use_chunks,
    ))
    messages = engine.llm.batch_chat_completion.await_args.args[0][0]
    prompt = "\n".join(message["content"] for message in messages)
    assert ("Acme is headquartered in Porto." in prompt) == use_chunks
    assert response.response == "test answer"


@pytest.mark.parametrize("kwargs", [
    {"neighbor_chunk_limit": -1}, {"neighbor_chunk_candidates": -1},
    {"neighbor_chunk_limit": 2, "neighbor_chunk_candidates": 1},
    {"neighbor_chunk_limit": True}, {"neighbor_chunk_candidates": 1.5},
])
def test_invalid_expansion_limits(kwargs):
    with pytest.raises(ValueError):
        LocalParams(**kwargs)
