"""Replay both discovery draws through the public engine with real chunk search."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

WORKTREE = Path(__file__).resolve().parents[1]
ROOT = WORKTREE.parents[1] / 'benchmark'
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE.parent / 'rouge_adaptive/experiments'))
from adaptive_recovery import RecoveryPlan
from review_answer import ReviewedAnswer
from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES
from ragu.search_engine.local_search import LocalSearchResult
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage

FROZEN_INITIAL_PLUGIN = WORKTREE.parent / 'evidence_ledger/experiments/overnight_answer.py'
spec = importlib.util.spec_from_file_location('frozen_initial_schema', FROZEN_INITIAL_PLUGIN)
frozen_initial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen_initial)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Replay:
    def __init__(self, initial, reviewed, recovered, trace, original_context, second, second_trace, initial_messages):
        self.initial, self.reviewed, self.recovered = initial, reviewed, recovered
        self.trace = trace
        self.original_context = original_context
        self.second = second
        self.second_trace = second_trace
        self.initial_messages = initial_messages
        self.batch_calls = 0
        self.single_calls = 0

    async def batch_chat_completion(self, conversations, output_schema, **kwargs):
        self.batch_calls += 1
        assert len(conversations) == 1
        if self.batch_calls == 1:
            assert conversations[0] == self.initial_messages, 'Initial answer message mismatch'
            assert output_schema.model_json_schema() == frozen_initial.EvidenceAnswer.model_json_schema()
            payload = json.loads(conversations[0][0]['content'].split('\n', 1)[1])
            assert payload['context'] == self.original_context
            row = self.initial
        else:
            assert self.batch_calls == 2
            assert conversations[0] == self.reviewed['messages']
            assert output_schema.model_json_schema() == ReviewedAnswer.model_json_schema()
            row = self.reviewed
        return [output_schema(**{k: row[k] for k in ('answer', 'evidence')})]

    async def chat_completion(self, messages, output_schema, **kwargs):
        self.single_calls += 1
        if self.single_calls >= 3:
            if self.single_calls == 3:
                assert messages == self.second['planner_messages'], 'Second planner mismatch'
                assert output_schema.model_json_schema() == RecoveryPlan.model_json_schema()
                return output_schema(queries=self.second_trace['queries'])
            assert self.single_calls == 4
            assert messages == self.second['messages'], 'Second review mismatch'
            assert output_schema.model_json_schema() == ReviewedAnswer.model_json_schema()
            return output_schema(**{k: self.second[k] for k in ('answer', 'evidence')})
        if self.single_calls == 1:
            assert messages == self.recovered['planner_messages'], 'Planner message mismatch'
            assert output_schema.model_json_schema() == RecoveryPlan.model_json_schema()
            return output_schema(queries=self.trace['queries'])
        assert self.single_calls == 2
        assert messages == self.recovered['messages'], 'Recovery review message mismatch'
        assert output_schema.model_json_schema() == ReviewedAnswer.model_json_schema()
        return output_schema(**{k: self.recovered[k] for k in ('answer', 'evidence')})


async def main():
    source = ROOT / 'overnight/runs/core_ledger/discovery'
    day = ROOT / 'rouge_day'
    indices = read(ROOT / 'overnight/split.json')['discovery']
    chunks = read(ROOT / 'runs/musique_full_dev/graph/kv_chunks.json')
    async def get_chunks(ids):
        return [chunks.get(cid) for cid in ids]
    vdb = NanoVectorDBStorage(384, score_threshold=.2,
        storage_folder=str(ROOT / 'runs/musique_full_dev/graph'), filename='vdb_chunk.json')
    graph = SimpleNamespace(index=SimpleNamespace(chunks_vector_db=vdb,
        chunks_kv_storage=SimpleNamespace(get_by_ids=get_chunks)))
    checked = []
    for suffix in ('', '_fresh'):
        initial_dir = source if not suffix else day / 'runs/baseline_repeat/discovery'
        embeddings = read(day / 'inference' / ('adaptive' + suffix) / 'query_embeddings.json')
        embeddings.update(read(day / 'inference' / ('deeper' + suffix) / 'query_embeddings.json'))
        async def embed_queries(queries):
            return [embeddings[q] for q in queries]
        embedder = SimpleNamespace(batch_embed_text=embed_queries)
        for mode in ('after_review',):
            single_calls = batch_calls = applied = 0
            for i in indices:
                name = f'{i:03}.json'
                original_context = read(source / 'contexts' / name)
                initial = read(initial_dir / 'answers' / name)
                reviewed = read(day / 'runs' / ('selective' + suffix) / 'discovery/answers' / name)
                recovered = read(day / 'runs' / ('adaptive' + suffix) / 'discovery/answers' / name)
                recovered_context = read(day / 'runs' / ('adaptive' + suffix) / 'discovery/contexts' / name)
                target_dir = day / 'runs' / ('deeper' + suffix) / 'discovery'
                expected = read(target_dir / 'answers' / name)
                expected_context = read(target_dir / 'contexts' / name)
                question = json.loads(original_context['messages'][0]['content'].split('\n', 1)[1])['question']
                initial_messages = read(initial_dir / 'contexts' / name)['messages']
                replay = Replay(initial, reviewed, recovered, recovered_context['trace'], original_context['context'], expected, expected_context['trace'], initial_messages)
                engine = EvidenceSearchEngine(replay, graph, embedder, max_context_length=8000,
                    tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
                engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
                original = EvidenceSearchRetrieve(question, LocalSearchResult(), rendered_context=original_context['context'])
                engine.batch_search = AsyncMock(return_value=[original])
                result = await engine.query(question, EvidenceSearchParams(recovery_mode=mode, second_recovery_policy='deeper'))
                assert result.response.answer == expected['answer'], (i, mode, suffix)
                assert str(result.retrieval) == expected_context['context'], (i, mode, suffix, 'context')
                assert str(original) == original_context['context']
                if replay.single_calls:
                    rounds = result.payload['answer_recovery']['rounds']
                    assert rounds[0]['hit_ids'] == recovered_context['trace']['hit_ids']
                    assert rounds[0]['added_ids'] == recovered_context['trace']['added_ids']
                    if replay.single_calls == 4:
                        assert rounds[1]['hit_ids'] == expected_context['trace']['hit_ids']
                        assert rounds[1]['added_ids'] == expected_context['trace']['added_ids']
                    applied += 1
                single_calls += replay.single_calls
                batch_calls += replay.batch_calls
            checked.append({'draw': suffix or 'historical', 'mode': mode, 'n': len(indices),
                            'search_recoveries': applied, 'replayed_single_calls': single_calls,
                            'replayed_batch_calls': batch_calls})
            print(checked[-1], flush=True)
    files = [WORKTREE / p for p in ('ragu/search_engine/evidence_search.py',
        'ragu/common/prompts/default_models.py', 'ragu/common/prompts/default_templates.py',
        'ragu/common/prompts/prompt_storage.py')]
    report = {'checks': checked, 'new_inference_calls': 0, 'mismatches': 0,
              'revision': subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip(),
              'parity_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'frozen_initial_plugin_sha256': hashlib.sha256(FROZEN_INITIAL_PLUGIN.read_bytes()).hexdigest(),
              'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'method': 'Public query replay; full schema and prompt equality; real NanoVDB/GraphRetriever over frozen query vectors; exact final context and added IDs; no gold'}
    (day / 'production_deeper_parity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


asyncio.run(main())
