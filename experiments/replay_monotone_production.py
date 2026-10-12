"""Offline public-pipeline replay; no inference provider or judge labels are loaded."""
import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


async def main(args):
    production = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    head = subprocess.check_output(['git', '-C', str(production), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == subprocess.check_output(['git', '-C', str(production), 'rev-parse', args.expected_commit], text=True).strip(), 'Wrong production commit')
    require(not subprocess.check_output(['git', '-C', str(production), 'status', '--porcelain'], text=True).strip(), 'Production worktree must be clean')
    sys.path.insert(0, str(production))
    from ragu import EvidenceSearchEngine, EvidenceSearchParams, EvidenceSearchRetrieve
    from ragu.common.prompts import DEFAULT_PROMPT_TEMPLATES
    from ragu.search_engine.local_search import LocalSearchResult
    import ragu.search_engine.evidence_search as implementation
    require(Path(implementation.__file__).resolve().is_relative_to(production), 'Wrong installed package')
    plugin = production / 'experiments/monotone_guard.py'
    require(plugin.read_bytes() == (production / 'ragu/common/prompts/_monotone_guard.py').read_bytes(), 'Frozen helper drift')
    questions_path = root / 'grounding_guard/questions.json'
    questions = read(questions_path)
    hashes = {str(questions_path): digest(questions_path), str(Path(__file__).resolve()): digest(__file__)}
    totals = {'new_llm_calls': 0, 'guard_calls_replayed': 0, 'trajectories': 0}
    summaries = {}
    schema = DEFAULT_PROMPT_TEMPLATES['evidence_answer'].pydantic_model
    for split in args.splits:
        run = root / 'grounding_monotone/runs' / ('monotone_' + split)
        manifest_path, receipt_path = run/'manifest.json', run/'complete.json'
        manifest, receipt = read(manifest_path), read(receipt_path)
        require(receipt.get('complete') is True and manifest['split'] == split, 'Incomplete or mismatched run')
        require(receipt['indices'] == manifest['indices'] == [q['index'] for q in questions[split]], 'Membership mismatch')
        require(digest(plugin) == digest(manifest['plugin']), 'Production differs from frozen experiment')
        required = [manifest_path] + [run/'decisions'/f"{i:03}.json" for i in manifest['indices']]
        require(all(str(p.resolve()) in receipt['sha256'] for p in required), 'Incomplete output receipt')
        bound = {**manifest['sha256'], **receipt['sha256']}
        for source in manifest['sources'].values():
            bound.update({str(Path(source['directory'])/name): value for name, value in source['files'].items()})
        for path, expected in bound.items():
            require(digest(path) == expected, 'Changed frozen file: ' + path)
        hashes.update(bound)
        hashes[str(receipt_path)] = digest(receipt_path)
        baseline = Path(manifest['sources']['baseline']['directory'])
        candidate = Path(manifest['sources']['candidate']['directory'])
        counts = {'questions': len(questions[split]), 'recovered': 0, 'accepted_recoveries': 0, 'calls': 0}
        for question in questions[split]:
            require(set(question) == {'index', 'id', 'question'}, 'Unexpected question fields')
            i = question['index']
            row = read(run/'decisions'/f'{i:03}.json')
            require(row['index'] == i and row['id'] == question['id'], 'Decision identity mismatch')
            values = []
            for folder in (baseline, candidate):
                answer = read(folder/'answers'/f'{i:03}.json')
                context = read(folder/'contexts'/f'{i:03}.json')
                require(answer['id'] == context['id'] == question['id'], 'Upstream identity mismatch')
                values.append((schema.model_validate({k: answer[k] for k in ('answer', 'evidence')}), context['context']))
            (old_answer, old_context), (new_answer, new_context) = values
            initial_unknown = old_answer.answer.strip().upper() == 'UNKNOWN'
            recovered = initial_unknown and new_answer.answer.strip().upper() != 'UNKNOWN'
            require(row['recovered'] is recovered, 'Recovery classification mismatch')
            if not initial_unknown:
                require(old_answer.model_dump() == new_answer.model_dump(), 'Upstream changed substantive answer')
            pending = list(row['transcript'])
            class ReplayLLM:
                initial_calls = 0

                async def batch_chat_completion(self, conversations, output_schema=str, **kwargs):
                    self.initial_calls += 1
                    require(self.initial_calls == 1 and len(conversations) == 1, 'Unexpected upstream inference')
                    require(output_schema.model_json_schema() == schema.model_json_schema(), 'Initial schema mismatch')
                    return [old_answer.model_copy(deep=True)]

                async def chat_completion(self, messages, output_schema=str, **kwargs):
                    require(bool(pending), 'Unexpected inference; no live provider exists')
                    saved = pending.pop(0)
                    require(messages == saved['messages'], f'Prompt mismatch: {split}/{i}')
                    require(output_schema.model_json_schema() == saved['schema'], f'Schema mismatch: {split}/{i}')
                    require(saved['response'] is not None, 'Missing cached response')
                    counts['calls'] += 1
                    return output_schema.model_validate(saved['response'])

            class ReplayEngine(EvidenceSearchEngine):
                async def batch_search(self, queries, params=None):
                    require(queries == [question['question']], 'Query mismatch')
                    return [EvidenceSearchRetrieve(queries[0], LocalSearchResult(), rendered_context=old_context)]

                async def _recover_answer(self, result, proposal, *, top_k=5):
                    result.response = new_answer.model_copy(deep=True)
                    result.retrieval = EvidenceSearchRetrieve(result.query, LocalSearchResult(), rendered_context=new_context)
                    result.payload['answer_recovery'] = {'applied': True, 'replayed_final_upstream_state': True}

            engine = ReplayEngine(ReplayLLM(), SimpleNamespace(index=SimpleNamespace()), SimpleNamespace(),
                max_context_length=8000, tokenizer_backend='tiktoken', tokenizer_model='gpt-4o')
            engine.update_prompt('local_search', DEFAULT_PROMPT_TEMPLATES['evidence_answer'])
            actual = await engine.query(question['question'], EvidenceSearchParams(recovery_mode='direct', recovery_guard='monotone'))
            require(not pending, 'Unused cached inference')
            accepted = recovered and row['result']['accepted']
            expected_answer = new_answer if accepted or initial_unknown and not recovered else old_answer
            expected_context = new_context if accepted or initial_unknown and not recovered else old_context
            require(actual.response.model_dump() == expected_answer.model_dump(), f'Answer/evidence mismatch: {split}/{i}')
            require(str(actual.retrieval) == expected_context, f'Returned context mismatch: {split}/{i}')
            audit = actual.payload['answer_guard']
            require(audit['applied'] is recovered, 'Applied flag mismatch')
            if recovered:
                require(audit['accepted'] is accepted and audit['checks']['monotone'] == row['result'], f'Full nested audit mismatch: {split}/{i}')
                require(audit['initial'] == old_answer.model_dump() and audit['candidate'] == new_answer.model_dump(), 'Audit response mismatch')
                require(audit['candidate_context'] == new_context, 'Audit context mismatch')
            counts['recovered'] += recovered
            counts['accepted_recoveries'] += bool(accepted)
        summaries[split] = counts
        totals['trajectories'] += counts['questions']
        totals['guard_calls_replayed'] += counts['calls']
        print('REPLAY', split, counts, flush=True)
    result = {'status': 'passed', 'production_commit': head, **totals, 'splits': summaries, 'sha256': hashes,
        'limitations': 'Public batch_query and all guard calls run; initial answer/context and final recovery state are injected from frozen artifacts. Retrieval, planning and intermediate reviews are not rerun. No live inference provider or judge labels are loaded.'}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASSED', totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--splits', nargs='+', choices=['discovery', 'confirmation'], default=['discovery', 'confirmation'])
    parser.add_argument('--output', required=True)
    asyncio.run(main(parser.parse_args()))
