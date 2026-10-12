# Experimental recovery acceptance with a fallback

This opt-in integration is provisional and awaits its independent evaluation gate. It preserves every answer accepted by the existing strict guard. When strict verification rejects a recovered answer, a source-span check and a separate full-context challenge must both accept it. The challenge receives the question, unchanged context and answer, without previous proof or verdict.

```python
from ragu import EvidenceSearchParams

params = EvidenceSearchParams(
    recovery_mode="after_review",
    second_recovery_policy="deeper",
    recovery_guard="monotone",
)
# await engine.query(question, params)
```

The guard defaults to `off`. It only checks substantive answers recovered from an initial `UNKNOWN`. An initial substantive answer is preserved. Rejection restores the original answer, evidence and retrieval context. Acceptance preserves the recovered answer and evidence exactly. Streaming remains unsupported when recovery is enabled.

The guard makes one to three additional calls. `response.payload["answer_guard"]["checks"]["monotone"]` contains the strict decision, optional fallback decision, and all executed calls with actual messages and structured outputs. Source IDs and quotes establish provenance; semantic support remains a fallible model judgment. Preserving strict accepts does not guarantee unchanged precision among all returned answers.

Customize the three registered prompts with `engine.update_prompt`: `evidence_guard_monotone_strict`, `evidence_guard_monotone_span`, and `evidence_guard_monotone_challenge`. Preserve each registered output schema (`Proof`, `SpanProof`, `ChallengeProof` respectively). An incompatible schema raises `ValueError`. Text customization is recorded in every nested call audit, but changes the evaluated method; previously measured results then no longer apply directly. The same configured engine LLM serves all stages.

Offline replay: `experiments/replay_monotone_production.py --root PATH_TO_BENCHMARK --expected-commit COMMIT --output RECEIPT.json`. By default it requires completed discovery and confirmation artifacts, checks their hashes, and replays all ordered calls with no live provider. It exercises the production answer and guard path while injecting frozen initial and final-recovery states; it does not repeat retrieval or intermediate recovery inference.
