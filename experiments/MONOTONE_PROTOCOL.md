# Preserve strict accepts with a challenged-span fallback

This composition was proposed AFTER reading H4 development results. It is a
bounded search extension, not a predeclared H4 result. No fresh 400-question judge
outcomes were inspected in designing it. The previously delivered strict guard
remains the baseline; this experiment does not change its prompt or acceptance.

Hypothesis: preserve every answer accepted by the existing strict guard and use
the independently challenged span method only to recover additional answers that
the strict guard rejected. This implements strict OR (span AND challenge), with
short-circuit evaluation. It changes routing only, not any component's prompts,
schemas, source-ID validation, quote validation or answer text.

First execute the exact frozen original strict chain verifier. An acceptance ends
the pipeline immediately. On rejection execute the exact frozen H4 verifier: the
original span checker first, then an independent challenge ONLY on span acceptance.
The challenge still sees only question, FULL original context and proposal; no
strict or span decisions, proofs, rationales or citations enter its messages.
Malformed structured responses remain fail-closed within their original verifier;
transport errors propagate. No new retrieval or answer generation is introduced.

All development calls already exist in source-bound strict and H4 artifacts.
A runner may replay them only after verifying original inputs, exact ordered
messages, schemas and artifact hashes. A missing replay must fail rather than
silently inventing a measurement. The full deployed method may make one, two or
three calls; report both full method cost and newly executed experimental calls.

API: async verify(question, context, proposal, llm), where the provider offers
async chat_completion(messages, output_schema). The standalone module contains
both frozen implementations. Result records strict, fallback, accepted, reasons,
proof, resolved_sources and all executed calls/messages in order. Stages are
strict, span and challenge. The logical union preserves strict accepts, but does
not itself guarantee precision: independently evaluate the added answers.

Freeze discovery selection before reading the fresh judge audit. This composition
must satisfy the same precision, source-support, grounded-coverage and answer-
retention gates; the earlier span confirmation failure remains disclosed. Do not
integrate into production or change defaults before a successful new gate.
