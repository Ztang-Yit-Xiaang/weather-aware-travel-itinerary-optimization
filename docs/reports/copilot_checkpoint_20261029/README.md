# Copilot checkpoint — 29 October 2026

The current checkpoint includes individualized whole-trip planning, typed local repairs, destination details, state/map synchronization safeguards, and the subsequent read-only/request-completeness repairs. This publication preserves source and tests before any proposed nonlinear optimizer work.

## Verified scope

- The final related six-module regression completed **339 passed, zero failures/errors/skips**, including 39 new request/protection checks. [JUnit receipt](focused_tests.xml), [source snapshot](source_snapshot.json).
- Nine website messages across two retained source epochs included **two actual paid OpenAI calls**. Both returned the exact long Bixby-to-Pfeiffer replacement. Seven messages were handled locally. Both replacement previews passed independent evaluation and matched the map; they were not accepted. [Verification](verification.json).
- [32 original website screenshots](review.html) are hash-checked, with their source epochs distinguished. They show synthetic test itineraries rather than saved personal sessions.
- Whole-trip planning uses the SciPy/HiGHS MILP. The current product's local replacement path uses a deterministic context-aware heuristic and independent evaluation. Research Progressive Repair remains a separate path.

## Risk findings remain open

The risk inspection is **read-only**, not a fix: missing numeric weather risk currently defaults to zero in local scoring; the whole-trip objective has no numeric weather-risk term; and the UI risk delta requires complete values and matching stop-ID sets. A retained eight-stop PASSED preview had numeric risk for seven stops. A passed configured evaluation is not a complete quantitative safety assessment. [Risk inspection and synthetic helper probe](risk_inspection.json).

## Remaining acceptance boundaries

These focused passes do not establish full repository regression, general language reliability, independent human UAT or release readiness. Historical failures and unexecuted scenarios remain in the local audit. A daily driving cap does not guarantee less total driving; fixed-day insertions still require exact directed road evidence. Short cap-only follow-ups and slash daily-cap syntax remain limited.

The current source uses a finite California candidate catalog. Official access checks apply to their retrieval time, and shared route/transport assumptions do not establish bookings or future availability. No nonlinear/nonconvex solver extension has been implemented by the risk inspection or this GitHub sync.

## Reproduction and publication boundaries

The synthetic default demo package is included under `runs/california-coast-coherent-demo-v1`. Launch the app using the repository's normal launcher; external map/routing assets and API credentials remain environment-specific. The recorded-runtime gzip fixtures are deliberate, deduplicated unit-test inputs, not disposable caches or fresh acceptance evidence.

Example focused offline regression command (no real OpenAI calls):

```text
python -B -m pytest --confcutdir=tests/product_app tests/product_app/test_request_completeness.py tests/product_app/test_copilot_memory.py tests/product_app/test_copilot_semantic_coherence.py tests/product_app/test_readonly_review.py tests/product_app/test_copilot_dispatch_safety.py tests/product_app/test_trip_planning.py -q
```

Raw paid-provider payloads, saved user sessions/conversations, private backups, machine identities, caches and disposable test trees are excluded. Original detailed local evidence remains preserved on the working machine.
