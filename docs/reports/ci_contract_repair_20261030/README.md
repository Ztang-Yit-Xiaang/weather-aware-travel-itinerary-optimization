# CI contract repair

The quality run for PR #2 at `8047d42e56d7533ceb5664e5bde8c78a05328817`
failed three checks, with 5,171 passes and 40 skips. The failed run is
[41015308337](https://github.com/Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization/actions/runs/41015308337).

The changes preserve the current hosting architecture and strict validation:

- Startup coherence now requires the exact three ordered, build-versioned assets:
  stylesheet, public deployment settings, then the app module. The checks still
  fetch each asset and compare its build version with health.
- The interaction registry explicitly covers all 108 current source bindings,
  including seven newly added controls and the revised interest-layer hide
  handler. Exact anchors, selectors, owners and unresolved/stale checks remain
  required. These registrations do not claim real browser execution.
- The independent private-copy fingerprint now includes the interest-region JSON
  inputs already bound by the product runtime. The E2E source manifest also pins
  interest configuration and all static assets, including vendor metadata/fonts.
  A regression test changes a synthetic interest input and verifies that both
  fingerprints change; omitting the input must be rejected.

All three affected test files passed locally: **249 passed**. See
[focused.xml](focused.xml). Local tests use disposable state and do not call paid
providers. The updated GitHub quality run is the authority for the full suite;
this focused result is not substituted for it.

GitHub Pages stays on its existing version. Supabase/ORS setup and eventual
Railway acceptance remain separate from these CI fixes. No provider migration,
production deployment or paid resource is part of this repair.
