# Product Dashboard Testing Report

**Artifact under test:** `runs/e3ux-weather-repair-demo-v6`  
**Status:** all E3.UX4 product, legacy, repository, integrity, and browser gates pass

## Dedicated product suite

`python -m pytest tests/product_dashboard -q`

Twenty-six tests pass. Coverage includes valid lineage, missing parent/diff,
unsafe paths, invalid content hashes, parent/child mismatch, certificate
mismatch, missing evaluator values, failed methods, exact-cap refusal,
infeasibility distinction, NaN/positive infinity/negative infinity,
permission provenance, customer/research models, metric direction and owner,
map/text alternative, export registration, non-overwrite, script escaping, host
path suppression, asset tampering, screenshot hashes, and UX5 exclusion.

## Static and artifact checks

- `python -m ruff check src tests scripts`
- `python scripts/validate_product_dashboard.py runs/e3ux-weather-repair-demo-v6`
- `python scripts/validate_dashboard_export.py`

The product validator checks schemas, run registration, source and asset hashes,
safe paths, semantic/accessibility tokens, forbidden browser APIs, truth-state
catalog completeness, ranking eligibility, disabled interaction, and manifest
lineage.

## Browser and accessibility checks

The 1440, 1024, 768, 430, 390, and 360px matrix passes without horizontal
document overflow, clipped controls, console issues, incomplete resources, or
map initialization failure. Day selection, customer/research switching,
evidence navigation, exact-failure visibility, certificate/hashes, touch target
size, focus visibility, and computed text contrast were inspected.

## Legacy regression boundary

The product path is additive. Legacy E3.C focused parity, regression, full
project checks, and the legacy export validator were rerun in the same
worktree:

- E3.C focused: 23 passed in 13.39s.
- Legacy regression matrix: 81 passed in 13.79s.
- Full pytest: 315 passed in 46.95s.
- Project checks: Ruff, context snapshot tests, and full pytest all passed.
- Ruff: all `src`, `tests`, and `scripts` checks passed.
- Legacy and product dashboard validators passed.
- Markdown local-link validation passed.

## Deliberately not tested

E3.UX5 interaction actions do not exist in the artifact, so permission grants,
acceptance, persistence, and immutable continuation-run flows are outside this
test report. They remain E5-dependent and deferred.
