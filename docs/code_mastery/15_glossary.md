# Glossary

| Term | Repository-specific meaning |
|---|---|
| Accepted itinerary | Immutable parent `PlanArtifactV2` owned by the user/run |
| Child plan | New `PlanArtifactV2` with parent lineage after repair |
| Content hash | Stable identifier derived from artifact content |
| `OwnedConstraint` | Typed commitment with owner/strength/status/provenance |
| `OwnershipPolicy` | Rules assigning change costs and allowed relaxation |
| `PlanRepository` | Append-only plan JSON store and index |
| `PlanDiff` | Typed parent/child changes and preservation cost |
| `RouteMatrix` | Content-addressed travel cells with provenance/validation |
| Road-valid | Supported by approved road evidence; never geodesic fallback |
| Repair radius | Scope of editable days/lodging/routes; smallest tried first |
| Lexicographic solve | Solve priority stages sequentially; later stages cannot degrade earlier ones beyond tolerance |
| Planner run | Record of requested/executed method, status, trace, and diagnostics |
| Independent evaluation | Recompute displayed-plan eligibility without trusting solver feasibility |
| Certificate | `PlanEvaluationCertificate` tied to plan content hash |
| Comparison eligible | Independent certificate permits comparison |
| Ranking eligible | Benchmark provenance, certificate, route, and metric gates all pass |
| Evidence complete | Required rows/provenance/route inputs/failures are present |
| Performance complete | Required methods actually completed or proved infeasibility |
| Exact cap refusal | Incomplete search; no child or optimality certificate |
| Complete infeasibility | Entire declared universe was searched/proved infeasible |
| Explanation evidence | Claims plus IDs of supporting diff/certificate/route/counterfactual records |
| Hypothetical probe | `test_only` preview that cannot be accepted/executed |
| Authorized continuation | New immutable run created after explicit session permission |
| Legacy parity | Frozen behavior/output contract preserved during ownership refactor |
| Customer mode | Plain-language progressive disclosure of canonical truth |
| Research mode | Same truth plus method, lineage, hashes, certificate, and diagnostics |
| `planned` | Accepted future work, not started |
| `ready` | Dependencies and material decisions resolved |
| `in-progress` | Implementation/evidence work started |
| `blocked` | Named dependency prevents progress |
| `implemented` | Code complete, required verification pending |
| `verified` | Exit tests/checks/evidence passed |
| `deferred` | Intentionally postponed with reason |

