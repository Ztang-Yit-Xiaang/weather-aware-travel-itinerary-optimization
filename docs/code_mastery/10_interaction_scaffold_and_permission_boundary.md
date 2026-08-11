# Interaction Scaffold and Permission Boundary

The scaffold exists, but E5 is deferred and default mode is disabled.

## Main Components

### `PermissionAwareClarificationController`

- **Why/category:** UI/workflow controller for ambiguous semantic candidates.
- **Caller:** interaction CLI/pipeline.
- **Inputs/outputs:** candidates, compiler, permission policy, probe executor,
  consequence/clarification policies in; pause/commit/ask result out.
- **State:** one interaction session; question budget and decisions are
  session-scoped.
- **Invariant:** inferred meaning cannot raise permission.
- **Tests/gate/state:** consequence/clarification and pipeline integration
  tests; E5; implemented scaffold, deferred phase.

### `AllowListedPatchCompiler`

- **Why/category:** stateless security/contract service.
- **Caller:** controller.
- **Inputs/outputs:** confirmed semantic candidate + parent in; typed
  allow-listed `ModelPatch` out.
- **Invariant:** known targets/parameters, JSON-safe finite values, required
  structural evidence.
- **Failure/tests:** rejects unknown or malformed patches;
  `test_patch_compiler.py`.

### `PermissionPolicy`

- **Why/category:** service classifying changes by ownership/permission.
- **Caller:** controller and continuation pipeline.
- **Inputs/outputs:** parent constraints + patch + session decision in;
  assessment and granted/denied IDs out.
- **Invariant:** booked permission is session-scoped; never-allowed stays
  blocked.
- **Tests:** `test_permission_policy.py`.

### `CounterfactualProbeExecutor`

- **Why/category:** sandbox service for hypothetical consequence preview.
- **Caller:** controller.
- **Inputs/outputs:** `test_only` request in; hypothetical plan/diff/diagnostic
  evaluation out.
- **Invariant:** strips execution certificate and cannot become accepted.
- **Tests:** `test_probe_executor.py`.

### `run_permission_aware_research_pipeline()`

- **Why/category:** workflow orchestrator for pause or authorized continuation.
- **Caller:** interaction CLI.
- **Inputs/outputs:** pipeline inputs + options/controller/permission decision
  in; `PermissionAwarePipelineRun` out.
- **State:** paused interaction artifacts or a new immutable continuation run.
- **Invariant:** disabled mode delegates without interaction artifacts; grant
  creates a child run and preserves parent.
- **Tests:** `tests/interaction/test_pipeline_integration.py`.

## Probe vs Repair

See [interaction permission boundary](diagrams/interaction_probe_vs_authorized_repair.md).

| Hypothetical probe | Authorized repair |
|---|---|
| `test_only=true` | explicit confirmed request |
| may show hypothetical child/diff | produces immutable continuation run |
| diagnostic evaluation only | independent execution certificate |
| never saved as accepted plan | child stored with parent lineage |
| cannot grant permission | consumes explicit session-scoped permission |

> **Beginner note / 初学者提示:** “The system can preview this” does not mean
> “the system is allowed to do this.”

