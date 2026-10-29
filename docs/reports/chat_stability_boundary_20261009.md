# Additional Copilot boundary repair review

20 new development cases reproduced 10 mismatches (8 unsafe suggestions and 2 unnecessary rejections). The repaired cases pass. The affected suite passes 242 checks; provider fixture tests pass 43. There were zero new paid calls.

[Open the full review](chat_stability_boundary_20261009.html)

Six actual deterministic website messages and 29 screenshots record input, submission, replies, corrected-window staging/reload, undo, and final conversation recovery. The test origin has no verified map. The accepted plan remains unchanged and the final draft is empty.

The additional production change is limited to copilot_typed_edits.py: use bound labels instead of time order; clarify unsupported bound directions; reject unresolved extra quantities and alternative positions; extract the new duration after removing its labelled old value.

The source is frozen. These are development regressions, not blind holdout or live-model accuracy evidence. Fresh live testing and the complete E2E-1/G5 gate remain open. The previous 165-screenshot repair report retains its earlier source scope.
