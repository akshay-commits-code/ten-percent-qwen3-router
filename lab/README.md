# Design-time scripts for the revised step 5 (pool / dev only)

Records of what was tried before the final run. None of these reads a locked file. They were run from the repo root.

- `modal_prune_lab.py`, `prune_lab.py`, `prune.py`, `modal_prune_lab2.py`: self-contained ideas that were dropped (feedback removes more experts; feedback changes how many experts each token uses). Results: `work/step5/lab_a_hi-IN.json`, `work/step5/lab2.json`.
- `modal_restore_probe.py`: how much putting back the base's own pruned experts helps Hindi. Result: `work/step5/restore_probe_hi-IN.json`.
- `modal_gate_lab.py`: per-item scores of the annotated items before and after the top group, used to study the label test under many shuffles. Result: `work/step5/gatelab.json`.
- `../modal_grow_lab.py`: the regrow step itself on dev. Results: `work/step5/growlab_a_hi-IN.json` (group sizes), `work/step5/growlab_final_*.json` (dress rehearsal with the frozen constants).
- `work/step5/locale_probe.json`: base and small-model answers on each locale's `dev` (the locale choice).
