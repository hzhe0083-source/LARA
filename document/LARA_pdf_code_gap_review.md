# LARA PDF-to-Code and Paper Logic Review

Date: 2026-05-17

Authoritative paper source: `document/LARA__Latent_Action_Routing_for_Embodied_Control.pdf`

Compared implementation areas:

- `Lara/model/framework/act.py`
- `Lara/model/modules/action_model/lara_latent.py`
- `Lara/model/modules/action_model/lara_moe.py`
- `Lara/model/framework/Lara_core.py`
- `Lara/dataloader/lerobot_v3_datasets.py`
- `scripts/config/lara_so101_ft.yaml`
- `scripts/config/lara_libero100_baseline.yaml`
- `scripts/run_lara_libero100_experiment.py`
- `scripts/audit_lara_paper_readiness.py`
- `Lara/evaluation/lara_protocol.py`

## Review Status

This document is a paper/code review artifact and patch-status tracker.

Severity labels:

- `P0`: must fix before claiming the PDF method is implemented.
- `P1`: should fix before serious benchmark runs.
- `P2`: documentation, scope, or experimental-rigor issue.

## Executive Summary

The repository has moved beyond a simple VLA-JEPA action adapter and now contains real scaffolding for posterior latent actions, VQ codes, prior prediction, MoE routing, direct action experts, episode pools, route diagnostics, and utility sidecar plumbing.

It is still not a completed implementation of the method described in the PDF. The largest remaining gaps are:

1. The transition head predicts raw proprioceptive state targets, while the PDF defines latent control-state prediction through an encoder target.
2. The default utility path uses reconstruction/state proxy labels, not real counterfactual downstream control utility.
3. The episode-level pool router often falls back to the current chunk context instead of the episode initial context required by the PDF.
4. Past actions are accepted in the action-head API but not actually used to form the context.
5. There is no completed training/evaluation evidence for counterfactual utility, matched resident budgets, closed-loop success, or benchmark tables.

One direct implementation bug was fixed after this review: the pairwise utility ranking loss is now a separate weighted term in the MoE total loss.

The PDF itself is mostly careful about framing the empirical result as an intended claim, but it still needs stronger formal definitions and stricter separation between proposal, implemented code, and validated evidence.

## Highest-Priority Action Table

| Priority | Issue | Why it matters | First concrete action |
| --- | --- | --- | --- |
| P0 | Utility rank loss was not explicitly in `total_loss` | PDF Eq. 34/35 says ranking is part of training | Fixed: return utility regression and rank loss separately, add `utility_rank_loss_weighted` to MoE total loss, and test the accounting |
| P0 | Transition target is raw state, not latent `h` | PDF claims latent boundary-state prediction through `F_bar` | Choose raw-state paper wording or implement encoded latent target |
| P0 | No real counterfactual utility evidence | Paper's utility-calibrated claim depends on downstream route outcomes | Collect forced-route rollout sidecar and train with it |
| P1 | Episode pool can be chunk-local | Resident pool claim requires stable episode-level selection | Require episode-start context and cache/reuse pool masks in evaluation |
| P1 | Past actions are ignored | PDF context includes `a_{t-k:t-1}` | Remove from PDF or implement past-action fusion |
| P1 | PDF is stale relative to current TeX/code horizons | Current code uses 60/10 for SO101 and 30/10 for first-pass LIBERO100 | Regenerate PDF after finalizing horizon text |

## Evidence Snapshot

The repository's own paper-readiness audit fails without real evidence:

```text
python3 scripts/audit_lara_paper_readiness.py --config scripts/config/lara_so101_ft.yaml --min-training-steps 1
```

Required missing evidence:

- `counterfactual_utility_sidecar`
- `closed_loop_protocol_records`
- `full_so101_training_artifact`
- `closed_loop_robot_eval_artifact`

The audit passes config-shape checks for SO101, default-on LARA flags, and the 60/10 SO101 horizon contract, but that is configuration evidence only. It is not evidence that the algorithm works.

Code evidence anchors:

- Transition head emits `state_dim`: `Lara/model/modules/action_model/lara_latent.py:47-66`.
- LIBERO v3 collate builds boundary targets from raw `observation.state`: `Lara/dataloader/lerobot_v3_datasets.py:109-121`.
- `past_actions` is converted but not used: `Lara/model/framework/act.py:369-370`.
- Direct action expert posterior is reconstruction-loss based: `Lara/model/framework/act.py:465-506`.
- Utility proxy labels are derived from expert losses: `Lara/model/modules/action_model/lara_moe.py:622-690`.
- Episode pool falls back to current chunk context when no initial context is passed: `Lara/model/modules/action_model/lara_moe.py:1376-1399`.
- `utility_rank_loss_weighted` is now included in `total_loss`: `Lara/model/modules/action_model/lara_moe.py`.
- LIBERO100 first-pass config uses 30/10 horizons and avg+max pool target: `scripts/config/lara_libero100_baseline.yaml:22-65`.
- Staged launcher disables utility before `utility_proxy`: `scripts/run_lara_libero100_experiment.py:115-192`.

PDF/TeX status:

- The authoritative PDF is a 12-page file generated on 2026-05-15.
- `document/LARA_collapse_paper.tex` has newer material that is not reflected in that PDF, including 60/10 horizon text and an implementation-status section.
- Because this review is explicitly PDF-based, newer TeX text is treated as draft material unless the PDF is regenerated.

## Code Versus PDF: Major Gaps

### 1. [P0] Transition Target Is Raw State, Not Latent State

PDF claim:

- Eq. 5/16: `h_hat_{t+H} = T_psi(h_t, z_t^a)`.
- Eq. 6/17: target is `h_bar_{t+H} = F_bar(o_{t+H}, g, p_{t+H}, a_{...})`.
- The target is a latent control state produced by an encoder or target encoder.

Code reality:

- `LatentActionTransitionHead` outputs `num_boundaries * state_dim`, not hidden-state dimension.
- `lerobot_v3_datasets.collate_fn()` builds `execution_state_target` and `prediction_state_target` directly from `observation.state`.
- `ActionHeadAdapter._transition_loss()` compares predicted vectors to those raw state targets.

Impact:

This is not the PDF's latent-state world model target. It is a proprioceptive boundary-state regression head. That can be useful, but the paper should not imply that the implemented transition target is `F_bar` latent state unless the code actually encodes future observations into latent states.

Required fix:

- Either revise the paper to say the current implementation predicts proprioceptive boundary state, or implement `F_bar(o_{t+H}, g, p_{t+H}, ...)` latent target extraction and train the transition head in that space.
- Patch status: `scripts/audit_lara_paper_readiness.py` now refuses full paper-readiness training artifacts unless they declare latent transition targets through `uses_latent_transition_targets` or `lara_transition_target_type: latent_state`. This prevents raw-state transition training from being mistaken for the PDF method, but it does not implement latent targets.

### 2. [Fixed P0] Utility Ranking Loss Is Explicitly Optimized

PDF claim:

- Eq. 34 defines `L_rank`.
- Eq. 35 includes `lambda_r L_rank` in the full objective.

Previous code reality:

- `LatentActionMoE.forward()` computes `utility_rank_loss_weighted`.
- `total_loss` omits `utility_rank_loss_weighted`.

Impact before fix:

The code logs ranking loss but does not train on it. This is a direct mismatch with the PDF's full objective and a likely implementation bug.

Patch status:

- `utility_calibration_objective()` now returns utility regression loss and pairwise rank loss as separate raw terms.
- `LatentActionMoE.forward()` now adds `utility_rank_loss_weighted` to `total_loss`.
- `tests/test_lara_moe.py` now checks that the raw calibration objective does not fold rank into regression and that the forward loss accounts for both weighted terms.

### 3. [P1] Past Actions Are in the Paper Context but Ignored in the Action Head

PDF claim:

- Eq. 3: `h_t = F_theta(o_t, g, p_t, a_{t-k:t-1})`.

Code reality:

- `ActionHeadAdapter.forward()` accepts `past_actions`.
- It only converts `past_actions` to a tensor, then never uses it.
- `QwenActionTokenizer.encode()` only injects image, instruction, latent action tokens, and embodied action placeholder tokens.

Impact:

The implemented context is closer to `F_theta(o_t, g, p_t)` plus learned placeholder tokens, not the PDF's explicit past-action-conditioned context.

Required fix:

- Either remove `a_{t-k:t-1}` from the paper context definition for the current implementation, or add a real past-action encoder/fusion path.

### 4. [P1] Episode Pool Router Is Not Reliably Episode-Level

PDF claim:

- Eq. 22: `P_tau = TopD(p_chi(m | g, h_1, b), d_tau)`.
- The pool is selected from the goal, initial context, and deployment budget.

Code reality:

- `LatentActionMoE._resident_pool()` uses `initial_context_tokens` when provided, otherwise falls back to current `conditioning_tokens`.
- `lara_libero100_baseline.yaml` and the ordinary v3 collate path have `include_episode_start: false`.
- `Lara.predict_action()` selects a resident pool from the current inference context unless a caller supplies an external resident pool mask.

Impact:

The pool can become a chunk-local top-D router, not a stable episode-level resident pool. This weakens the paper's resident-memory claim unless evaluation explicitly caches and reuses `resident_pool_mask` from an episode-start context.

Required fix:

- Make `include_episode_start` mandatory for paper two-level routing experiments.
- In evaluation, choose the pool once at episode start and pass the same `resident_pool_mask` for later chunks.
- Report whether the pool was selected from `h_1` or from each chunk.

### 5. [P1] Pool Teacher Differs from the PDF

PDF claim:

- Eq. 23 defines the pool target as the average posterior responsibility over chunks:
  `q_bar_tau(m) = (1 / |T_tau|) sum_t q_post(m | h_t, a_{t:t+H-1})`.

Code reality:

- `aggregate_episode_responsibilities()` supports average, max, and utility-weighted targets.
- The SO101/LIBERO configs default to `avg_weight=1.0` and `max_weight=1.0`.

Impact:

The code is using a stronger target than the PDF formula. This may be a better design because max preserves rare critical experts, but it is not Eq. 23 as written in the PDF.

Required fix:

- Update the PDF formula to include avg/max/utility pool target weights, or set `lara_pool_target_max_weight: 0.0` for experiments claiming Eq. 23.

### 6. [P0] Utility Calibration Is Mostly Proxy-Based, Not Counterfactual Control Utility

PDF claim:

- Eq. 31 defines candidate route utility from downstream control quantities.
- Eq. 32-34 train centered utility and pairwise route ranking.
- The text frames this as counterfactual utility fine-tuning.

Code reality:

- When no sidecar is provided, `utility_scores` are derived from direct-expert action reconstruction losses or state-transition consistency losses.
- Sidecar plumbing exists, but the audit confirms no real sidecar/evaluator labels are present.
- `utility_proxy` stage intentionally uses small proxy weights, but direct config launches can still enable proxy utility losses with weight 1.0.

Impact:

Current utility training is not real counterfactual downstream control utility. It is an offline proxy for reconstruction or state consistency. That is useful as a bootstrap, but it cannot support the paper's utility-calibrated routing claim.

Required fix:

- Treat proxy utility as a warm-up or ablation.
- Collect forced-route closed-loop or evaluator-side labels and train with `counterfactual_utility_labels_path`.
- In paper tables, distinguish `LARA-proxy` from `LARA-counterfactual`.

### 7. [P1] Stage Order Exists in the Launcher, but Config Defaults Can Bypass It

PDF claim:

- Algorithm 1 and the utility section imply a staged process: latent/action reconstruction, expert responsibility, router distillation, then later utility fine-tuning.

Code reality:

- `scripts/run_lara_libero100_experiment.py` has explicit staged overrides that disable utility in dense/latent/experts/router/joint stages and use small weights in `utility_proxy`.
- Base configs enable latent, MoE, direct action experts, and utility proxy losses by default.

Impact:

The staged launcher is much closer to the PDF. Directly launching the YAML is not. The paper and README should state that paper-aligned training must use the staged launcher or equivalent staged configs.

Required fix:

- Add a fail-fast warning when training a paper config directly with all default-on proxy losses.
- Or split base configs into actual stage-specific YAMLs with utility off until the final stage.

### 8. [P1] Training Uses Dense Posterior Weights, Not Sparse Top-K Weights

PDF claim:

- Eq. 27-28 define top-k active experts and normalized sparse route weights.

Code reality:

- During training with a posterior teacher, `LatentActionMoE.forward()` uses `weights = posterior_probs`, not the top-k `router_probs`, to mix residual expert outputs.
- Sparse `router_probs` and `active_mask` are still computed, logged, and used at inference.

Impact:

This creates a teacher-forced dense training path and a sparse inference path. That may stabilize training, but it is a train-inference mismatch and should be explicit in the method.

Required fix:

- Either train with sparse posterior top-k weights, or describe teacher-forced dense posterior expert mixing as a training-only relaxation and measure the gap to sparse inference.

### 9. [P2] Continuous Latent Variant Is Not Implemented

PDF claim:

- Section 4.2 presents both discrete and continuous latent action variants.

Code reality:

- `LatentActionHead` implements only the discrete VQ path with `VectorQuantizer` and `LatentActionPrior`.

Impact:

This is acceptable if the paper says the implementation uses the discrete variant. It is not acceptable if experiments claim both variants.

Required fix:

- Mark continuous latent actions as future work or add a variational latent path and ablation.

### 10. [P2] PDF Experimental Stack Is Not Implemented

PDF claim:

- Minimal stack: ManiSkill3, LIBERO, CALVIN.
- Tables report LIBERO and CALVIN placeholders.

Code reality:

- The repository has LIBERO100 and MetaWorld MT50 dataloader/config paths.
- There is no visible ManiSkill3 or CALVIN training/evaluation implementation.
- PDF tables are still `TBD`.

Impact:

The current code can support early LIBERO100 experiments, not the full PDF experimental protocol.

Required fix:

- Narrow the PDF benchmark scope to what will actually be run first, or add missing benchmark integrations.

## Paper Logic and Rigor Issues

### 1. Posterior Responsibility Is Circular Without a Precise Schedule

The PDF says responsibilities identify which expert explains each chunk, and experts are trained using those responsibilities. But early experts are weak, so the posterior can be arbitrary or collapsed. The Limitations section mentions this, but the method needs a concrete schedule:

- dense warm start duration,
- when experts are enabled,
- entropy or posterior smoothing schedule,
- when the posterior becomes trusted,
- how dead experts are revived or detected.

Without this, "latent-action experts specialize through posterior responsibility" is plausible but underspecified.

### 2. Utility Objective Blurs Offline Reconstruction and Downstream Control

Eq. 31 combines action reconstruction, state distance, value/progress, uncertainty, and cost. If value/progress comes from proxies rather than real outcomes, the objective is not clearly downstream control utility. The paper should separate:

- demonstration-likelihood utility,
- latent-state consistency utility,
- simulator/robot counterfactual utility,
- sparse success/reward value utility.

Each should have different claim strength.

### 3. Episode Average Pool Target Can Drop Rare Critical Experts

The PDF pool teacher averages posterior responsibility over the episode. Rare but essential experts can disappear under averaging. The code already compensates with a max term, which is a sign that the PDF formula is underpowered.

The paper should include either:

- avg + max + utility target, or
- an explicit rare-expert coverage loss and diagnostic.

### 4. Matched-Compute and Resident-Budget Protocol Needs Accounting Rules

The PDF says all comparisons must report total parameters, resident parameters, active parameters, FLOPs, latency, and VRAM. It does not fully define:

- whether shared VLA backbone FLOPs are included,
- whether all experts are physically loaded or truly resident-subset loaded,
- how residual-token experts and direct-action experts are counted,
- how candidate utility evaluation cost is accounted during training and evaluation,
- whether route-pool selection cost is amortized over the episode.

Without these rules, "matched compute" can be interpreted inconsistently.

### 5. `Utility-Calibrated Experts` Is Too Strong Without Counterfactual Labels

The title and conclusion call the experts utility-calibrated. The PDF does frame results as intended, but the wording still risks overclaiming if only proxy utility is implemented.

Safer phrasing until labels exist:

- "Latent Action Routing with Utility-Calibrated Routing"
- "Toward Utility-Calibrated Experts"
- "Proxy-Utility-Calibrated Experts" for current code.

### 6. PDF Does Not Include the Current 60/10 or 30/10 Horizon Design

The PDF version uses a single chunk horizon `H`. The current TeX and code distinguish prediction horizon and execution horizon. This matters because:

- training may predict 60 or 30 frames,
- latent/router/utility may use only the first 10 frames,
- closed-loop execution only applies the first 10 frames.

Because the user asked to use the PDF as source, the PDF is currently behind the code. The PDF should be regenerated from the updated TeX or edited to include the receding-horizon contract.

### 7. Results Tables Are Placeholders

Tables 1 and 2 are entirely `TBD`. That is acceptable for a design note, not for a paper making empirical claims. Until runs exist, the abstract and conclusion should keep using "hypothesis", "proposal", or "intended claim" language.

## What Is Already Aligned

The code does align with several central PDF mechanisms:

- posterior latent action encoder conditioned on context and future action chunk;
- discrete VQ codebook with prior prediction from current context;
- action reconstruction loss for latent actions;
- expert posterior from per-expert reconstruction losses;
- context-only chunk router;
- episode pool router and top-k active routing machinery;
- direct action experts capable of chunk reconstruction;
- action masks for trajectory-end chunks;
- route diagnostics and protocol summarization scaffolding.

These are meaningful pieces, but they are still closer to research scaffolding plus early integration than completed LARA evidence.

## Recommended Priority Fixes

1. Add `utility_rank_loss_weighted` to MoE `total_loss` and test it.
2. Decide whether transition targets are raw state or latent `h`; make paper and code match.
3. Make episode-start context mandatory for two-level routing experiments.
4. Disable proxy utility by default outside explicit `utility_proxy` stage.
5. Add real counterfactual utility sidecar collection from forced-route rollouts.
6. Add a training artifact format that records final losses, enabled flags, checkpoint path, data split, and horizon contract.
7. Regenerate the PDF from the updated TeX after the 60/10 and implementation-status sections are finalized.
8. Narrow the experimental protocol to LIBERO100 first, then expand to CALVIN/ManiSkill only when implementation exists.

## Completion Checklist for Paper-Ready Claims

Before claiming the full PDF method is implemented, the repository should have:

- completed staged training logs for dense, latent, experts, router, and utility/counterfactual stages;
- trained checkpoints for at least Dense VLA, Generic MoE, Latent-Action MoE, LARA, and LARA + route pool;
- real counterfactual utility labels with at least two candidate experts per context;
- closed-loop LIBERO100 rollout records with success, latency, FLOPs, VRAM, route traces, resident fractions, and route diagnostics;
- evidence that episode pools are selected once per episode and reused;
- route diagnostics showing posterior-router KL, teacher mass at resident pool, teacher mass at active top-k, critical expert miss rate, route regret, route-switch frequency, and expert usage entropy;
- matched active compute and resident parameter accounting;
- ablations for no latent code, generic MoE, distillation-only, utility-only, no episode pool, and varying resident pool size.
