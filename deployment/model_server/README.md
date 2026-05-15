
# start policy server


```bash

your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16 \
    --rollout_trace_path ./runs/lara_rollouts.jsonl
```

`--rollout_trace_path` is optional. When set, the server appends one JSONL
record per `reset` or `record_outcome` message with raw MoE route traces such
as `router_probs_sequence`, `active_mask_sequence`, and `pool_mask_sequence`.
Those records can be summarized by `scripts/summarize_lara_protocol.py`.


# connect to policy server for debug

```bash
python deployment/model_server/debug_server_policy.py

# plus server_policy.py into your vla controler by ref to debug_server_policy.py
```
