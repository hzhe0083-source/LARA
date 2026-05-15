export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=disabled

accelerate launch \
  --config_file ./Lara/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./Lara/training/train_lara.py \
  --config_yaml ./scripts/config/lara_so101_ft.yaml
