

#cd /mnt/petrelfs/yejinhui/Projects/Lara
#export PYTHONPATH=$(pwd):${PYTHONPATH}

port=6680
gpu_id=0
# export DEBUG=true
#export lara_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/Lara/bin/python

your_ckpt=./models/VLA-JEPA/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
