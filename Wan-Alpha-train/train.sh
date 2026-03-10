#!/bin/bash
export http_proxy=http://11.180.207.8:11113
export https_proxy=$http_proxy
export all_proxy=$http_proxy

if ! python3 -c "import deepspeed" &> /dev/null; then
    pip install -e .
    pip3 install deepspeed==0.15.4
    pip install modelscope
    pip install tensorboard
    pip install imageio
    pip install imageio-ffmpeg
    pip install xfuser==0.4.1 --no-deps
    pip install ftfy
else
    echo "pip requirements installed."
fi


ACCELERATE_CONFIG_FILE="configs/accelerate_config/accelerate_wxg.yaml"
RANK_NUM=$((WORLD_SIZE * 8))
MASTER_PORT=54321
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

job_name="12"
output_path="./checkpoints"
mkdir -p ${output_path}


for id in {0..7}; do
  CUDA_VISIBLE_DEVICES=$id accelerate launch \
    --mixed_precision='bf16' \
    --num_processes=1 \
    --num_machines=$WORLD_SIZE \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    examples/wanvideo/model_training/prepare_trans_alpha_mask.py \
    --dataset_metadata_path data.csv \
    --height 640 \
    --width 624 \
    --data_file_keys video_fgr,video_pha \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.t2v." \
    --output_path "" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --job_name $job_name \
    --job_id $id \
    --job_num 8 \
    --new_vae_path "VAE/pytorch_model.bin" \
    --use_gradient_checkpointing_offload  2>&1 | tee -a ${output_path}/train_prepare.log &
done
trap 'kill 0' SIGINT
wait



accelerate launch \
  --mixed_precision='bf16' \
  --num_processes=$RANK_NUM \
  --num_machines=$WORLD_SIZE \
  --machine_rank=$RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --config_file $ACCELERATE_CONFIG_FILE \
  examples/wanvideo/model_training/train_gauss_ellipse.py \
  --dataset_metadata_path data.csv \
  --height 640 \
  --width 624 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $output_path \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --initial_learnable_value 0.05 \
  --job_name $job_name \
  --use_gradient_checkpointing_offload 2>&1 | tee -a ${output_path}/train.log

