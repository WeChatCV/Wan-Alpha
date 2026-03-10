export http_proxy=http://11.180.207.8:11113
export https_proxy=$http_proxy
export all_proxy=$http_proxy


pip install -e .
pip install peft --upgrade
pip install av==13.1.0
pip install tensorboard
pip install deepspeed==0.15.4
pip install lightning
pip install imageio
pip install modelscope
pip install ftfy
pip install opencv-python
pip install imageio-ffmpeg
pip install seaborn


save_path="./checkpoints"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_vae.py \
  --train_dataset_path data.csv \
  --val_dataset_path val.csv \
  --output_path $save_path \
  --vae_path "Wan-2.1/Wan2.1-T2V-14B/Wan2.1_VAE.pth" \
  --num_frames 17 \
  --height 272 \
  --width 272 \
  --training_strategy deepspeed_stage_2 \
  --max_epochs 100 \
  --learning_rate 1e-4 \
  --lora_rank 128 \
  --batch_size 2 \
  --model_id_with_origin_paths "Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors" \
  --job_name "prompt" \
  --use_gradient_checkpointing_offload \
  --dataloader_num_workers 8 2>&1 | tee -a ${save_path}/train.log
