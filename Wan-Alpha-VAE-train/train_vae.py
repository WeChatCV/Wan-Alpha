import os
import argparse
from einops import rearrange
import lightning as pl
from peft import LoraConfig, inject_adapter_in_model

import torch
import torchvision

from wan_video_vae_new import VideoVAE_, CausalConv3d, NewCausalConv3d, Mapping
from transparent_video_dataset_image import TextVideoDataset
from loss_tools import *
from vgg_loss import VGGLoss
from diffsynth.pipelines.wan_video_maskpre_new import WanVideoPipeline, ModelConfig

from loss_mask_find_rgb_pha_3value_semi import gen_gauss_mask_for_video

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self, vae_path, learning_rate=1e-5, lora_rank=4, init_lora_weights="kaiming", out_path="checkpoints/wan_vae", trained_model_path=None
    ):
        super().__init__()
        # 1. Init VAE
        self.vae_fgr, self.vae_pha = VideoVAE_(z_dim=16), VideoVAE_(z_dim=16)
        self.vae_fgr.load_state_dict(torch.load(vae_path, map_location=torch.device("cpu")))
        self.vae_pha.load_state_dict(torch.load(vae_path, map_location=torch.device("cpu")))

        del self.vae_pha.encoder
        del self.vae_pha.conv1
        del self.vae_fgr.encoder.head
        del self.vae_fgr.conv1

        def replace_causalconv3d(model):
            for name, module in model.named_modules():
                if isinstance(module, CausalConv3d) and not hasattr(module, "conv3D"):
                    new_module = NewCausalConv3d(module)

                    if "." not in name:
                        parent_module = model
                        setattr(parent_module, name.rsplit('.', 1)[-1], new_module)
                    else:
                        parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                        setattr(parent_module, name.rsplit('.', 1)[-1], new_module)

        replace_causalconv3d(self.vae_fgr.decoder)
        replace_causalconv3d(self.vae_pha.decoder)

        self.vae_fgr.requires_grad_(False)
        self.vae_pha.requires_grad_(False)

        self.vae_fgr.eval()
        self.vae_pha.eval()

        # 2. Add LoRA to Decoder
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        def get_specific_layer_names(model):
            # Create a list to store the layer names
            layer_names = []
            # Recursively visit all modules and submodules
            for name, module in model.named_modules():
                # Check if the module is an instance of the specified layers
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    # model name parsing 
                    layer_names.append(name)
            return layer_names

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=1.0,
            init_lora_weights=init_lora_weights,
            target_modules=get_specific_layer_names(self.vae_fgr.decoder), # 'to_qkv', 'proj'
        )
        self.vae_fgr.decoder = inject_adapter_in_model(lora_config, self.vae_fgr.decoder)
        self.vae_pha.decoder = inject_adapter_in_model(lora_config, self.vae_pha.decoder)

        # 3. Add Merge Module
        mid_dim = 192
        self.middle = Mapping(mid_dim * 4, 16, mid_dim=mid_dim * 2)

        # 4. Dtype
        self.torch_dtype = torch.bfloat16
        self.vae_fgr.to(dtype=self.torch_dtype)
        self.vae_pha.to(dtype=self.torch_dtype)
        self.middle.to(dtype=self.torch_dtype)

        # 5. Loss Functions
        self.sobel_edge_loss = VideoSobelEdgeLoss()
        self.sobel_edge_loss.eval()

        self.perceptual_loss = VGGLoss()
        self.perceptual_loss.to(dtype=self.torch_dtype)
        self.perceptual_loss.eval()

        # 6. Others
        self.learning_rate = learning_rate
        self.out_path = out_path

        self.bg_color = [
            torch.tensor([-1, -1, -1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([-1, -1, 1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([-1, 1, -1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([-1, 1, 1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([1, -1, -1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([1, -1, 1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([1, 1, -1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
            torch.tensor([1, 1, 1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype),
        ]

        # if trained_model_path:
        #     state_dict = torch.load(trained_model_path, map_location='cpu')
        #     state_dict_vae = {}
        #     self.load_state_dict(state_dict)

        model_configs = []

        model_id_with_origin_paths = args.model_id_with_origin_paths
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=self.device, model_configs=model_configs, skip_download=True, redirect_common_files=False,
        local_model_path="Wan-AI/",
        tokenizer_config=ModelConfig(model_id="Wan2.1-T2V-1.3B", origin_file_pattern="google/*"))
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([])
        self.use_gradient_checkpointing = True
        self.use_gradient_checkpointing_offload = args.use_gradient_checkpointing_offload

    def pad_rgb(self, video_fgr, video_pha, num, is_hard=False):
        frame_alpha = (video_pha + 1.) * 0.5
        if is_hard:
            frame_alpha = torch.where(frame_alpha > 0, 1, 0)
        
        fg = video_fgr * frame_alpha + self.bg_color[num].to(self.device) * (1.0 - frame_alpha)
        return fg.to(self.device, dtype=self.torch_dtype)

    def process_to_save_video(self, tensor):
        tensor = rearrange(tensor[0], "C T H W -> T H W C")
        tensor = tensor * 0.5 + 0.5
        return torch.clamp(tensor, 0, 1) * 255

    def forward_preprocess(self, data, rgb_mask):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": ""}
        # print("data size:", data.size())
        inputs_nega = {}

        # You need to replace this with the path to your cached data.
        cache_path = ["data_cache/500_video_0"]
        
        inputs_shared = {
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        # "input_video": data["video"],
        "cache_path": cache_path,
        "latent_height": data.size(3),
        "latent_width": data.size(4),
        "num_frames": data.size(2),
        "batch_size": args.batch_size,
        "job_name": args.job_name,
        "rgb_mask": rgb_mask.to(self.device, dtype=torch.float32),
        # Please do not modify the following parameters
        # unless you clearly know what this will cause.
        "cfg_scale": 1,
        "tiled": False,
        "rand_device": self.pipe.device,
        "use_gradient_checkpointing": self.use_gradient_checkpointing,
        "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        "cfg_merge": False,
        # "vace_scale": 1,
        }
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
        
    def training_step(self, batch, batch_idx):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # --- Data ---
        video_fgr = batch["video_fgr"].to(self.device, dtype=self.torch_dtype)
        video_pha = batch["video_pha"].to(self.device, dtype=self.torch_dtype)
        rgb_mask = gen_gauss_mask_for_video(video_pha)
        rgb_mask = rgb_mask.unsqueeze(1)


        ### input for vae encoder
        video_fgr_render_enc = self.pad_rgb(video_fgr.float(), video_pha.float(),
                                            torch.randint(low=0, high=8, size=(1,))[0], is_hard=True)

        ### for blender loss, hard alpha and soft alpha
        random_bg_color_hard = torch.randint(low=0, high=8, size=(1,))
        video_fgr_hard_alpha_recon = self.pad_rgb(video_fgr.float(), video_pha.float(), random_bg_color_hard[0], is_hard=True)

        random_bg_color_soft = torch.randint(low=0, high=8, size=(1,))
        video_fgr_soft_alpha_recon = self.pad_rgb(video_fgr.float(), video_pha.float(), random_bg_color_soft[0], is_hard=False)  

        # --- VAE Encode ---
        fgr_latents = self.vae_fgr.encode_simple(video_fgr_render_enc)
        pha_latents = self.vae_fgr.encode_simple(video_pha)

        merged_latents = self.middle(torch.cat([fgr_latents, pha_latents], dim=1))

        # --- T2V ---
        inputs = self.forward_preprocess(merged_latents, rgb_mask)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        inputs["input_latents"] = merged_latents
        #.to(self.device)
        t2v_loss = self.pipe.training_loss(**models, **inputs)

        # --- VAE Decode ---
        recon_video_fgr = self.vae_fgr.decode_simple(merged_latents)
        recon_video_pha = self.vae_pha.decode_simple(merged_latents)

        # --- Loss ---
        ### for blender loss, hard alpha and soft alpha
        recon_video_fgr_hard_alpha_recon = self.pad_rgb(recon_video_fgr.float(), recon_video_pha.float(), random_bg_color_hard[0], is_hard=True)
        recon_video_fgr_soft_alpha_recon = self.pad_rgb(recon_video_fgr.float(), recon_video_pha.float(), random_bg_color_soft[0], is_hard=False)

        ### l1 loss
        loss_l1_fgr_hard = torch.mean(torch.abs(video_fgr_hard_alpha_recon - recon_video_fgr_hard_alpha_recon))
        loss_l1_fgr_soft = torch.mean(torch.abs(video_fgr_soft_alpha_recon - recon_video_fgr_soft_alpha_recon))
        loss_l1_pha = torch.mean(torch.abs(video_pha - recon_video_pha))
        loss_recon = loss_l1_fgr_hard + loss_l1_fgr_soft+ loss_l1_pha

        ### perceptual loss
        loss_per_fgr_hard = self.perceptual_loss(video_fgr_hard_alpha_recon, recon_video_fgr_hard_alpha_recon)
        loss_per_fgr_soft = self.perceptual_loss(video_fgr_soft_alpha_recon, recon_video_fgr_soft_alpha_recon)
        loss_per_pha = self.perceptual_loss(video_pha, recon_video_pha)
        loss_per = loss_per_fgr_hard + loss_per_fgr_soft + loss_per_pha

        ### sobel loss
        loss_sobel_fgr_hard = self.sobel_edge_loss(video_fgr_hard_alpha_recon, recon_video_fgr_hard_alpha_recon)
        loss_sobel_fgr_soft = self.sobel_edge_loss(video_fgr_soft_alpha_recon, recon_video_fgr_soft_alpha_recon)
        loss_sobel_pha = self.sobel_edge_loss(video_pha, recon_video_pha)
        loss_sobel = loss_sobel_fgr_hard + loss_sobel_fgr_soft + loss_sobel_pha

        t2v_loss = 0.1*t2v_loss
        loss = loss_recon + loss_per + loss_sobel + t2v_loss

        # --- Log ---
        self.log("total_loss", loss, prog_bar=True)
        self.log("total_recon", loss_recon, prog_bar=True)
        self.log("total_per", loss_per, prog_bar=True)
        self.log("total_sobel", loss_sobel, prog_bar=True)

        self.log("loss_l1_fgr_hard", loss_l1_fgr_hard, prog_bar=True)
        self.log("loss_l1_fgr_soft", loss_l1_fgr_soft, prog_bar=True)
        self.log("loss_l1_pha", loss_l1_pha, prog_bar=True)
        
        self.log("loss_per_fgr_hard", loss_per_fgr_hard, prog_bar=True)
        self.log("loss_per_fgr_soft", loss_per_fgr_soft, prog_bar=True)
        self.log("loss_per_pha", loss_per_pha, prog_bar=True)

        self.log("loss_sobel_fgr_hard", loss_sobel_fgr_hard, prog_bar=True)
        self.log("loss_sobel_fgr_soft", loss_sobel_fgr_soft, prog_bar=True)
        self.log("loss_sobel_pha", loss_sobel_pha, prog_bar=True)

        self.log("loss_t2v", t2v_loss, prog_bar=True)
        
        if self.global_step % 1000 == 0 and self.trainer.is_global_zero: 
            torchvision.io.write_video(os.path.join(self.out_path, f"recon_video_fgr_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr, video_fgr], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"recon_video_fgr_hard_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr_hard_alpha_recon, video_fgr_hard_alpha_recon], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"recon_video_fgr_soft_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr_soft_alpha_recon, video_fgr_soft_alpha_recon], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"recon_video_pha_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_pha, video_pha], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
        return loss
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # --- Data ---
        video_fgr = batch["video_fgr"].to(self.device, dtype=self.torch_dtype)
        video_pha = batch["video_pha"].to(self.device, dtype=self.torch_dtype)
        rgb_mask = gen_gauss_mask_for_video(video_pha)
        rgb_mask = rgb_mask.unsqueeze(1)

        ### input for vae encoder
        video_fgr_render_enc = self.pad_rgb(video_fgr.float(), video_pha.float(),
                                            torch.randint(low=0, high=8, size=(1,))[0], is_hard=True)

        ### for blender loss, hard alpha and soft alpha
        random_bg_color_hard = torch.randint(low=0, high=8, size=(1,))
        video_fgr_hard_alpha_recon = self.pad_rgb(video_fgr.float(), video_pha.float(), random_bg_color_hard[0], is_hard=True)

        random_bg_color_soft = torch.randint(low=0, high=8, size=(1,))
        video_fgr_soft_alpha_recon = self.pad_rgb(video_fgr.float(), video_pha.float(), random_bg_color_soft[0], is_hard=False)  

        # --- VAE Encode ---
        fgr_latents = self.vae_fgr.encode_simple(video_fgr_render_enc)
        pha_latents = self.vae_fgr.encode_simple(video_pha)

        merged_latents = self.middle(torch.cat([fgr_latents, pha_latents], dim=1))

        # --- T2V ---
        inputs = self.forward_preprocess(merged_latents,rgb_mask)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        inputs["input_latents"] = merged_latents
        t2v_loss = self.pipe.training_loss(**models, **inputs)

        # --- VAE Decode ---
        recon_video_fgr = self.vae_fgr.decode_simple(merged_latents)
        recon_video_pha = self.vae_pha.decode_simple(merged_latents)

        # --- Loss ---
        ### for blender loss, hard alpha and soft alpha
        recon_video_fgr_hard_alpha_recon = self.pad_rgb(recon_video_fgr.float(), recon_video_pha.float(), random_bg_color_hard[0], is_hard=True)
        recon_video_fgr_soft_alpha_recon = self.pad_rgb(recon_video_fgr.float(), recon_video_pha.float(), random_bg_color_soft[0], is_hard=False)

        ### l1 loss
        loss_l1_fgr_hard = torch.mean(torch.abs(video_fgr_hard_alpha_recon - recon_video_fgr_hard_alpha_recon))
        loss_l1_fgr_soft = torch.mean(torch.abs(video_fgr_soft_alpha_recon - recon_video_fgr_soft_alpha_recon))
        loss_l1_pha = torch.mean(torch.abs(video_pha - recon_video_pha))
        loss_recon = loss_l1_fgr_hard + loss_l1_fgr_soft+ loss_l1_pha

        ### perceptual loss
        loss_per_fgr_hard = self.perceptual_loss(video_fgr_hard_alpha_recon, recon_video_fgr_hard_alpha_recon)
        loss_per_fgr_soft = self.perceptual_loss(video_fgr_soft_alpha_recon, recon_video_fgr_soft_alpha_recon)
        loss_per_pha = self.perceptual_loss(video_pha, recon_video_pha)
        loss_per = loss_per_fgr_hard + loss_per_fgr_soft + loss_per_pha

        ### sobel loss
        loss_sobel_fgr_hard = self.sobel_edge_loss(video_fgr_hard_alpha_recon, recon_video_fgr_hard_alpha_recon)
        loss_sobel_fgr_soft = self.sobel_edge_loss(video_fgr_soft_alpha_recon, recon_video_fgr_soft_alpha_recon)
        loss_sobel_pha = self.sobel_edge_loss(video_pha, recon_video_pha)
        loss_sobel = loss_sobel_fgr_hard + loss_sobel_fgr_soft + loss_sobel_pha

        t2v_loss = 0.1*t2v_loss
        loss = loss_recon + loss_per + loss_sobel + t2v_loss

        # --- Log ---
        self.log("val_total_loss", loss, prog_bar=True)
        self.log("val_total_recon", loss_recon, prog_bar=True)
        self.log("val_total_per", loss_per, prog_bar=True)
        self.log("val_total_sobel", loss_sobel, prog_bar=True)

        self.log("val_loss_l1_fgr_hard", loss_l1_fgr_hard, prog_bar=True)
        self.log("val_loss_l1_fgr_soft", loss_l1_fgr_soft, prog_bar=True)
        self.log("val_loss_l1_pha", loss_l1_pha, prog_bar=True)
        
        self.log("val_loss_per_fgr_hard", loss_per_fgr_hard, prog_bar=True)
        self.log("val_loss_per_fgr_soft", loss_per_fgr_soft, prog_bar=True)
        self.log("val_loss_per_pha", loss_per_pha, prog_bar=True)

        self.log("val_loss_sobel_fgr_hard", loss_sobel_fgr_hard, prog_bar=True)
        self.log("val_loss_sobel_fgr_soft", loss_sobel_fgr_soft, prog_bar=True)
        self.log("val_loss_sobel_pha", loss_sobel_pha, prog_bar=True)

        self.log("val_loss_t2v", t2v_loss, prog_bar=True)
        
        if self.trainer.is_global_zero and not os.path.exists(f"val_recon_video_fgr_{self.global_step}.mp4"):
            torchvision.io.write_video(os.path.join(self.out_path, f"val_recon_video_fgr_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr, video_fgr], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"val_recon_video_fgr_hard_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr_hard_alpha_recon, video_fgr_hard_alpha_recon], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"val_recon_video_fgr_soft_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_fgr_soft_alpha_recon, video_fgr_soft_alpha_recon], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
            torchvision.io.write_video(os.path.join(self.out_path, f"val_recon_video_pha_{self.global_step}.mp4"),
                                                    self.process_to_save_video(torch.cat([recon_video_pha, video_pha], dim=3).cpu()),
                                                    fps=16, video_codec="h264")
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, list(self.vae_fgr.parameters()) + list(self.vae_pha.parameters()) + list(self.middle.parameters()))
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=1000,
        help="How often within one training epoch to check the validation set.",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=None,
        help="Path of trained .bin file.",
    )
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--job_name", type=str, default="new", help="Different name evey time.")
    args = parser.parse_args()
    return args


    
def train(args):
    train_dataset = TextVideoDataset(
        args.train_dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    
    val_dataset = TextVideoDataset(
        args.val_dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )

    train_dataset.data = [d for d in train_dataset.data if d not in val_dataset.data]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )

    model = LightningModelForTrain(
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        init_lora_weights=args.init_lora_weights,
        out_path=args.output_path,
        trained_model_path = args.trained_model_path
    )
    
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = [TensorBoardLogger(os.path.join(args.output_path, "tensorboard"))]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    train(args)
