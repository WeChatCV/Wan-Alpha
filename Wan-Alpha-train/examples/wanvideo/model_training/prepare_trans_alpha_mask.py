import torch, os, json
import sys
from diffsynth.pipelines.wan_video_new_trans_random_pre import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from diffsynth.models.wan_video_vae_merge_new import MergedWanVideoVAE

import imageio, os, torch, warnings, torchvision, argparse
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
import glob
import torchvision.transforms.functional as TF

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--job_name", type=str, default="new", help="Different name evey time.")
    parser.add_argument("--job_id", type=int, default=0, help="Gradient accumulation steps.")
    parser.add_argument("--job_num", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--new_vae_path", type=str,  help="New vae path.")
    return parser



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        metadata = pd.read_csv(metadata_path)
        self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            

    
    def get_height_width(self, image):
        width, height = image.size
        original_aspect = width / height        
        
        target_area = 640 * 624

        best_width = 0
        best_height = 0
        min_diff = float('inf')
        
        for w in range(1, 2000):
            h = target_area // w
            if h * w != target_area:
                continue
            current_aspect = w / h
            aspect_diff = abs(current_aspect - original_aspect)
            if aspect_diff < min_diff:
                min_diff = aspect_diff
                best_width = w
                best_height = h
        
        if best_width == 0 or best_height == 0:
            approximate_size = int(target_area ** 0.5)
            best_width = approximate_size
            best_height = approximate_size
        
        if original_aspect > (best_width / best_height):
            new_width = int(height * (best_width / best_height))
            left = (width - new_width) // 2
            right = left + new_width
            image = image.crop((left, 0, right, height))
        else:
            new_height = int(width * (best_height / best_width))
            top = (height - new_height) // 2
            bottom = top + new_height
            image = image.crop((0, top, width, bottom))

        image = image.resize((best_width, best_height), Image.LANCZOS)
        return image

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)

        frames = []
        for frame_id in range(reader.count_frames()):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            if frame_id == 0:
                print("origin:", frame.size)
            
            frame = self.get_height_width(frame)
            if frame.size[0] / frame.size[1] > 16/9 or frame.size[0] / frame.size[1] < 9/16:
                assert False, f"The aspect ratio of the video does not meet the requirements.: {frame.size[0] / frame.size[1]}，{file_path} "
            frames.append(frame)
        print("新的宽高", frames[0].size)
        reader.close()
        return frames

    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        # image = self.crop_and_resize(image, *self.get_height_width(image))
        image = self.get_height_width(image)
        return image
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    def apply_background_color(self, img):
        img2 = Image.new('RGB', size=(img.width, img.height), color=(255, 255, 255))
        img2.paste(img, (0, 0), mask=img)
        return img2

    def load_video_from_folder(self, folder_path):
        frame_paths = glob.glob(os.path.join(folder_path, "*.png"))
        frame_paths.sort()

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("RGBA")
            frame = self.apply_background_color(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        return frames
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = data[key]
                data[key] = self.load_data(path)

                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        new_vae_path=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, skip_download=True, redirect_common_files=False,
                                                     local_model_path="Wan-AI/",
                                                     tokenizer_config=ModelConfig(model_id="Wan2.1-T2V-1.3B", origin_file_pattern="google/*"), is_preparing=True)
        self.pipe.vae = MergedWanVideoVAE(new_vae_path).to(device=torch.device("cpu"), dtype=torch.bfloat16)

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
            
    def pad_rgb(self, video_fgr, video_pha, is_hard=False):
        self.torch_dtype = torch.bfloat16
        self.device = video_fgr.device
        frame_alpha = (video_pha + 1.) * 0.5
        if is_hard:
            frame_alpha = torch.where(frame_alpha > 0, 1, 0)
        
        fg = video_fgr * frame_alpha + torch.tensor([1, 1, 1], dtype=torch.float32).reshape(1, 3, 1, 1, 1).to(dtype=self.torch_dtype).to(self.device) * (1.0 - frame_alpha)
        return fg.to(self.device, dtype=self.torch_dtype)
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        print("data.keys:", data.keys())
        inputs_nega = {}
        
        if type(data["video_fgr"])==list:
            # CFG-unsensitive parameters
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video_fgr": data["video_fgr"],
                "input_video_pha": data["video_pha"],
                "height": data["video_fgr"][0].size[1],
                "width": data["video_fgr"][0].size[0],
                "num_frames": len(data["video_fgr"]),
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
            }
        else:
            data["video_fgr"] = [data["video_fgr"]]
            data["video_pha"] = [data["video_pha"]]
            # CFG-unsensitive parameters
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video_fgr": data["video_fgr"],
                "input_video_pha": data["video_pha"],
                "height": data["video_fgr"][0].size[1],
                "width": data["video_fgr"][0].size[0],
                "num_frames": 1,
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
            }
        
        # Extra inputs
        # assert len(self.extra_inputs) == 0
        for extra_input in self.extra_inputs:
            fgr_tensor = TF.to_tensor(data["video_fgr"][0])
            pha_tensor = TF.to_tensor(data["video_pha"][0])
            padded_tensor = self.pad_rgb(fgr_tensor.float(), pha_tensor.float(),is_hard=False)
            input_0 = TF.to_pil_image(padded_tensor[0][0].float())
            end_1 = TF.to_pil_image(padded_tensor[0][-1].float())
            if extra_input == "input_image":
                inputs_shared["input_image"] = input_0
            elif extra_input == "end_image":
                inputs_shared["end_image"] = end_1
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, count, cache_path,inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)

        all_data = {}
        for key_name in ["context", "input_latents", "vace_context"]:
            if key_name in inputs and inputs[key_name] is not None:
                all_data[key_name] = inputs[key_name].cpu()
        all_data["gauss_mask"] = inputs["gauss_mask"]
        
        assert all_data["input_latents"].shape[2] == (len(data["video_fgr"]) - 1) // 4 + 1
        data["cache_path"] = data["cache_path"]+"_"+args.job_name
        if not os.path.exists(os.path.dirname(data["cache_path"])):
            print("cache_path:",os.path.dirname(data["cache_path"]))
            os.makedirs(os.path.dirname(data["cache_path"]))
        torch.save(all_data, data["cache_path"])
        


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        new_vae_path=args.new_vae_path,
    )
    job_id = args.job_id
    job_num = args.job_num
    dataset.data = dataset.data[job_id::job_num]  
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=1)
    model, dataloader = accelerator.prepare(model, dataloader)

    count = 0
    cache_path = []
    for data in tqdm(dataloader):
        count += 1
        with accelerator.accumulate(model):
            inputs = model(data, count, cache_path)
