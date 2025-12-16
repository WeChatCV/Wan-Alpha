<div align="center">

  <h1>
    Wan-Alpha
  </h1>

  <h3>Video Generation with Stable Transparency via Shiftable RGB-A Distribution Learner</h3>



[![arXiv](https://img.shields.io/badge/arXiv-2509.24979-red)](https://arxiv.org/pdf/2509.24979)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://donghaotian123.github.io/Wan-Alpha/)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model_v1.0-orange)](https://huggingface.co/htdong/Wan-Alpha)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Model_v1.0-blue)](https://huggingface.co/htdong/Wan-Alpha_ComfyUI)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model_v2.0-yellow)](https://huggingface.co/htdong/Wan-Alpha-v2.0)


</div>

<img src="assets/teaser.png" alt="Wan-Alpha Qualitative Results" style="max-width: 100%; height: auto;">

>Qualitative results of video generation using **Wan-Alpha-v2.0**. Our model successfully generates various scenes with accurate and clearly rendered transparency. Notably, it can synthesize diverse semi-transparent objects, glowing effects, and fine-grained details such as hair.

---

### 🔥 News

* **[2025.12.16]** Released Wan-Alpha v2.0, the Wan2.1-14B-T2V–adapted weights and inference code are now open-sourced.
* **[2025.12.16]** We update our paper on [arXiv](https://arxiv.org/pdf/2509.24979).
* **[2025.09.30]** Our technical report is available on [arXiv](https://arxiv.org/pdf/2509.24979).
* **[2025.09.30]** Released Wan-Alpha v1.0, the Wan2.1-14B-T2V–adapted weights and inference code are now open-sourced.

---

### 📝 To-Do List

- [x] **Paper**: Available on [arXiv](https://arxiv.org/pdf/2509.24979).
- [x] **Inference Code**: Released inference pipeline for Wan-Alpha v1.0 and v2.0.
- [x] **Model Weights**: Released checkpoints for Wan-Alpha v1.0 and v2.0.
- [ ] **Image-to-Video**: Release Wan-Alpha-I2V model weights.
- [ ] **Dataset**: Open-source the VAE and T2V training dataset.
- [ ] **Training Code (VAE&T2V)**: Release training scripts for the VAE and text-to-RGBA video generation.


### 🌟 Showcase

##### Text-to-Video Generation with Alpha Channel

<!-- | Prompt | Preview Video | Alpha Video |
| :---: | :---: | :---: |
| "Medium shot. A little girl holds a bubble wand and blows out colorful bubbles that float and pop in the air. The background of this video is transparent. Realistic style." |
  <div style="display: flex; gap: 10px;">
    <img src="girl.gif" alt="..." style="flex: 1; min-width: 200px;">
  </div> |
  <div style="display: flex; gap: 10px;">
    <img src="girl_pha.gif" alt="..." style="flex: 1; min-width: 200px;">
  </div> | -->
| Prompt | Preview Video | Alpha Video |
| :---: | :---: | :---: |
| "The background of this video is transparent. It features a beige, woven rattan hanging chair with soft seat and back cushions. Realistic style. Medium shot." | <img src="assets/squirrel.gif" width="320" height="180" style="object-fit:contain; display:block; margin:auto;"/> | <img src="assets/squirrel_pha.gif" width="320" height="180" style="object-fit:contain; display:block; margin:auto;"/> |

##### For more results, please visit [Our Website](https://donghaotian123.github.io/Wan-Alpha/)

### 🚀 Quick Start

##### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/WeChatCV/Wan-Alpha.git
cd Wan-Alpha

# Create and activate Conda environment
conda create -n Wan-Alpha python=3.11 -y
conda activate Wan-Alpha

# Install dependencies
pip install -r requirements.txt
```

##### 2. Model Download
Download [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)

Download [Lightx2v-T2V-14B](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors)

Download [Wan-Alpha](https://huggingface.co/htdong/Wan-Alpha-v2.0)

### 🧪 Usage
You can test our model through:
```bash
torchrun --nproc_per_node=8 --master_port=29501 generate_dora_lightx2v_mask.py --size 832*480\
         --ckpt_dir "path/to/your/Wan-2.1/Wan2.1-T2V-14B" \
         --dit_fsdp --t5_fsdp --ulysses_size 8 \
         --vae_lora_checkpoint "path/to/your/decoder.bin" \
         --lora_path "path/to/your/t2v.safetensors" \
         --lightx2v_path "path/to/your/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
         --sample_guide_scale 1.0 \
         --frame_num 81 \
         --sample_steps 4 \
         --lora_ratio 1.0 \
         --lora_prefix "" \
         --alpha_shift_mean 0.05 \
         --cache_path_mask "path/to/your/gauss_mask" \
         --prompt_file ./data/prompt.txt \
         --output_dir ./output 
```
You can specify the weights of `Wan2.1-T2V-14B` with `--ckpt_dir`, `LightX2V-T2V-14B` with `--lightx2v_path`, `Wan-Alpha-VAE` with `--vae_lora_checkpoint`, and `Wan-Alpha-T2V` with `--lora_path`. Finally, you can find the rendered RGBA videos with a checkerboard background and PNG frames at `--output_dir`.

You can use `gen_gaussian_mask.py` to generate a Gaussian mask from an existing alpha video. Alternatively, you can directly create a Gaussian ellipse video, which can be either static or dynamic (e.g., moving from left to right). Note that alpha_shift_mean is a fixed parameter.

**Prompt Writing Tip:**  You need to specify that the background of the video is transparent, the visual style, the shot type (such as close-up, medium shot, wide shot, or extreme close-up), and a description of the main subject. Prompts support both Chinese and English input.

```bash
# An example of prompt.
This video has a transparent background. Close-up shot. A colorful parrot flying. Realistic style.
```

### 🔨 Official ComfyUI Version

Coming soon...


### 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)
* [LightX2V](https://github.com/ModelTC/LightX2V) (inference acceleration)
* [WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) (inference acceleration)

We sincerely thank the authors and contributors of these projects.


### ✏ Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@misc{dong2025wanalpha,
      title={Video Generation with Stable Transparency via Shiftable RGB-A Distribution Learner}, 
      author={Haotian Dong and Wenjing Wang and Chen Li and Jing Lyu and Di Lin},
      year={2025},
      eprint={2509.24979},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24979}, 
}
``` 

### 📬 Contact Us

If you have any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/WeChatCV/Wan-Alpha/issues) . We look forward to your feedback!