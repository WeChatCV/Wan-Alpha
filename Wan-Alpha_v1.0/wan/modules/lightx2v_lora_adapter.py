import gc
import os

import torch
import logging
from safetensors import safe_open

class WanLoraWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        self.override_dict = {}  # On CPU

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_metadata:
            return lora_name

        self.lora_metadata[lora_name] = {"path": lora_path}
        return lora_name

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(self.model.dtype) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        assert lora_name in self.lora_metadata
        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.state_dict()

        self._apply_lora_weights(weight_dict, lora_weights, alpha)
        self.model.load_state_dict(weight_dict)

        del lora_weights

    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_pairs = {}
        lora_diffs = {}

        def try_lora_pair(key, prefix, suffix_a, suffix_b, target_suffix):
            if key.endswith(suffix_a):
                base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
                pair_key = key.replace(suffix_a, suffix_b)
                if pair_key in lora_weights:
                    lora_pairs[base_name] = (key, pair_key)

        def try_lora_diff(key, prefix, suffix, target_suffix):
            if key.endswith(suffix):
                base_name = key[len(prefix) :].replace(suffix, target_suffix)
                lora_diffs[base_name] = key

        prefixs = [
            "",  # empty prefix
            "diffusion_model.",
        ]
        for prefix in prefixs:
            for key in lora_weights.keys():
                if not key.startswith(prefix):
                    continue

                try_lora_pair(key, prefix, "lora_A.weight", "lora_B.weight", "weight")
                try_lora_pair(key, prefix, "lora_down.weight", "lora_up.weight", "weight")
                try_lora_diff(key, prefix, "diff", "weight")
                try_lora_diff(key, prefix, "diff_b", "bias")
                try_lora_diff(key, prefix, "diff_m", "modulation")

        applied_count = 0
        for name, param in weight_dict.items():
            if name in lora_pairs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()
                name_lora_A, name_lora_B = lora_pairs[name]
                lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
                lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
                if param.shape == (lora_B.shape[0], lora_A.shape[1]):
                    param += torch.matmul(lora_B, lora_A) * alpha
                    applied_count += 1
            elif name in lora_diffs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()

                name_diff = lora_diffs[name]
                lora_diff = lora_weights[name_diff].to(param.device, param.dtype)
                if param.shape == lora_diff.shape:
                    param += lora_diff * alpha
                    applied_count += 1

        assert applied_count > 0
        logging.info(f"applied_count in LightX2V: {applied_count}")
