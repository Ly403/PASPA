# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import AutoencoderDC
from diffusers.models import AutoencoderKL
from termcolor import colored
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    T5EncoderModel, 
    T5Tokenizer,
    CLIPTokenizer,
    CLIPTextModelWithProjection
)
from transformers import logging as transformers_logging

from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint
from diffusion.utils.config import instantiate_from_config
transformers_logging.set_verbosity_error()


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    # model = MODELS.build(cfg, default_args=kwargs)
    model = instantiate_from_config(cfg)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(
    name="T5", 
    device="cuda", 
    dtype=torch.float16,
    local_model_path=None,
    projection_dim=768
):
    text_encoder_dict = {
        "clip": "openai/clip-vit-base-patch32",
        "clip-L-14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "clip-G-14": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "clip-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "clip-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    if "T5" in name:
        if local_model_path is not None:
            tokenizer = T5Tokenizer.from_pretrained(local_model_path)
            text_encoder = T5EncoderModel.from_pretrained(local_model_path, torch_dtype=dtype).to(device)
        else:
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_dict[name])
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_dict[name], torch_dtype=dtype).to(device).eval()
    elif "clip" in name:
        if local_model_path is not None:
            tokenizer = CLIPTokenizer.from_pretrained(local_model_path)
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                local_model_path, torch_dtype=dtype,
                projection_dim=projection_dim
            ).to(device).eval()
        else:
            tokenizer = CLIPTokenizer.from_pretrained(text_encoder_dict[name])
            text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_dict[name], torch_dtype=dtype,
                projection_dim=projection_dim).to(device).eval()
    elif "gemma" in name or "Qwen" in name:
        if local_model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            tokenizer.padding_side = "right"
            text_encoder = (
                AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=dtype)
                .get_decoder()
                .to(device)
            ).eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_dict[name])
            tokenizer.padding_side = "right"
            text_encoder = (
                AutoModelForCausalLM.from_pretrained(text_encoder_dict[name], torch_dtype=dtype)
                .get_decoder()
                .to(device)
            ).eval()
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder

