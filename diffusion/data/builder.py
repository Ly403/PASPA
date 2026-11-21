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

import os
import time
from termcolor import colored
from torch.utils.data import DataLoader

from diffusion.utils.logger import get_root_logger
from diffusion.utils.config import instantiate_from_config


def build_dataset(cfg, split: str):
    logger = get_root_logger()

    dataset_type = cfg["target"]
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    dataset = instantiate_from_config(cfg)
    logger.info(
        f"{colored(f'{split.capitalize()} Dataset {dataset_type} constructed: ', 'green', attrs=['bold'])}"
        f"time: {(time.time() - t):.2f} s, length: {len(dataset)}"
    )
    return dataset

def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if "batch_sampler" in kwargs:
        batch_sampler = kwargs.pop("batch_sampler")
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, **kwargs
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, **kwargs
        )
    return dataloader
