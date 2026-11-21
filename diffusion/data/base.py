from typing import Dict, Optional, Union
from omegaconf import DictConfig
from diffusion.utils.config import instantiate_from_config, set_config_value
from diffusion.data.builder import build_dataloader, build_dataset


class DataModule:
    def __init__(
        self,
        config,
        train_dataset_config: Optional[Dict | DictConfig] = None,
        validation_dataset_config: Optional[Dict | DictConfig] = None,
        test_dataset_config: Optional[Dict | DictConfig] = None,
        predict_dataset_config: Optional[Dict | DictConfig] = None,
    ):
        self.train_dataset = build_dataset(
            train_dataset_config, "train"
        ) if train_dataset_config is not None else None
        self.validation_dataset = build_dataset(
            validation_dataset_config, "validation"
        ) if validation_dataset_config is not None else None
        self.test_dataset = build_dataset(
            test_dataset_config, "test"
        ) if test_dataset_config is not None else None
        self.predict_dataset = build_dataset(
            predict_dataset_config, "predict"
        ) if predict_dataset_config is not None else None
        self.config = config
        
    def prepare_data(
        self, 
        train_sampler_config: Optional[Dict | DictConfig] = None,
        validation_sampler_config: Optional[Dict | DictConfig] = None,
        test_sampler_config: Optional[Dict | DictConfig] = None,
        predict_sampler_config: Optional[Dict | DictConfig] = None,
    ):
        
        if self.train_dataset is not None and train_sampler_config is not None:
            self.train_sampler = instantiate_from_config({
                "target": train_sampler_config.target,
                "params": {
                    "dataset": self.train_dataset,
                    **train_sampler_config.params
                }
            })
        else:
            self.train_sampler = None
        
        if self.validation_dataset is not None and validation_sampler_config is not None:
            self.validation_sampler = instantiate_from_config({
                "target": validation_sampler_config.target,
                "params": {
                    "dataset": self.validation_dataset,
                    **validation_sampler_config.params
                }
            })
        else:
            self.validation_sampler = None
            
        if self.test_dataset is not None and test_sampler_config is not None:
            self.test_sampler = instantiate_from_config({
                "target": test_sampler_config.target,
                "params": {
                    "dataset": self.test_dataset,
                    **test_sampler_config.params
                }
            })
        else:
            self.test_sampler = None
            
        if self.predict_dataset is not None and predict_sampler_config is not None:
            self.predict_sampler = instantiate_from_config({
                "target": predict_sampler_config.target,
                "params": {
                    "dataset": self.predict_dataset,
                    **predict_sampler_config.params
                }
            })
        else:
            self.predict_sampler = None

        num_workers = int(self.config.data.num_workers)
        if num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        train_worker_kwargs = {}
        if num_workers > 0:
            train_worker_kwargs = {
                "persistent_workers": True,
                "prefetch_factor": 4,
            }

        if self.train_dataset is not None:
            self.train_dataloader = build_dataloader(
                self.train_dataset,
                num_workers=num_workers,
                batch_size=self.config.data.train_batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                **train_worker_kwargs,
            )
        else:
            self.train_dataloader = None
        
        if self.validation_dataset is not None:
            self.validation_dataloader = build_dataloader(
                self.validation_dataset,
                num_workers=num_workers,
                batch_size=self.config.data.validation_batch_size,
                shuffle=False,
                sampler=self.validation_sampler,
            )
        else:
            self.validation_dataloader = None
        
        if self.test_dataset is not None:
            self.test_dataloader = build_dataloader(
                self.test_dataset,
                num_workers=num_workers,
                batch_size=self.config.data.test_batch_size,
                shuffle=False,
                sampler=self.test_sampler,
            )
        else:
            self.test_dataloader = None
        
        if self.predict_dataset is not None:
            self.predict_dataloader = build_dataloader(
                self.predict_dataset,
                num_workers=num_workers,
                batch_size=self.config.data.predict_batch_size,
                shuffle=False,
                sampler=self.predict_sampler,
            )
        else:
            self.predict_dataloader = None
    
    @property
    def _train_dataset(self):
        return self.train_dataset
    
    @property
    def _validation_dataset(self):
        return self.validation_dataset
    
    @property
    def _test_dataset(self):
        return self.test_dataset
    
    @property
    def _train_dataloader(self):
        return self.train_dataloader
    
    @property
    def _validation_dataloader(self):
        return self.validation_dataloader
    
    @property
    def _test_dataloader(self):
        return self.test_dataloader
    
    @property
    def _predict_dataloader(self):
        return self.predict_dataloader
