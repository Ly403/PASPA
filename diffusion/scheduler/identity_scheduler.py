
import torch
from typing import Optional, Tuple, Union, List
from diffusers import SchedulerMixin, ConfigMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from diffusers.configuration_utils import register_to_config
import numpy as np



@dataclass
class IdentitySchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor
    

class IdentityScheduler(SchedulerMixin, ConfigMixin):
    """
    Identity scheduler.
    """
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        constant_timestep: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        timesteps = np.array([constant_timestep] * num_train_timesteps, dtype=np.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self.timesteps = timesteps
        self.constant_timestep = constant_timestep
        

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index


    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        *args,
        **kwargs
    ):  
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.array([self.constant_timestep] * self.num_inference_steps, dtype=np.float32)
        self.timesteps = torch.from_numpy(self.timesteps).to(dtype=torch.float32).to(device)
        self._step_index = None
        self._begin_index = None


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
        *args,
        **kwargs
    ) -> Union[IdentitySchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        prev_sample = sample
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)

        return IdentitySchedulerOutput(prev_sample=prev_sample)



