
import torch
from typing import Optional, Tuple, Union
from diffusers import FlowMatchEulerDiscreteScheduler as Scheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput


class FlowMatchEulerDiscreteScheduler(Scheduler):
    def __init__(
        self, 
        prediction_type: str = "velocity",
        *args, 
        **kwargs
    ):
        assert prediction_type in ["velocity", "epsilon", "xstart"], "prediction_type must be either 'velocity' or 'epsilon' or 'xstart'"
        self.prediction_type = prediction_type
        super().__init__(*args, **kwargs)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noise = None,
        current_sigma_override: Optional[float] = None,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

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

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            # start_sigma path overrides current_sigma with a continuous value
            # (used when starting from a non-discrete-grid noise level).
            if current_sigma_override is not None:
                current_sigma = current_sigma_override
            else:
                current_sigma = sigma
            next_sigma = sigma_next
            dt = next_sigma - current_sigma
        
        if self.prediction_type == "velocity":
            if self.config.stochastic_sampling:
                x0 = sample - current_sigma * model_output
                noise = torch.randn_like(sample)
                prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                prev_sample = sample + dt * model_output
        
        elif self.prediction_type == "xstart":
            if self.config.stochastic_sampling:
                x0 = model_output
                noise = torch.randn_like(sample)
                prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                if noise is None:
                    prev_sample = next_sigma * sample / current_sigma + model_output * (1 - next_sigma / current_sigma)
                else:
                    prev_sample = (1.0 - next_sigma) * model_output + next_sigma * noise
                # prev_sample = next_sigma * sample / current_sigma + model_output * (1 - next_sigma / current_sigma)
                
        elif self.prediction_type == "epsilon":
            if self.config.stochastic_sampling:
                x0 = sample / (1 - current_sigma) - (current_sigma / (1 - current_sigma)) * model_output
                noise = torch.randn_like(sample)
                prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                prev_sample = (1 - next_sigma) / (1 - current_sigma) * sample \
                    + (1 - (1 - next_sigma) / (1 - current_sigma)) * model_output   
            
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)