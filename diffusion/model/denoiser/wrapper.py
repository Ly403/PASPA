import torch
import torch.nn as nn
from diffusion.model.nets.basic_modules import MPFourierTimeEmbedding


class DenoiserWrapper(nn.Module):
    """Normalizes integer timesteps to c_noise in [0, 1] and feeds it to the
    inner model. Optionally predicts a per-frame logvar from a Fourier
    embedding of c_noise (used by EDM-style uncertainty-weighted losses)."""

    def __init__(
        self,
        model,
        training_step: int = 1000,
        return_logvar: bool = False,
        logvar_channels: int = 128,
    ):
        super().__init__()
        self.model = model
        self.training_step = training_step
        self.return_logvar = return_logvar
        if return_logvar:
            self.logvar_embed = MPFourierTimeEmbedding(logvar_channels, reduce_channels=0)
            self.logvar_pred = nn.Linear(logvar_channels, 1)
            nn.init.constant_(self.logvar_pred.weight, 0.0)
            if self.logvar_pred.bias is not None:
                nn.init.constant_(self.logvar_pred.bias, 0.0)

    def forward(self, motion, timestep, **kwargs):
        c_noise = timestep / self.training_step
        model_output = self.model(motion, c_noise, **kwargs)
        motion_output = model_output.pop("motion")
        if self.return_logvar:
            if c_noise.dim() == 1:
                c_noise = c_noise.unsqueeze(1)
            logvar_embed = self.logvar_embed(c_noise)
            model_output["logvar"] = self.logvar_pred(logvar_embed).reshape(
                c_noise.shape[0], c_noise.shape[1], 1
            )
        return {"motion": motion_output, **model_output}
