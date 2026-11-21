import torch
import numpy as np
from diffusion.model.utils import expand_dims_like
from diffusion.guider.cfg_schedule import BaseCFGSchedule, ConstantCFGSchedule, build_cfg_schedule
from diffusion.utils import rotation_conversions as rc

class IdentityGuider:
    def __init__(self):
        pass
    
    def __call__(
        self, 
        motion, 
        expression, 
        audio, 
        motion_timestep, 
        expression_timestep, 
        audio_timestep, 
        model_kwargs, 
        is_sample: bool = False, 
        task="text2all"
    ):
        return (
            motion, 
            expression, 
            audio, 
            motion_timestep, 
            expression_timestep, 
            audio_timestep, 
            model_kwargs
        )
    
    def guide(
        self, 
        motion, 
        expression, 
        audio, 
    ):
        return motion, expression, audio

class VanillaOmniFlowCFGuider:
    def __init__(
        self,
        ucg_prob: float = 0.1,
        ucg_motion_prob = None,
        ucg_expression_prob = None,
        ucg_audio_prob = None,
        cfg_motion_scale: float = 4.5,
        cfg_expression_scale: float = 4.5,
        cfg_audio_scale: float = 4.5,
        max_classes=11195,
        max_training_steps=1000,
    ):
        self.ucg_prob = ucg_prob
        self.ucg_motion_prob = ucg_motion_prob if ucg_motion_prob is not None else ucg_prob
        self.ucg_expression_prob = ucg_expression_prob if ucg_expression_prob is not None else ucg_prob
        self.ucg_audio_prob = ucg_audio_prob if ucg_audio_prob is not None else ucg_prob
        self.cfg_motion_scale = cfg_motion_scale
        self.cfg_expression_scale = cfg_expression_scale
        self.cfg_audio_scale = cfg_audio_scale
        self.max_classes = max_classes
        self.max_training_steps_index = max_training_steps - 1
    
    def __call__(
        self, 
        motion, 
        expression, 
        audio, 
        motion_timestep, 
        expression_timestep, 
        audio_timestep, 
        model_kwargs,
        is_sample: bool = False,
        task="text2all",
    ):
        assert task in [
            "text2all", 
            "text2motion", "text2expression", "text2audio"
            "text2motion+expression", "text2motion+audio", "text2expression+audio",
            "text+audio2motion", "text+audio2expression","text+audio2motion+expression",
            "text+motion2expression", "text+motion2audio","text+motion2expression+audio",
            "text+expression2motion", "text+expression2audio","text+expression2motion+audio",
            "text+motion+expression2audio", "text+motion+audio2expression", "text+expression+audio2motion",
            ################################################################################################
            "uncondition2all",
            "uncondition2motion", "uncondition2expression", "uncondition2audio"
            "uncondition2motion+expression", "uncondition2motion+audio", "uncondition2expression+audio",
            "audio2motion", "audio2expression","audio2motion+expression",
            "motion2expression", "motion2audio","motion2expression+audio",
            "expression2motion", "expression2audio","expression2motion+audio",
            "motion+expression2audio", "motion+audio2expression", "expression+audio2motion",
        ]
        if is_sample:
            return self.sample_guiding(
                motion, expression, audio, motion_timestep, expression_timestep, audio_timestep, model_kwargs, task)
        else:
            return self.training_drop(
                motion, expression, audio, motion_timestep, expression_timestep, audio_timestep, model_kwargs, task)
    
    def text_drop(
        self,
        emb,
    ):
        # 只drop text标签即可
        if self.ucg_prob > 0.0:
            drop_ids = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_prob)
                        * torch.ones(emb.shape[0], device=emb.device)
                    ),
                    emb,
                )
            )
            if not torch.is_floating_point(emb):
                emb = torch.where(drop_ids.bool(), torch.randint_like(emb, self.max_classes), emb)
            else:
                emb = torch.where(drop_ids.bool(), torch.randn_like(emb), emb)
        return emb
    
    def motion_drop(
        self,
        motion,
        motion_timestep,
    ):
        if self.ucg_motion_prob > 0.0:
            drop_ids = torch.bernoulli(
                (1.0 - self.ucg_motion_prob)
                * torch.ones(motion.shape[0], device=motion.device)
            )
            drop_ids_bc = expand_dims_like(drop_ids, motion)
            motion = torch.where(drop_ids_bc.bool(), torch.randn_like(motion), motion)
            motion_timestep = torch.where(
                drop_ids.bool(), torch.ones_like(motion_timestep) * self.max_training_steps_index, motion_timestep)
        return motion, motion_timestep
    
    def expression_drop(
        self,
        expression,
        expression_timestep,
    ):
        if self.ucg_expression_prob > 0.0:
            drop_ids = torch.bernoulli(
                (1.0 - self.ucg_expression_prob)
                * torch.ones(expression.shape[0], device=expression.device)
            )
            drop_ids_bc = expand_dims_like(drop_ids, expression)
            expression = torch.where(drop_ids_bc.bool(), torch.randn_like(expression), expression)
            expression_timestep = torch.where(
                drop_ids.bool(), torch.ones_like(expression_timestep) * self.max_training_steps_index, expression_timestep)
        return expression, expression_timestep
    
    def audio_drop(
        self,
        audio,
        audio_timestep,
    ):
        if self.ucg_audio_prob > 0.0:
            drop_ids = torch.bernoulli(
                (1.0 - self.ucg_audio_prob)
                * torch.ones(audio.shape[0], device=audio.device)
            )
            drop_ids_bc = expand_dims_like(drop_ids, audio)
            audio = torch.where(drop_ids_bc.bool(), torch.randn_like(audio), audio)
            audio_timestep = torch.where(
                drop_ids.bool(), torch.ones_like(audio_timestep) * self.max_training_steps_index, audio_timestep)
        return audio, audio_timestep
    
    def training_drop(
        self, 
        motion, 
        expression, 
        audio, 
        motion_timestep, 
        expression_timestep, 
        audio_timestep, 
        model_kwargs,
        task="text2all",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
        elif task.startswith("motion2"):
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
        elif task.startswith("expression2"):
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
        elif task.startswith("audio2"):
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        
        elif task.startswith("text+motion2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
        elif task.startswith("text+expression2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
        elif task.startswith("text+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        
        elif task.startswith("motion+expression2"):
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
        elif task.startswith("motion+audio2"):
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        elif task.startswith("expression+audio2"):
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        
        elif task.startswith("text+motion+expression2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
        elif task.startswith("text+motion+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            motion, motion_timestep = self.motion_drop(motion, motion_timestep)
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        elif task.startswith("text+expression+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.text_drop(model_kwargs["txt"])
            expression, expression_timestep = self.expression_drop(expression, expression_timestep)
            audio, audio_timestep = self.audio_drop(audio, audio_timestep)
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "expression": expression, 
            "audio": audio, 
            "timestep_motion": motion_timestep, 
            "timestep_expression": expression_timestep, 
            "timestep_audio": audio_timestep, 
            **model_kwargs
        }
    
    def sample_guiding(self, 
        motion, 
        expression, 
        audio, 
        motion_timestep, 
        expression_timestep, 
        audio_timestep, 
        model_kwargs,
        task="text2all",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            motion = torch.cat([motion] * 2)
            expression = torch.cat([expression] * 2)
            audio = torch.cat([audio] * 2)
            motion_timestep = torch.cat([motion_timestep] * 2)
            expression_timestep = torch.cat([expression_timestep] * 2)
            audio_timestep = torch.cat([audio_timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            kwargs["txt"] = torch.cat([emb, un_emb])
        elif task.startswith("motion2"):
            un_motion = model_kwargs.get("uncond_motion", None)
            if un_motion is None:
                un_motion = torch.zeros_like(motion)
            expression = torch.cat([expression] * 2)
            audio = torch.cat([audio] * 2)
            expression_timestep = torch.cat([expression_timestep] * 2)
            audio_timestep = torch.cat([audio_timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            motion = torch.cat([motion, un_motion])
            motion_timestep = torch.cat([motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index])
        elif task.startswith("expression2"):
            un_expression = model_kwargs.get("uncond_expression", None)
            if un_expression is None:
                un_expression = torch.zeros_like(expression)
            motion = torch.cat([motion] * 2)
            audio = torch.cat([audio] * 2)
            motion_timestep = torch.cat([motion_timestep] * 2)
            audio_timestep = torch.cat([audio_timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            expression = torch.cat([expression, un_expression])
            expression_timestep = torch.cat([expression_timestep, torch.ones_like(expression_timestep) * self.max_training_steps_index])
        elif task.startswith("audio2"):
            un_audio = model_kwargs.get("uncond_audio", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            motion = torch.cat([motion] * 2)
            expression = torch.cat([expression] * 2)
            motion_timestep = torch.cat([motion_timestep] * 2)
            expression_timestep = torch.cat([expression_timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            audio = torch.cat([audio, un_audio])
            audio_timestep = torch.cat([audio_timestep, torch.ones_like(audio_timestep) * self.max_training_steps_index])

        elif task.startswith("text+motion2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            un_motion = model_kwargs.get("uncond_motion", None)
            if un_motion is None:
                un_motion = torch.zeros_like(motion)
            expression = torch.cat([expression] * 4)
            audio = torch.cat([audio] * 4)
            expression_timestep = torch.cat([expression_timestep] * 4)
            audio_timestep = torch.cat([audio_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            kwargs["txt"] = torch.cat([emb, un_emb, emb, un_emb])
            motion = torch.cat([motion, un_motion, un_motion, motion])
            motion_timestep = torch.cat([
                motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index, 
                torch.ones_like(motion_timestep) * self.max_training_steps_index, motion_timestep
            ])
        elif task.startswith("text+expression2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            un_expression = model_kwargs.get("uncond_expression", None)
            if un_expression is None:
                un_expression = torch.zeros_like(expression)
            motion = torch.cat([motion] * 4)
            audio = torch.cat([audio] * 4)
            motion_timestep = torch.cat([motion_timestep] * 4)
            audio_timestep = torch.cat([audio_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            kwargs["txt"] = torch.cat([emb, un_emb, emb, un_emb])
            expression = torch.cat([expression, un_expression, un_expression, expression])
            expression_timestep = torch.cat([
                expression_timestep, torch.ones_like(expression_timestep) * self.max_training_steps_index, 
                torch.ones_like(expression_timestep) * self.max_training_steps_index, expression_timestep
            ])
        elif task.startswith("text+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            un_audio = model_kwargs.get("uncond_audio", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            motion = torch.cat([motion] * 4)
            expression = torch.cat([expression] * 4)
            motion_timestep = torch.cat([motion_timestep] * 4)
            expression_timestep = torch.cat([expression_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            kwargs["txt"] = torch.cat([emb, un_emb, emb, un_emb])
            audio = torch.cat([audio, un_audio, un_audio, audio])
            audio_timestep = torch.cat([
                audio_timestep, torch.ones_like(audio_timestep) * self.max_training_steps_index, 
                torch.ones_like(audio_timestep) * self.max_training_steps_index, audio_timestep
            ])
        
        elif task.startswith("motion+expression2"):
            un_motion = model_kwargs.get("uncond_motion", None)
            if un_motion is None:
                un_motion = torch.zeros_like(motion)
            un_expression = model_kwargs.get("uncond_expression", None)
            if un_expression is None:
                un_expression = torch.zeros_like(expression)
            audio = torch.cat([audio] * 4)
            audio_timestep = torch.cat([audio_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            motion = torch.cat([motion, un_motion, motion, un_motion])
            motion_timestep = torch.cat([
                motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index, 
                motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index
            ])
            expression = torch.cat([expression, un_expression, un_expression, expression])
            expression_timestep = torch.cat([
                expression_timestep, torch.ones_like(expression_timestep) * self.max_training_steps_index, 
                torch.ones_like(expression_timestep) * self.max_training_steps_index, expression_timestep
            ])
        elif task.startswith("motion+audio2"):
            un_motion = model_kwargs.get("uncond_motion", None)
            if un_motion is None:
                un_motion = torch.zeros_like(motion)
            un_audio = model_kwargs.get("uncond_audio", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            expression = torch.cat([expression] * 4)
            expression_timestep = torch.cat([expression_timestep] * 4)
            audio = torch.cat([audio] * 4)
            audio_timestep = torch.cat([audio_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            motion = torch.cat([motion, un_motion, motion, un_motion])
            motion_timestep = torch.cat([
                motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index, 
                motion_timestep, torch.ones_like(motion_timestep) * self.max_training_steps_index
            ])
            audio = torch.cat([audio, un_audio, un_audio, audio])
            audio_timestep = torch.cat([
                audio_timestep, torch.ones_like(audio_timestep) * self.max_training_steps_index, 
                torch.ones_like(audio_timestep) * self.max_training_steps_index, audio_timestep
            ])
        elif task.startswith("expression+audio2"):
            un_expression = model_kwargs.get("uncond_expression", None)
            if un_expression is None:
                un_expression = torch.zeros_like(expression)
            audio = torch.cat([audio] * 4)
            audio_timestep = torch.cat([audio_timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            expression = torch.cat([expression, un_expression, expression, un_expression])
            expression_timestep = torch.cat([
                expression_timestep, torch.ones_like(expression_timestep) * self.max_training_steps_index, 
                expression_timestep, torch.ones_like(expression_timestep) * self.max_training_steps_index
            ])
            audio = torch.cat([audio, un_audio, un_audio, audio])
            audio_timestep = torch.cat([
                audio_timestep, torch.ones_like(audio_timestep) * self.max_training_steps_index, 
                torch.ones_like(audio_timestep) * self.max_training_steps_index, audio_timestep
            ])
        
        elif task.startswith("text+motion+expression2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("text+motion+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("text+expression+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "expression": expression, 
            "audio": audio, 
            "timestep_motion": motion_timestep, 
            "timestep_expression": expression_timestep, 
            "timestep_audio": audio_timestep, 
            **kwargs
        }
    
    def guide(
        self, 
        motion, 
        expression, 
        audio,
        task="text2all",
    ):
        if task.startswith("uncondition2"):
            pass
        
        elif task.startswith("text2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            audio, uncond_audio = audio.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
            expression_pred = uncond_expression + self.cfg_expression_scale * (expression - uncond_expression)
            audio_pred = uncond_audio + self.cfg_audio_scale * (audio - uncond_audio)
        elif task.startswith("motion2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            audio, uncond_audio = audio.chunk(2, dim=0)
            motion_pred = motion # motion是作为条件的，不用生成motion，保留原始motion条件的状态不变就好了
            expression_pred = uncond_expression + self.cfg_expression_scale * (expression - uncond_expression)
            audio_pred = uncond_audio + self.cfg_audio_scale * (audio - uncond_audio)
        elif task.startswith("expression2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            audio, uncond_audio = audio.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
            expression_pred = expression # expression是作为条件的，不用生成expression，保留原始expression条件的状态不变就好了
            audio_pred = uncond_audio + self.cfg_audio_scale * (audio - uncond_audio)
        elif task.startswith("audio2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            audio, uncond_audio = audio.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
            expression_pred = uncond_expression + self.cfg_expression_scale * (expression - uncond_expression)
            audio_pred = audio # audio是作为条件的，不用生成audio，保留原始audio条件的状态不变就好了
        
        elif task.startswith("text+motion2"):
            motion, un_motion, _, _ = motion.chunk(4, dim=0)
            expression, un_expression_untext_unmotion, un_expression_text_unmotion, un_expression_untext_motion \
                = expression.chunk(4, dim=0)
            audio, un_audio_untext_unmotion, un_audio_text_unmotion, un_audio_untext_motion \
                = audio.chunk(4, dim=0)
            motion_pred = motion
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")    
        elif task.startswith("text+expression2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("text+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("motion+expression2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("motion+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("expression+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("text+motion+expression2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")  
        elif task.startswith("text+motion+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        elif task.startswith("text+expression+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion_pred,
            "expression": expression_pred,
            "audio": audio_pred
        }

class VanillaSimpleFlowCFGuiderSimple:
    def __init__(
        self,
        ucg_prob: float = 0.1,
        cfg_motion_scale: float = 4.5,
        cfg_expression_scale: float = 4.5,
        max_classes=11195,
        max_training_steps=1000,
        disable_expression_guidance=False,
    ):
        self.ucg_prob = ucg_prob
        self.cfg_motion_scale = cfg_motion_scale
        self.cfg_expression_scale = cfg_expression_scale
        self.max_classes = max_classes
        self.max_training_steps_index = max_training_steps - 1
        self.disable_expression_guidance = disable_expression_guidance
    
    def __call__(
        self, 
        motion, 
        expression,
        timestep,
        model_kwargs,
        is_sample: bool = False,
        task="audio2motion+expression",
    ):
        assert task in [
            "text2motion", "audio2motion", "uncondition2motion"
            "text2motion+expression", "audio2motion+expression", "uncondition2motion+expression"
        ]
        if is_sample:
            return self.sample_guiding(
                motion, expression, timestep, model_kwargs, task)
        else:
            return self.training_drop(
                motion, expression, timestep, model_kwargs, task)
    
    def cond_drop(
        self,
        emb,
    ):
        # 只drop text标签即可
        if self.ucg_prob > 0.0:
            drop_ids = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_prob)
                        * torch.ones(emb.shape[0], device=emb.device)
                    ),
                    emb,
                )
            )
            if not torch.is_floating_point(emb):
                emb = torch.where(drop_ids.bool(), torch.randint_like(emb, self.max_classes), emb)
            else:
                emb = torch.where(drop_ids.bool(), torch.randn_like(emb), emb)
        return emb
    
    
    def training_drop(
        self, 
        motion, 
        expression, 
        timestep,
        model_kwargs,
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            model_kwargs["txt"] = self.cond_drop(model_kwargs["txt"])
        elif task.startswith("audio2"):
            assert "audio" in model_kwargs, "audio is required"
            model_kwargs["audio"] = self.cond_drop(model_kwargs["audio"])
        elif task.startswith("text+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            assert "audio" in model_kwargs, "audio is required"
            model_kwargs["txt"] = self.cond_drop(model_kwargs["txt"])
            model_kwargs["audio"] = self.cond_drop(model_kwargs["audio"])
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "expression": expression, 
            "timestep": timestep,
            **model_kwargs
        }
    
    def sample_guiding(
        self, 
        motion, 
        expression,  
        timestep,
        model_kwargs,
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            motion = torch.cat([motion] * 2)
            expression = torch.cat([expression] * 2)
            timestep = torch.cat([timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            kwargs["txt"] = torch.cat([emb, un_emb])
        
        elif task.startswith("audio2"):
            audio = model_kwargs["audio"]
            un_audio = model_kwargs.get("uncond_audio", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            motion = torch.cat([motion] * 2)
            expression = torch.cat([expression] * 2)
            timestep = torch.cat([timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if key == "audio" or key == "uncond_audio":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            kwargs["audio"] = torch.cat([audio, un_audio])

        elif task.startswith("text+audio2"):
            assert "txt" in model_kwargs, "text embedding is required"
            emb = model_kwargs["txt"]
            un_emb = model_kwargs.get("uncond_txt", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            audio = model_kwargs["audio"]
            un_audio = model_kwargs.get("uncond_audio", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            motion = torch.cat([motion] * 4)
            expression = torch.cat([expression] * 4)
            timestep = torch.cat([timestep] * 4)
            kwargs = {}
            for key in model_kwargs:
                if key == "txt" or key == "uncond_txt":
                    continue
                if key == "audio" or key == "uncond_audio":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 4)
            kwargs["txt"] = torch.cat([emb, un_emb, emb, un_emb])
            kwargs["audio"] = torch.cat([audio, un_audio, un_audio, audio])
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "expression": expression, 
            "timestep": timestep,
            **kwargs
        }
    
    def guide(
        self, 
        motion, 
        expression, 
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        
        elif task.startswith("text2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
            if not self.disable_expression_guidance:
                expression_pred = uncond_expression + self.cfg_expression_scale * (expression - uncond_expression)
            else:
                expression_pred = expression # 保留cond下生成的expression

        elif task.startswith("audio2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            expression, uncond_expression = expression.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
            if not self.disable_expression_guidance:
                expression_pred = uncond_expression + self.cfg_expression_scale * (expression - uncond_expression)
            else:
                expression_pred = expression # 保留cond下生成的expression
        
        elif task.startswith("text+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion_pred,
            "expression": expression_pred,
        }
        
class IdentitySimpleFlowCFGuider:
    def __init__(self):
        pass
    
    def __call__(
        self, 
        motion, 
        expression,
        timestep,
        model_kwargs,
        is_sample: bool = False,
        task="audio2motion+expression",
    ):
        return {
            "motion": motion, 
            "expression": expression, 
            "timestep": timestep, 
            **model_kwargs
        }
    
    def guide(
        self, 
        motion, 
        expression, 
        task="audio2motion+expression",
    ):
        return {
            "motion": motion,
            "expression": expression,
        }

class IdentityEmageGuider:
    def __init__(self):
        pass
    
    def __call__(
        self, 
        motion, 
        timestep,
        model_kwargs,
        is_sample: bool = False,
        task="audio2motion+expression",
        **kwargs,
    ):
        return {
            "motion": motion, 
            "timestep": timestep, 
            **model_kwargs,
            **kwargs,
        }
    
    def guide(
        self, 
        motion, 
        task="audio2motion+expression",
        **kwargs,
    ):
        return {
            "motion": motion,
        }


class BLDCFGGuider:
    """
    Unified CFG guider for BLD (Block-Level Diffusion) inference and training.

    Supports two independent CFG axes:
    1. Audio/condition CFG: drop audio context to zero (controlled by ucg_prob / cfg_audio_scale)
    2. History-state CFG: replace history state with null_state (controlled by cfg_history_scale)

    Each axis can be enabled/disabled independently:
    - cfg_audio_scale > 0 and ucg_prob > 0: enables audio condition CFG
    - cfg_history_scale > 0: enables history-state CFG (requires model to have learnable null_state)

    Training:
    - Audio CFG: randomly zeros context/expression with probability ucg_prob
    - History CFG: handled internally by GatedDeltaNetForBLD (not by this guider)

    Inference (decode) — single-axis or joint mode (cfg_mode="joint", default):
    - Audio CFG only: 2B batch, cond=v11 / uncond=v01
    - History CFG only: 2B batch, cond=v11 / uncond=v10
    - Both (joint): 2B batch, cond=v11 / uncond=v00, scale=max(γ_a, γ_s)
    - Prefill / cache-write: passes through without doubling

    Inference (decode) — multi-condition mode (cfg_mode != "joint"):
    Requires both axes enabled. Uses 3B or 4B batch to decompose guidance:
    - independent: v00 + γ_a*(v10-v00) + γ_s*(v01-v00)           [3B]
    - audio_first: v00 + γ_a*(v10-v00) + γ_s*(v11-v10)           [3B]
    - state_first: v00 + γ_s*(v01-v00) + γ_a*(v11-v01)           [3B]
    - interaction: v00 + γ_a*(v10-v00) + γ_s*(v01-v00) + γ_aγ_sΔ [4B]

    Corner definitions:
    - v00 = v(x_t | audio_dropped, null_state)
    - v10 = v(x_t | audio_present, null_state)
    - v01 = v(x_t | audio_dropped, clean_state)
    - v11 = v(x_t | audio_present, clean_state)
    """

    # Corner specs for multi-condition CFG: each tuple is (audio_is_cond, state_is_clean)
    # Chunk order convention: v00 always first (fully uncond baseline)
    _CORNER_SPECS = {
        "independent":  [(False, False), (True, False),  (False, True)],                    # v00, v10, v01
        "audio_first":  [(False, False), (True, False),  (True, True)],                     # v00, v10, v11
        "state_first":  [(False, False), (False, True),  (True, True)],                     # v00, v01, v11
        "interaction":  [(False, False), (True, False),  (False, True), (True, True)],      # v00, v10, v01, v11
    }
    _VALID_CFG_MODES = ("joint", "independent", "audio_first", "state_first", "interaction")

    def __init__(
        self,
        ucg_prob: float = 0.0,
        cfg_audio_scale: float = 0.0,
        cfg_history_scale: float = 0.0,
        use_so3_guidance: bool = False,
        cfg_schedule: BaseCFGSchedule = None,
        cfg_schedule_type: str = "constant",
        cfg_schedule_kwargs: dict = None,
        drop_expression_with_context: bool = True,
        cfg_mode: str = "joint",
        # Legacy aliases for backward compatibility
        cfg_motion_scale: float = None,
    ):
        self.ucg_prob = ucg_prob
        # cfg_motion_scale is a legacy alias for cfg_audio_scale
        if cfg_motion_scale is not None and cfg_audio_scale == 0.0:
            cfg_audio_scale = cfg_motion_scale
        self.cfg_audio_scale = cfg_audio_scale
        self.cfg_history_scale = cfg_history_scale
        self.use_so3_guidance = use_so3_guidance
        self.drop_expression_with_context = drop_expression_with_context

        if cfg_mode not in self._VALID_CFG_MODES:
            raise ValueError(f"cfg_mode must be one of {self._VALID_CFG_MODES}, got '{cfg_mode}'")
        self.cfg_mode = cfg_mode

        self.audio_cfg_enabled = (self.cfg_audio_scale > 0.0 and self.ucg_prob > 0.0)
        self.history_cfg_enabled = (self.cfg_history_scale > 0.0)
        # Multi-condition CFG: only when both axes are active and mode is not joint
        self.use_multicond_cfg = (
            self.audio_cfg_enabled and self.history_cfg_enabled and self.cfg_mode != "joint"
        )

        if self.use_multicond_cfg and not self.drop_expression_with_context:
            raise ValueError(
                f"Multi-condition CFG (cfg_mode='{cfg_mode}') requires drop_expression_with_context=True. "
                f"Otherwise expression leaks audio information into audio-dropped corners, "
                f"breaking the v00/v01 corner semantics."
            )

        if cfg_schedule is not None:
            self.cfg_schedule = cfg_schedule
        else:
            cfg_schedule_kwargs = cfg_schedule_kwargs or {}
            self.cfg_schedule = build_cfg_schedule(cfg_schedule_type, **cfg_schedule_kwargs)

    @staticmethod
    def _expand_scale_like(scale, x: torch.Tensor):
        if not isinstance(scale, torch.Tensor):
            return scale
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-1)
        return scale.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def _so3_extrapolate_rot6d(cond_rot6d: torch.Tensor, uncond_rot6d: torch.Tensor, scale):
        """Geodesic CFG extrapolation on SO(3) for rot6d channels only.

        Formula:
            R_guided = R_u @ Exp(scale * Log(R_u^T @ R_c))
        where R_u is uncond rotation and R_c is cond rotation.
        """
        origin_shape = cond_rot6d.shape
        cond_rot6d = cond_rot6d.reshape(*origin_shape[:-1], -1, 6)
        uncond_rot6d = uncond_rot6d.reshape(*origin_shape[:-1], -1, 6)

        rot_c = rc.rotation_6d_to_matrix(cond_rot6d)
        rot_u = rc.rotation_6d_to_matrix(uncond_rot6d)

        rel_rot = rot_u.transpose(-1, -2) @ rot_c
        rel_omega = rc.matrix_to_axis_angle(rel_rot)
        rel_omega = rel_omega * BLDCFGGuider._expand_scale_like(scale, rel_omega)
        guided_rel_rot = rc.axis_angle_to_matrix(rel_omega)
        guided_rot = rot_u @ guided_rel_rot

        guided_rot6d = rc.matrix_to_rotation_6d(guided_rot)
        return guided_rot6d.reshape(origin_shape)

    def _guide_motion(self, cond_motion: torch.Tensor, uncond_motion: torch.Tensor, scale):
        if not self.use_so3_guidance:
            return uncond_motion + scale * (cond_motion - uncond_motion)

        rot_dim = cond_motion.shape[-1] - 3
        guided_rot = self._so3_extrapolate_rot6d(
            cond_motion[..., :rot_dim],
            uncond_motion[..., :rot_dim],
            scale,
        )
        cond_rest = cond_motion[..., rot_dim:]
        uncond_rest = uncond_motion[..., rot_dim:]
        rest_scale = self._expand_scale_like(scale, cond_rest)
        guided_rest = uncond_rest + rest_scale * (cond_rest - uncond_rest)
        return torch.cat([guided_rot, guided_rest], dim=-1)

    def __call__(
        self,
        motion,
        timestep,
        *args,
        model_kwargs=None,
        is_sample: bool = False,
        task="audio2motion",
        **runtime_kwargs,
    ):
        if model_kwargs is None:
            if len(args) > 0 and isinstance(args[0], dict):
                model_kwargs = args[0]
            else:
                model_kwargs = {}

        merged_kwargs = dict(model_kwargs)
        merged_kwargs.update(runtime_kwargs)

        if is_sample:
            return self.sample_guiding(motion, timestep, merged_kwargs, task)
        return self.training_drop(motion, timestep, merged_kwargs, task)

    # ==================== Training ====================

    def _generate_drop_mask(self, batch_size: int, device):
        if self.ucg_prob > 0.0:
            return torch.bernoulli(self.ucg_prob * torch.ones(batch_size, device=device)).bool()
        return torch.zeros(batch_size, device=device).bool()

    @staticmethod
    def _zero_with_mask(x, drop_mask):
        if x is None:
            return None
        drop_mask_bc = expand_dims_like(drop_mask.to(dtype=x.dtype), x)
        return x * (1.0 - drop_mask_bc)

    def training_drop(self, motion, timestep, model_kwargs, task="audio2motion"):
        if task.startswith("uncondition2"):
            return {"motion": motion, "timestep": timestep, **model_kwargs}

        kwargs = dict(model_kwargs)

        # Audio condition drop (training)
        if self.audio_cfg_enabled and "context" in kwargs:
            context = kwargs["context"]
            drop_mask = self._generate_drop_mask(context.shape[0], context.device)
            kwargs["context"] = self._zero_with_mask(context, drop_mask)
            if self.drop_expression_with_context and kwargs.get("expression") is not None:
                kwargs["expression"] = self._zero_with_mask(kwargs["expression"], drop_mask)

        # History drop is handled internally by GatedDeltaNetForBLD (not here)

        return {"motion": motion, "timestep": timestep, **kwargs}

    # ==================== Sampling ====================

    @staticmethod
    def _duplicate_kv_cache(kv_cache, n_copies=2):
        """Duplicate kv_cache along batch dim for NB inference."""
        if kv_cache is None:
            return None
        duplicated = []
        for block_cache in kv_cache:
            new_cache = {}
            for key, value in block_cache.items():
                if value is None:
                    new_cache[key] = None
                elif isinstance(value, torch.Tensor):
                    if key in ("deltanet_state", "k", "v") or key.endswith("_k") or key.endswith("_v") \
                        or "conv_cache" in key:
                        new_cache[key] = torch.cat([value] * n_copies, dim=0)
                    elif key in ("global_end_index", "local_end_index"):
                        new_cache[key] = value
                    else:
                        if value.dim() >= 1:
                            new_cache[key] = torch.cat([value] * n_copies, dim=0)
                        else:
                            new_cache[key] = value
                else:
                    new_cache[key] = value
            duplicated.append(new_cache)
        return duplicated

    def sample_guiding(self, motion, timestep, model_kwargs, task="audio2motion"):
        if task.startswith("uncondition2"):
            return {"motion": motion, "timestep": timestep, **model_kwargs}

        # Prefill / cache-write: no CFG doubling
        if model_kwargs.get("prefill_only", False) or model_kwargs.get("update_kv_cache", False):
            return {"motion": motion, "timestep": timestep, **model_kwargs}

        # If neither CFG axis is active at inference, pass through
        if not self.audio_cfg_enabled and not self.history_cfg_enabled:
            return {"motion": motion, "timestep": timestep, **model_kwargs}

        B = motion.shape[0]

        # Multi-condition CFG: 3B or 4B batch via corner specs
        if self.use_multicond_cfg:
            return self._sample_guiding_multicond(motion, timestep, model_kwargs, B)

        # ---- Existing 2B path (single-axis or joint) ----
        # Double batch
        motion = torch.cat([motion, motion], dim=0)
        timestep = torch.cat([timestep, timestep], dim=0)

        kwargs = {}
        for key, value in model_kwargs.items():
            if key in ("uncond_context", "uncond_expression"):
                continue
            if key == "kv_cache" and isinstance(value, dict):
                # Duplicate kv_cache for history CFG (needed even for audio-only CFG to keep batch aligned)
                new_kv = {}
                for cache_key, cache_list in value.items():
                    if isinstance(cache_list, list):
                        new_kv[cache_key] = self._duplicate_kv_cache(cache_list)
                    else:
                        new_kv[cache_key] = cache_list
                kwargs["kv_cache"] = new_kv
            elif key == "context" and self.audio_cfg_enabled:
                context = value
                uncond_context = model_kwargs.get("uncond_context", None)
                if uncond_context is None:
                    uncond_context = torch.zeros_like(context)
                kwargs["context"] = torch.cat([context, uncond_context], dim=0)
            elif key == "expression" and self.audio_cfg_enabled and self.drop_expression_with_context:
                if value is None:
                    kwargs["expression"] = None
                else:
                    uncond_expression = model_kwargs.get("uncond_expression", None)
                    if uncond_expression is None:
                        uncond_expression = torch.zeros_like(value)
                    kwargs["expression"] = torch.cat([value, uncond_expression], dim=0)
            elif value is None:
                kwargs[key] = None
            elif isinstance(value, torch.Tensor):
                kwargs[key] = torch.cat([value, value], dim=0)
            else:
                kwargs[key] = value

        # History CFG: construct history_cfg_alpha for the 2B batch
        # cond half: alpha=1 (use real state), uncond half: alpha=0 (use null state)
        if self.history_cfg_enabled:
            kwargs["history_cfg_alpha"] = torch.cat([
                torch.ones(B, dtype=torch.float32, device=motion.device),
                torch.zeros(B, dtype=torch.float32, device=motion.device),
            ], dim=0)

        return {"motion": motion, "timestep": timestep, **kwargs}

    def _sample_guiding_multicond(self, motion, timestep, model_kwargs, B):
        """Construct NB batch for multi-condition CFG (independent/audio_first/state_first/interaction)."""
        specs = self._CORNER_SPECS[self.cfg_mode]
        n_corners = len(specs)

        # Repeat motion and timestep
        motion = torch.cat([motion] * n_corners, dim=0)
        timestep = torch.cat([timestep] * n_corners, dim=0)

        # Prepare cond/uncond variants
        cond_context = model_kwargs.get("context", None)
        uncond_context = model_kwargs.get("uncond_context", None)
        if cond_context is not None and uncond_context is None:
            uncond_context = torch.zeros_like(cond_context)

        cond_expression = model_kwargs.get("expression", None)
        uncond_expression = model_kwargs.get("uncond_expression", None)
        if cond_expression is not None and uncond_expression is None:
            uncond_expression = torch.zeros_like(cond_expression)

        # Build per-corner tensors for context, expression, history_cfg_alpha
        context_chunks = []
        expression_chunks = []
        alpha_chunks = []
        for audio_on, state_on in specs:
            if cond_context is not None:
                context_chunks.append(cond_context if audio_on else uncond_context)
            if self.drop_expression_with_context and cond_expression is not None:
                expression_chunks.append(cond_expression if audio_on else uncond_expression)
            alpha_chunks.append(
                torch.ones(B, dtype=torch.float32, device=motion.device) if state_on
                else torch.zeros(B, dtype=torch.float32, device=motion.device)
            )

        # Build kwargs
        kwargs = {}
        for key, value in model_kwargs.items():
            if key in ("uncond_context", "uncond_expression"):
                continue
            if key == "context" and context_chunks:
                kwargs["context"] = torch.cat(context_chunks, dim=0)
            elif key == "expression" and self.drop_expression_with_context and expression_chunks:
                kwargs["expression"] = torch.cat(expression_chunks, dim=0)
            elif key == "kv_cache" and isinstance(value, dict):
                new_kv = {}
                for cache_key, cache_list in value.items():
                    if isinstance(cache_list, list):
                        new_kv[cache_key] = self._duplicate_kv_cache(cache_list, n_copies=n_corners)
                    else:
                        new_kv[cache_key] = cache_list
                kwargs["kv_cache"] = new_kv
            elif value is None:
                kwargs[key] = None
            elif isinstance(value, torch.Tensor):
                kwargs[key] = torch.cat([value] * n_corners, dim=0)
            else:
                kwargs[key] = value

        kwargs["history_cfg_alpha"] = torch.cat(alpha_chunks, dim=0)

        return {"motion": motion, "timestep": timestep, **kwargs}

    # ==================== Guide (fusion) ====================

    def _get_timestep_for_schedule(self, timestep, per_chunk_B):
        """Extract timestep tensor suitable for cfg_schedule.get_scale()."""
        if timestep is None:
            return None
        ts = timestep
        if ts.dim() > 1:
            ts = ts[:, 0]
        if ts.shape[0] > per_chunk_B:
            ts = ts[:per_chunk_B]
        return ts

    def guide(self, motion, timestep=None, task="audio2motion"):
        if task.startswith("uncondition2"):
            return {"motion": motion}

        if not self.audio_cfg_enabled and not self.history_cfg_enabled:
            return {"motion": motion}

        # Multi-condition CFG: 3 or 4 chunk fusion
        if self.use_multicond_cfg:
            return self._guide_multicond(motion, timestep)

        # ---- Existing 2B path (single-axis or joint) ----
        cond_motion, uncond_motion = motion.chunk(2, dim=0)

        # Determine effective scale: use the max of the two active scales
        # (when both are active, they share the same cond/uncond split)
        if self.audio_cfg_enabled and self.history_cfg_enabled:
            base_scale = max(self.cfg_audio_scale, self.cfg_history_scale)
        elif self.audio_cfg_enabled:
            base_scale = self.cfg_audio_scale
        else:
            base_scale = self.cfg_history_scale

        ts = self._get_timestep_for_schedule(timestep, cond_motion.shape[0])
        scale = self.cfg_schedule.get_scale(ts, base_scale) if ts is not None else base_scale

        motion_pred = self._guide_motion(cond_motion, uncond_motion, scale)
        return {"motion": motion_pred}

    def _guide_multicond(self, motion, timestep):
        """Multi-condition CFG fusion for independent/audio_first/state_first/interaction."""
        if self.use_so3_guidance:
            raise NotImplementedError(
                f"SO(3) geodesic guidance is not yet supported with multi-condition CFG "
                f"(cfg_mode='{self.cfg_mode}'). Use cfg_mode='joint' or disable use_so3_guidance."
            )
        specs = self._CORNER_SPECS[self.cfg_mode]
        n_corners = len(specs)
        chunks = motion.chunk(n_corners, dim=0)
        B = chunks[0].shape[0]

        # Compute two scales independently
        ts = self._get_timestep_for_schedule(timestep, B)
        if ts is not None:
            audio_scale = self.cfg_schedule.get_scale(ts, self.cfg_audio_scale)
            history_scale = self.cfg_schedule.get_scale(ts, self.cfg_history_scale)
        else:
            audio_scale = self.cfg_audio_scale
            history_scale = self.cfg_history_scale

        # Chunk order: always v00 first, then mode-specific corners
        # See _CORNER_SPECS for the exact layout
        v00 = chunks[0]

        if self.cfg_mode == "independent":
            # [v00, v10, v01]
            v10, v01 = chunks[1], chunks[2]
            # v00 + γ_a*(v10-v00) + γ_s*(v01-v00)
            motion_pred = (
                v00
                + self._expand_scale_like(audio_scale, v00) * (v10 - v00)
                + self._expand_scale_like(history_scale, v00) * (v01 - v00)
            )
        elif self.cfg_mode == "audio_first":
            # [v00, v10, v11]
            v10, v11 = chunks[1], chunks[2]
            # v00 + γ_a*(v10-v00) + γ_s*(v11-v10)
            motion_pred = (
                v00
                + self._expand_scale_like(audio_scale, v00) * (v10 - v00)
                + self._expand_scale_like(history_scale, v00) * (v11 - v10)
            )
        elif self.cfg_mode == "state_first":
            # [v00, v01, v11]
            v01, v11 = chunks[1], chunks[2]
            # v00 + γ_s*(v01-v00) + γ_a*(v11-v01)
            motion_pred = (
                v00
                + self._expand_scale_like(history_scale, v00) * (v01 - v00)
                + self._expand_scale_like(audio_scale, v00) * (v11 - v01)
            )
        elif self.cfg_mode == "interaction":
            # [v00, v10, v01, v11]
            v10, v01, v11 = chunks[1], chunks[2], chunks[3]
            # Δ = (v11-v10) - (v01-v00)
            delta = (v11 - v10) - (v01 - v00)
            sa = self._expand_scale_like(audio_scale, v00)
            sh = self._expand_scale_like(history_scale, v00)
            # v00 + γ_a*(v10-v00) + γ_s*(v01-v00) + γ_a*γ_s*Δ
            motion_pred = v00 + sa * (v10 - v00) + sh * (v01 - v00) + sa * sh * delta

        return {"motion": motion_pred}


class VanillaCFGEmageGuider:
    def __init__(
        self,
        ucg_prob: float = 0.1,
        cfg_motion_scale: float = 7.5,
        cfg_schedule: BaseCFGSchedule = None,
        cfg_schedule_type: str = "constant",
        cfg_schedule_kwargs: dict = None,
    ):
        """
        使用标准 CFG 形式: output = uncond + scale * (cond - uncond)
        
        Args:
            ucg_prob: unconditional guidance 的 drop 概率
            cfg_motion_scale: 基础的 CFG scale，推荐范围 5.0-10.0，默认 7.5
            cfg_schedule: CFG scale 调度器实例，如果提供则直接使用
            cfg_schedule_type: 调度器类型，可选 "constant", "interval", "linear"
            cfg_schedule_kwargs: 传递给调度器的额外参数
        """
        self.ucg_prob = ucg_prob
        self.cfg_motion_scale = cfg_motion_scale
        
        # 初始化 CFG schedule
        if cfg_schedule is not None:
            self.cfg_schedule = cfg_schedule
        else:
            cfg_schedule_kwargs = cfg_schedule_kwargs or {}
            self.cfg_schedule = build_cfg_schedule(cfg_schedule_type, **cfg_schedule_kwargs)
    
    def __call__(
        self, 
        motion, 
        timestep,
        model_kwargs,
        is_sample: bool = False,
        task="audio2motion+expression",
    ):
        assert task in [
            "text2motion", "audio2motion", "uncondition2motion"
            "text2motion+expression", "audio2motion+expression", "uncondition2motion+expression"
        ]
        if is_sample:
            return self.sample_guiding(
                motion, timestep, model_kwargs, task)
        else:
            return self.training_drop(
                motion, timestep, model_kwargs, task)
    
    def _generate_drop_mask(self, batch_size, device):
        """
        生成 drop mask，用于同时 drop audio/text 和 expression。
        
        Returns:
            drop_mask: (batch_size,) bool tensor，True 表示该样本需要 drop（使用 unconditional）
        """
        if self.ucg_prob > 0.0:
            drop_mask = torch.bernoulli(
                self.ucg_prob * torch.ones(batch_size, device=device)
            ).bool()
        else:
            drop_mask = torch.zeros(batch_size, device=device).bool()
        return drop_mask
    
    def _apply_drop_mask(self, emb, drop_mask):
        """
        根据 drop_mask 将 tensor 转换为 list，被 drop 的位置设为 None。
        
        Args:
            emb: (N, T, D) tensor
            drop_mask: (N,) bool tensor
            
        Returns:
            list of tensors or None
        """
        _emb = []
        for i in range(emb.shape[0]):
            _emb.append(None if drop_mask[i] else emb[i])
        return _emb
    
    def cond_drop(
        self,
        emb,
    ):
        """兼容旧接口：单独 drop 一个条件"""
        drop_mask = self._generate_drop_mask(emb.shape[0], emb.device)
        return self._apply_drop_mask(emb, drop_mask)
    
    def training_drop(
        self, 
        motion, 
        timestep,
        model_kwargs,
        task="audio2motion+expression",
    ):
        """
        训练时的条件 drop。
        audio/text 和 expression 使用相同的 drop_mask，保证同步 drop。
        """
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            # 生成统一的 drop_mask
            drop_mask = self._generate_drop_mask(
                model_kwargs["txt"].shape[0], 
                model_kwargs["txt"].device
            )
            # 同时 drop text 和 expression
            model_kwargs["txt"] = self._apply_drop_mask(model_kwargs["txt"], drop_mask)
            if "expression" in model_kwargs and model_kwargs["expression"] is not None:
                model_kwargs["expression"] = self._apply_drop_mask(model_kwargs["expression"], drop_mask)
                
        elif task.startswith("audio2"):
            assert "audio" in model_kwargs, "audio is required"
            # 生成统一的 drop_mask
            drop_mask = self._generate_drop_mask(
                model_kwargs["audio"].shape[0], 
                model_kwargs["audio"].device
            )
            # 同时 drop audio 和 expression
            model_kwargs["audio"] = self._apply_drop_mask(model_kwargs["audio"], drop_mask)
            if "expression" in model_kwargs and model_kwargs["expression"] is not None:
                model_kwargs["expression"] = self._apply_drop_mask(model_kwargs["expression"], drop_mask)
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "timestep": timestep,
            **model_kwargs
        }
    
    def _build_cond_uncond_list(self, emb):
        """
        将 tensor 构造为 [cond_0, cond_1, ..., uncond_0, uncond_1, ...] 形式的 list。
        前半部分是 conditional，后半部分是 unconditional (None)。
        """
        _emb = []
        for i in range(emb.shape[0]):
            _emb.append(emb[i])
        for i in range(emb.shape[0]):
            _emb.append(None)
        return _emb
    
    def sample_guiding(
        self, 
        motion, 
        timestep,
        model_kwargs,
        task="audio2motion",
    ):
        """
        采样时的 CFG 准备。
        构造 [conditional, unconditional] 的 batch，audio/text 和 expression 同步处理。
        """
        if task.startswith("uncondition2"):
            kwargs = model_kwargs
            
        elif task.startswith("text2"):
            assert "txt" in model_kwargs, "text embedding is required"
            
            motion = torch.cat([motion] * 2)
            timestep = torch.cat([timestep] * 2)
            
            kwargs = {}
            for key in model_kwargs:
                if key == "txt":
                    # text 构造 [cond, uncond] 形式
                    kwargs["txt"] = self._build_cond_uncond_list(model_kwargs["txt"])
                elif key == "expression":
                    # expression 也构造 [cond, uncond] 形式，与 text 同步
                    if model_kwargs["expression"] is not None:
                        kwargs["expression"] = self._build_cond_uncond_list(model_kwargs["expression"])
                    else:
                        kwargs["expression"] = None
                elif model_kwargs[key] is None:
                    kwargs[key] = None
                elif isinstance(model_kwargs[key], torch.Tensor):
                    kwargs[key] = torch.cat([model_kwargs[key]] * 2)
                else:
                    kwargs[key] = model_kwargs[key]
        
        elif task.startswith("audio2"):
            assert "audio" in model_kwargs, "audio is required"
            
            motion = torch.cat([motion] * 2)
            timestep = torch.cat([timestep] * 2)
            
            kwargs = {}
            for key in model_kwargs:
                if key == "audio":
                    # audio 构造 [cond, uncond] 形式
                    kwargs["audio"] = self._build_cond_uncond_list(model_kwargs["audio"])
                elif key == "expression":
                    # expression 也构造 [cond, uncond] 形式，与 audio 同步
                    if model_kwargs["expression"] is not None:
                        kwargs["expression"] = self._build_cond_uncond_list(model_kwargs["expression"])
                    else:
                        kwargs["expression"] = None
                elif model_kwargs[key] is None:
                    kwargs[key] = None
                elif isinstance(model_kwargs[key], torch.Tensor):
                    kwargs[key] = torch.cat([model_kwargs[key]] * 2)
                else:
                    kwargs[key] = model_kwargs[key]
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "timestep": timestep,
            **kwargs
        }
    
    def guide(
        self, 
        motion,
        timestep,
        task="audio2motion",
    ):
        """
        CFG 标准形式: output = uncond + scale * (cond - uncond)
        - scale = 1.0 时，output = cond（纯条件输出）
        - scale > 1.0 时，向条件方向外推增强
        - 推荐 scale 范围: 5.0 - 10.0，默认 7.5
        """
        if task.startswith("uncondition2"):
            motion_pred = motion
        
        elif task.startswith("text2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            # 获取基于 timestep 的 scale
            scale = self.cfg_schedule.get_scale(timestep, self.cfg_motion_scale)
            # 标准 CFG: uncond + scale * (cond - uncond)
            motion_pred = uncond_motion + scale * (motion - uncond_motion)

        elif task.startswith("audio2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            # 获取基于 timestep 的 scale
            scale = self.cfg_schedule.get_scale(timestep, self.cfg_motion_scale)
            # 标准 CFG: uncond + scale * (cond - uncond)
            motion_pred = uncond_motion + scale * (motion - uncond_motion)
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion_pred,
        }

class SimpleVanillaCFGEmageGuider:
    def __init__(
        self,
        ucg_prob: float = 0.1,
        cfg_motion_scale: float = 4.5,
        cfg_expression_scale: float = 4.5,
        max_classes=11195,
        max_training_steps=1000,
        disable_expression_guidance=False,
    ):
        self.ucg_prob = ucg_prob
        self.cfg_motion_scale = cfg_motion_scale
        self.cfg_expression_scale = cfg_expression_scale
        self.max_classes = max_classes
        self.max_training_steps_index = max_training_steps - 1
        self.disable_expression_guidance = disable_expression_guidance
    
    def __call__(
        self, 
        motion, 
        timestep,
        model_kwargs,
        is_sample: bool = False,
        task="audio2motion+expression",
    ):
        assert task in [
            "text2motion", "audio2motion", "uncondition2motion"
            "text2motion+expression", "audio2motion+expression", "uncondition2motion+expression"
        ]
        if is_sample:
            return self.sample_guiding(
                motion, timestep, model_kwargs, task)
        else:
            return self.training_drop(
                motion, timestep, model_kwargs, task)
    
    def cond_drop(
        self,
        emb,
    ):
        # 只drop text标签即可
        if self.ucg_prob > 0.0:
            drop_ids = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_prob)
                        * torch.ones(emb.shape[0], device=emb.device)
                    ),
                    emb,
                )
            )
            if not torch.is_floating_point(emb):
                emb = torch.where(drop_ids.bool(), torch.randint_like(emb, self.max_classes), emb)
            else:
                emb = torch.where(drop_ids.bool(), torch.randn_like(emb), emb)
        return emb
    
    
    def training_drop(
        self, 
        motion, 
        timestep,
        model_kwargs,
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "context" in model_kwargs, "text embedding is required"
            model_kwargs["context"] = self.cond_drop(model_kwargs["context"])
        elif task.startswith("audio2"):
            assert "context" in model_kwargs, "audio is required"
            model_kwargs["context"] = self.cond_drop(model_kwargs["context"])
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "timestep": timestep,
            **model_kwargs
        }
    
    def sample_guiding(
        self, 
        motion, 
        timestep,
        model_kwargs,
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        elif task.startswith("text2"):
            assert "context" in model_kwargs, "text embedding is required"
            emb = model_kwargs["context"]
            un_emb = model_kwargs.get("uncond_context", None)
            if un_emb is None:
                un_emb = torch.zeros_like(emb)
            motion = torch.cat([motion] * 2)
            timestep = torch.cat([timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if key == "context" or key == "uncond_context":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            kwargs["context"] = torch.cat([emb, un_emb])
        
        elif task.startswith("audio2"):
            audio = model_kwargs["audio"]
            un_audio = model_kwargs.get("uncond_context", None)
            if un_audio is None:
                un_audio = torch.zeros_like(audio)
            motion = torch.cat([motion] * 2)
            timestep = torch.cat([timestep] * 2)
            kwargs = {}
            for key in model_kwargs:
                if key == "context" or key == "uncond_context":
                    continue
                if model_kwargs[key] is None:
                    kwargs[key] = None
                    continue
                kwargs[key] = torch.cat([model_kwargs[key]] * 2)
            kwargs["context"] = torch.cat([audio, un_audio])
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion, 
            "timestep": timestep,
            **kwargs
        }
    
    def guide(
        self, 
        motion, 
        task="audio2motion+expression",
    ):
        if task.startswith("uncondition2"):
            pass
        
        elif task.startswith("text2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)

        elif task.startswith("audio2"):
            motion, uncond_motion = motion.chunk(2, dim=0)
            motion_pred = uncond_motion + self.cfg_motion_scale * (motion - uncond_motion)
        
        elif task.startswith("text+audio2"):
            raise NotImplementedError(f"Unsupported task: {task}, has not been implemented yet.")
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return {
            "motion": motion_pred,
        }
