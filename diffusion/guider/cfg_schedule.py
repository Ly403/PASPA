import torch
from abc import ABC, abstractmethod


class BaseCFGSchedule(ABC):
    """CFG Scale 调度器的基类"""
    
    @abstractmethod
    def get_scale(self, timestep: torch.Tensor, base_scale: float) -> torch.Tensor:
        """
        根据 timestep 返回对应的 CFG scale
        
        Args:
            timestep: 当前的时间步，shape 为 (batch_size,)
            base_scale: 基础的 CFG scale 值
            
        Returns:
            scale: 对应的 CFG scale，shape 为 (batch_size, 1, 1) 或标量
        """
        pass


class ConstantCFGSchedule(BaseCFGSchedule):
    """
    常量 CFG Schedule，始终返回固定的 scale
    """
    
    def __init__(self):
        pass
    
    def get_scale(self, timestep: torch.Tensor, base_scale: float) -> torch.Tensor:
        """始终返回 base_scale"""
        return base_scale
    
    def __repr__(self):
        return "ConstantCFGSchedule()"


class IntervalCFGSchedule(BaseCFGSchedule):
    """
    区间 CFG Schedule，只在指定的 timestep 区间内应用 CFG
    
    在 [start, end] 区间内返回 base_scale，区间外返回 1.0（即不应用 CFG）
    """
    
    def __init__(
        self, 
        start: int = 100, 
        end: int = 600,
        outside_scale: float = 1.0,
    ):
        """
        Args:
            start: 区间起始 timestep（包含）
            end: 区间结束 timestep（包含）
            outside_scale: 区间外的 scale，默认为 1.0（不应用 CFG）
        """
        self.start = start
        self.end = end
        self.outside_scale = outside_scale
    
    def get_scale(self, timestep: torch.Tensor, base_scale: float) -> torch.Tensor:
        """
        在 [start, end] 区间内返回 base_scale，否则返回 outside_scale
        """
        # timestep shape: (batch_size,) 或 (batch_size * 2,) for CFG
        # 我们只需要前半部分的 timestep（CFG 会复制一份）
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # 判断是否在区间内
        in_interval = (timestep >= self.start) & (timestep <= self.end)
        
        # 创建 scale tensor
        scale = torch.where(
            in_interval,
            torch.full_like(timestep, base_scale, dtype=torch.float32),
            torch.full_like(timestep, self.outside_scale, dtype=torch.float32),
        )
        
        # 扩展维度以便广播 (batch_size,) -> (batch_size, 1, 1)
        scale = scale.view(-1, 1, 1)
        
        return scale
    
    def __repr__(self):
        return f"IntervalCFGSchedule(start={self.start}, end={self.end}, outside_scale={self.outside_scale})"


class LinearCFGSchedule(BaseCFGSchedule):
    """
    线性 CFG Schedule，scale 随 timestep 线性变化
    
    在 timestep=max_timestep 时 scale=start_scale，
    在 timestep=0 时 scale=end_scale
    """
    
    def __init__(
        self,
        start_scale: float = 1.0,
        end_scale: float = 4.5,
        max_timestep: int = 1000,
    ):
        """
        Args:
            start_scale: timestep=max_timestep 时的 scale
            end_scale: timestep=0 时的 scale
            max_timestep: 最大 timestep 值
        """
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.max_timestep = max_timestep
    
    def get_scale(self, timestep: torch.Tensor, base_scale: float) -> torch.Tensor:
        """
        线性插值计算 scale
        scale = start_scale + (end_scale - start_scale) * (max_timestep - timestep) / max_timestep
        """
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # 计算线性插值
        t_normalized = (self.max_timestep - timestep.float()) / self.max_timestep
        scale = self.start_scale + (self.end_scale - self.start_scale) * t_normalized
        
        # 扩展维度
        scale = scale.view(-1, 1, 1)
        
        return scale
    
    def __repr__(self):
        return f"LinearCFGSchedule(start_scale={self.start_scale}, end_scale={self.end_scale}, max_timestep={self.max_timestep})"


def build_cfg_schedule(schedule_type: str = "constant", **kwargs) -> BaseCFGSchedule:
    """
    构建 CFG Schedule 的工厂函数
    
    Args:
        schedule_type: schedule 类型，可选 "constant", "interval", "linear"
        **kwargs: 传递给对应 schedule 类的参数
        
    Returns:
        CFG Schedule 实例
    """
    if schedule_type == "constant":
        return ConstantCFGSchedule()
    elif schedule_type == "interval":
        return IntervalCFGSchedule(**kwargs)
    elif schedule_type == "linear":
        return LinearCFGSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

