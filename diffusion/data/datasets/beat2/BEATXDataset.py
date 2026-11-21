import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from diffusion.utils.rotation_conversions import axis_angle_to_rotation_6d
from scipy.io import loadmat
import random
import lmdb
import pickle

def symmetrize_smplx_parameters(pose_params, trans):
    """
    对称化 SMPLX 模型的姿态和形状参数
    
    参数:
        pose_params: 姿态参数, shape (N, 165) 或 (165,)
        beta_params: 形状参数, shape (N, 10) 或 (10,), 可选
        left_right_pairs: 左右关节对的列表, 可选
        
    返回:
        对称化后的姿态参数和形状参数
    """

    # SMPLX 标准左右关节对 (0-based 索引)
    left_right_pairs =( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body
                     (23,24), #eyes
                     (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51), (37,52), (38,53), (39,54) # hands
                    )
    
    # 确保输入是 torch.Tensor
    if not isinstance(pose_params, torch.Tensor):
        pose_params = torch.tensor(pose_params, dtype=torch.float32)
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans, dtype=torch.float32)

    
    # 创建对称的姿态参数
    sym_pose = pose_params.clone()
    
    # 对每个左右对进行对称化
    for left, right in left_right_pairs:
        # 确保索引在有效范围内
        if left*3+3 <= sym_pose.shape[1] and right*3+3 <= sym_pose.shape[1]:
            # 交换左右参数并翻转旋转方向
            left_rot = sym_pose[:, left*3:left*3+3].clone()
            right_rot = sym_pose[:, right*3:right*3+3].clone()
            
            # 对称化: 左右交换并翻转某些轴的旋转
            sym_pose[:, left*3:left*3+3] = right_rot * torch.tensor([1, -1, -1], dtype=torch.float32)
            sym_pose[:, right*3:right*3+3] = left_rot * torch.tensor([1, -1, -1], dtype=torch.float32)
    
    sym_pose[:, 0:3] = pose_params[:, 0:3] * torch.tensor([1, -1, -1], dtype=torch.float32)
    sym_trans = trans.clone()
    sym_trans[...,0] *=-1
    return sym_pose, sym_trans

class H5Dataset(Dataset):
    def __init__(
        self, 
        h5_path, 
        rotation_method="rot6d",
        use_symmetry_aug=False
    ):
        self.h5_path = h5_path
        
        # -------------------------------------------------------
        # 修改 1: 初始化句柄为 None，绝对不要在这里由 self 长期持有打开的文件
        # -------------------------------------------------------
        self.h5_handler = None
        
        # -------------------------------------------------------
        # 修改 2: 临时打开一次获取长度，然后立即关闭
        # 这样 Dataset 对象可以被安全地 pickle 序列化传给子进程
        # -------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            self.num = len(f.keys())

        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        # 路径防错处理
        try:
            mat_path = os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')
            mat_dict = loadmat(mat_path)
        except:
            raise FileNotFoundError(f"Rest joint position file not found at {mat_path}")

        for k, v in mat_dict.items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.joints = 55
        self.use_symmetry_aug = use_symmetry_aug 
        if use_symmetry_aug:
            assert rotation_method == "axisangle", "symmetry augmentation only support axisangle"
                
    def __len__(self):
        return self.num

    def _open_h5(self):
        """
        辅助函数：在 worker 进程中打开文件
        """
        # swmr=True 有助于多进程并发读取稳定性
        # libver='latest' 使用最新的 HDF5 格式，通常性能更好
        self.h5_handler = h5py.File(self.h5_path, "r", swmr=True, libver='latest')

    def __getitem__(self, idx):
        # -------------------------------------------------------
        # 修改 3: 懒加载逻辑
        # 如果当前进程没有句柄，或者句柄意外关闭了，就打开它
        # -------------------------------------------------------
        if self.h5_handler is None or not self.h5_handler.id.valid:
            self._open_h5()

        # 以下逻辑保持完全不变
        grp = self.h5_handler[f"{idx:010d}"]
        
        betas = grp["betas"][:]
        poses_aa = grp["poses"][:]
        expressions = grp["expressions"][:]
        trans = grp['trans'][:]
        pid = int(grp['pid'][()])
        audio = grp['audio'][:]
        huberts = grp['huberts'][:]
        ori_vid = grp['ori_vid'][()]

        if self.rotation_method == "rot6d":
            # 假设外部函数
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas)
        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        trans = torch.from_numpy(trans)
        huberts = torch.from_numpy(huberts)
        
        assert len(audio) == 16000 * len(poses)//30
        audio = torch.from_numpy(audio)
        
        J = self.rest_joint_position[pid].clone()
        pid_offset = 0
        if self.use_symmetry_aug and random.random() < 0.5:
            # 假设外部函数
            poses, trans = symmetrize_smplx_parameters(poses, trans)
            pid_offset = 30
            J[..., 0] = -J[..., 0]
            
        return {
            "beta": betas.unsqueeze(0).repeat(poses.shape[0], 1),
            "pose": poses,
            "facial": expressions,
            "trans":trans,
            "audio":audio,
            "audio_feat":huberts[0],
            "ori_vid":str(ori_vid, encoding='utf-8'),
            "id":pid + pid_offset,
            "J": J
        }

class LMDBDataset(Dataset):
    def __init__(
        self, 
        lmdb_path, 
        rotation_method="rot6d",
        use_symmetry_aug=False,
        map_size=1024*1024*1024*1024, # 1TB
        readahead=True,
        rest_joint_position_path=None,
    ):
        self.lmdb_path = lmdb_path
        self.rotation_method = rotation_method
        self.use_symmetry_aug = use_symmetry_aug
        self.map_size = map_size
        self.readahead = readahead
        self.require_rest_joint_position = rest_joint_position_path is not None
        # 路径检查
        if not os.path.exists(lmdb_path):
            if os.path.exists(lmdb_path + ".lmdb"):
                self.lmdb_path = lmdb_path + ".lmdb"
            else:
                raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")

        # 预加载长度信息
        env = lmdb.open(
            self.lmdb_path, 
            subdir=False,
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False,
            max_readers=256  # 增加读取器数量
        )
        with env.begin() as txn:
            length_bytes = txn.get('__len__'.encode('ascii'))
            if length_bytes is None:
                raise ValueError(f"LMDB key '__len__' not found in {self.lmdb_path}.")
            self.num = int(length_bytes.decode('ascii'))
        env.close()

        # 预加载joint位置数据
        self.rest_joint_position = {}
        if rest_joint_position_path is None:
            mat_path = os.path.join(os.path.dirname(__file__), 'rest_joint_position.mat')
            try:
                mat_dict = loadmat(mat_path)
            except Exception:
                print(f"Warning: rest_joint_position.mat not found at {mat_path}")
                mat_dict = {}
        else:
            mat_path = os.path.abspath(os.path.expanduser(rest_joint_position_path))
            if not os.path.isfile(mat_path):
                raise FileNotFoundError(
                    f"Rest joint position file not found at {mat_path}")
            mat_dict = loadmat(mat_path)
        self.rest_joint_position_path = mat_path

        for k, v in mat_dict.items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.joints = 55
        if use_symmetry_aug:
            assert rotation_method == "axisangle", "symmetry augmentation only support axisangle"
            
        # 使用进程本地存储
        self._local = torch.utils.data.get_worker_info()
        
        self.trans_mean = [-0.0658139, 1.28218, 0.667069] # 提前算好
        self.trans_std = [0.228098, 0.109556, 0.572325] # 提前算好
        self.rot6d_mean = 0.145201
        self.rot6d_std = 0.558793

        
    def __len__(self):
        return self.num

    def _get_env(self):
        """获取当前进程的LMDB环境"""
        worker_info = torch.utils.data.get_worker_info()
        # DataLoader workers have a numeric id; direct dataset access
        # (including preflight/smoke tests with num_workers=0) does not.
        worker_id = worker_info.id if worker_info is not None else -1
        if hasattr(self, '_envs'):
            if worker_id in self._envs:
                return self._envs[worker_id]
        else:
            self._envs = {}
            
        # 创建新的环境
        env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=self.readahead,  # 启用预读取
            meminit=False,
            max_readers=256,
            map_size=self.map_size
        )
        self._envs[worker_id] = env
        return env

    def __getitem__(self, idx):
        # 获取当前worker的环境
        env = self._get_env()
        
        with env.begin() as txn:
            key = f"{idx:010d}".encode('ascii')
            byte_data = txn.get(key)
            
        if byte_data is None:
            raise IndexError(f"Key {key} not found in LMDB at {self.lmdb_path}.")
            
        # 使用更快的反序列化方法
        data_dict = pickle.loads(byte_data)

        # 数据处理逻辑
        betas = data_dict["betas"]
        poses_aa = data_dict["poses"]
        expressions = data_dict["expressions"]
        trans = data_dict['trans']
        pid = int(data_dict['pid'])
        audio = data_dict['audio']
        huberts = data_dict['huberts']
        ori_vid = data_dict['ori_vid']

        if self.rotation_method == "rot6d":
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas)
        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        trans = torch.from_numpy(trans)
        huberts = torch.from_numpy(huberts)
        
        assert len(audio) == 16000 * len(poses) // 30
        audio = torch.from_numpy(audio)
        
        if pid in self.rest_joint_position:
            J = self.rest_joint_position[pid].clone()
        elif self.require_rest_joint_position:
            raise KeyError(
                f"pid {pid} is missing from the explicitly configured rest "
                f"joint position file {self.rest_joint_position_path}")
        else:
            J = torch.zeros((55, 3)) 

        pid_offset = 0
        if self.use_symmetry_aug and random.random() < 0.5:
            poses, trans = symmetrize_smplx_parameters(poses, trans)
            pid_offset = 30
            J[..., 0] = -J[..., 0]
            
        sample = {
            "beta": betas.unsqueeze(0).repeat(poses.shape[0], 1),
            "pose": poses,
            "facial": expressions,
            "trans": trans,
            "audio": audio,
            "audio_feat": huberts[0],
            "ori_vid": str(ori_vid) if not isinstance(ori_vid, bytes) else str(ori_vid, encoding='utf-8'),
            "id": pid + pid_offset,
            "J": J
        }
        return self._augment_sample_from_record(sample, data_dict)

    def _augment_sample_from_record(self, sample, data_dict):
        """Dataset-specific extension hook; BEAT keeps its original contract."""

        return sample

    def normalize(self, trans, rot6d):
        # 先只normalize trans：
        normalized_trans = trans.clone()
        normalized_trans[..., 0] = (normalized_trans[..., 0] - self.trans_mean[0]) / self.trans_std[0]
        normalized_trans[..., 1] = (normalized_trans[..., 1] - self.trans_mean[1]) / self.trans_std[1]
        normalized_trans[..., 2] = (normalized_trans[..., 2] - self.trans_mean[2]) / self.trans_std[2]
        return normalized_trans, rot6d
        # normalized_rot6d = rot6d.clone()
        # normalized_rot6d = (normalized_rot6d - self.rot6d_mean) / self.rot6d_std
        # return normalized_trans, normalized_rot6d

    def denormalize(self, trans, rot6d):
        denormalized_trans = trans.clone()
        denormalized_trans[..., 0] = denormalized_trans[..., 0] * self.trans_std[0] + self.trans_mean[0]
        denormalized_trans[..., 1] = denormalized_trans[..., 1] * self.trans_std[1] + self.trans_mean[1]
        denormalized_trans[..., 2] = denormalized_trans[..., 2] * self.trans_std[2] + self.trans_mean[2]
        return trans, rot6d
        # denormalized_rot6d = rot6d.clone()
        # denormalized_rot6d = denormalized_rot6d * self.rot6d_std + self.rot6d_mean
        # return denormalized_trans, denormalized_rot6d

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_envs'):
            for env in self._envs.values():
                env.close()

# Example usage
if __name__ == "__main__":
    h5_path = "/remote-home/liuyi/Sana/.cache/train_seq_size_60_stride_size_20_global.h5"
    ds = H5Dataset(h5_path, rotation_method="axisangle", use_symmetry_aug=False)

    print('#dataset', len(ds))

    data_id = 4569
    betas = ds[data_id]["betas"].numpy()
    huberts = ds[data_id]["huberts"].numpy()
    poses = ds[data_id]["poses"].numpy()
    audio = ds[data_id]["audio"].numpy()
    expressions = ds[data_id]["expressions"].numpy()
    trans = ds[data_id]["trans"].numpy()

    print(poses.shape, huberts.shape)
