import os
import wget
import math
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import linalg
import torch
import tempfile
import shutil
from .motion_encoder import VAESKConv

class LVDFace(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0

    def compute(self, pred_vertices, target_vertices):
        t, c = pred_vertices.shape
        diff_pred = pred_vertices[1:, :] - pred_vertices[:-1, :]  
        diff_target = target_vertices[1:, :] - target_vertices[:-1, :] 
        loss = np.abs(diff_pred - diff_target) 
        loss = np.sum(loss)
        self.counter += t * c
        self.sum += loss

    def avg(self):
        return self.sum / self.counter

    def reset(self):
        self.counter = 0
        self.sum = 0


class MSEFace(object):
    def __init__(self):
        self.counter = 0  
        self.sum = 0      

    def compute(self, pred_vertices, target_vertices):
        t, c = pred_vertices.shape
        loss = np.square(pred_vertices - target_vertices)  
        self.sum += np.sum(loss) 
        self.counter += t * c   

    def avg(self):
        if self.counter == 0:
            return 0  
        return self.sum / self.counter

    def reset(self):
        self.counter = 0
        self.sum = 0


class L1div(object):
    def __init__(self):
        self.counter = 0  
        self.sum = 0     

    def compute(self, results):
        self.counter += results.shape[0] 
        mean = np.mean(results, axis=0)  
        sum_l1 = np.sum(np.abs(results - mean), axis=None)  
        self.sum += sum_l1

    def avg(self):
        if self.counter == 0:
            return 0 
        return self.sum / self.counter

    def reset(self):
        self.counter = 0
        self.sum = 0


class SRGR(object):
    def __init__(self, threshold=0.1, joints=47, joint_dim=3):
        self.threshold = threshold
        self.pose_dimes = joints
        self.joint_dim = joint_dim
        self.counter = 0
        self.sum = 0

    def run(self, results, targets, semantic=None, verbose=False):
        if semantic is None:
            semantic = np.ones(results.shape[0])
            avg_weight = 1.0
        else:
            # srgr == 0.165 when all success, scale range to [0, 1]
            avg_weight = 0.165
        results = results.reshape(-1, self.pose_dimes, self.joint_dim)
        targets = targets.reshape(-1, self.pose_dimes, self.joint_dim)
        semantic = semantic.reshape(-1)
        diff = np.linalg.norm(results - targets, axis=2)  # T, J
        if verbose:
            print(diff)
        success = np.where(diff < self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            success[i, :] *= semantic[i] * (1 / avg_weight)
        rate = np.sum(success) / (success.shape[0] * success.shape[1])
        self.counter += success.shape[0]
        self.sum += rate * success.shape[0]
        return rate

    def avg(self):
        return self.sum / self.counter

    def reset(self):
        self.counter = 0
        self.sum = 0


class BC(object):
    def __init__(self, download_path=None, sigma=0.3, order=7, upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]):
        self.sigma = sigma
        self.order = order
        self.upper_body = upper_body
        self.pose_data = []
        if download_path is not None:
            os.makedirs(download_path, exist_ok=True)
            model_file_path = os.path.join(download_path, "mean_vel_smplxflame_30.npy")
            if not os.path.exists(model_file_path):
                print(f"Downloading {model_file_path}")
                wget.download(
                    "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/test_sequences/weights/mean_vel_smplxflame_30.npy",
                    model_file_path,
                )
        self.mmae = np.load(os.path.join(download_path, "mean_vel_smplxflame_30.npy")) if download_path is not None else None
        self.threshold = 0.10
        self.counter = 0
        self.sum = 0

    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        hop_length = 512
        if without_file:
            y = wave
        else:
            y, sr = librosa.load(wave, sr=sr_audio)
    
        short_y = y[t_start:t_end] if t_start is not None else y
        short_y = short_y.astype(np.float32)
        onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units="time")
        return onset_t

    def load_motion(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            data_each_file = pose
        else:
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432:
                        continue
                    line_data_np = np.fromstring(line_data, sep=" ")
                    if pose_fps == 15 and i % 2 == 0:
                        continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121]], 0))
            data_each_file = np.array(data_each_file)  # T*165
        # print(data_each_file.shape)
        joints = data_each_file.transpose(1, 0)
        dt = 1 / pose_fps
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        # print(vel.shape)

        if self.mmae is not None:
            vel = np.linalg.norm(vel, axis=2) / self.mmae
        else:
            print("Warning: mmae is not provided, using max value of vel as mmae")
            self.mmae = np.linalg.norm(vel, axis=2).max()
            vel = np.linalg.norm(vel, axis=2) / self.mmae
        # print(vel.shape) # T*J

        beat_vel_all = []
        for i in range(vel.shape[1]):
            vel_mask = np.where(vel[:, i] > self.threshold)
            beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order)
            beat_vel_list = [j for j in beat_vel[0] if j in vel_mask[0]]
            beat_vel_all.append(np.array(beat_vel_list))
        return beat_vel_all

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_vel_all = self.load_motion(pose, i, i + dur, pose_fps)
            dis_all_b2a = self.compute(onset_raw, beat_vel_all)
            print(f"{i}s: ", dis_all_b2a)

    @staticmethod
    def plot_onsets(audio, sr, onset_times_1, onset_times_2):
        fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[0])
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[1])

        for onset in onset_times_1:
            axarr[0].axvline(onset, color="r", linestyle="--", alpha=0.9, label="Onset Method 1")
        axarr[0].legend()
        axarr[0].set(title="Onset Method 1", xlabel="", ylabel="Amplitude")

        for onset in onset_times_2:
            axarr[1].axvline(onset, color="b", linestyle="-", alpha=0.7, label="Onset Method 2")
        axarr[1].legend()
        axarr[1].set(title="Onset Method 2", xlabel="Time (s)", ylabel="Amplitude")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("Audio waveform with Onsets")
        plt.savefig("./onset.png", dpi=500)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max), y_axis="log", x_axis="time", ax=ax[0])
        ax[1].plot(self.times, self.oenv, label="Onset strength")
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label="Raw onsets", color="r")
        ax[1].legend()
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label="Backtracked", color="r")
        ax[2].legend()
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label="Backtracked (RMS)", color="r")
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)

    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        return vel / pose_fps + offset

    @staticmethod
    def GAHR(a, b, sigma):
        # Speech clips can legitimately contain no detected audio onset.
        # GAHR has no alignment evidence in that case, so score it as zero,
        # matching the TalkSHOW/SemTalk offline evaluator's empty-beat rule.
        if len(a) == 0 or len(b) == 0:
            return 0.0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = min(abs(a_each - b_each) for a_each in a)
            dis_all_b2a += math.exp(-(l2_min**2) / (2 * sigma**2))
        return dis_all_b2a / len(b)

    @staticmethod
    def fix_directed_GAHR(a, b, sigma):
        a = BC.motion_frames2time(a, 0, 30)
        b = BC.motion_frames2time(b, 0, 30)
        a = [0] + a + [len(a) / 30]
        b = [0] + b + [len(b) / 30]
        return BC.GAHR(a, b, sigma)

    def compute(self, onset_bt_rms, beat_vel, length=1, pose_fps=30):
        avg_dis_all_b2a_list = []
        for its, beat_vel_each in enumerate(beat_vel):
            if its not in self.upper_body:
                continue
            if beat_vel_each.size == 0:
                avg_dis_all_b2a_list.append(0)
                continue
            pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
            avg_dis_all_b2a_list.append(self.GAHR(pose_bt, onset_bt_rms, self.sigma))
        self.sum += (sum(avg_dis_all_b2a_list) / len(self.upper_body)) * length
        self.counter += length

    def avg(self):
        return self.sum / self.counter if self.counter > 0 else 0.0

    def reset(self):
        self.counter = 0
        self.sum = 0


class Arg(object):
    def __init__(self):
        self.vae_length = 240
        self.vae_test_dim = 330
        self.vae_test_len = 32
        self.vae_layer = 4
        self.vae_test_stride = 20
        self.vae_grow = [1, 1, 2, 1]
        self.variational = False


class FGD(object):
    def __init__(self, download_path="./emage/", device="cuda", feature_offload=False):
        if download_path is not None:
            os.makedirs(download_path, exist_ok=True)
            model_file_path = os.path.join(download_path, "AESKConv_240_100.bin")
            smplx_model_dir = os.path.join(download_path, "smplx_models", "smplx")
            smplx_model_file_path = os.path.join(smplx_model_dir, "SMPLX_NEUTRAL_2020.npz")
            if not os.path.exists(model_file_path):
                print(f"Downloading {model_file_path}")
                wget.download(
                    "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/test_sequences/weights/AESKConv_240_100.bin",
                    model_file_path,
                )

            os.makedirs(smplx_model_dir, exist_ok=True)
            if not os.path.exists(smplx_model_file_path):
                print(f"Downloading {smplx_model_file_path}")
                wget.download(
                    "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
                    smplx_model_file_path,
                )
        args = Arg()
        self.eval_model = VAESKConv(args, model_save_path=download_path)  # Assumes LocalEncoder is defined elsewhere
        old_stat = torch.load(download_path + "AESKConv_240_100.bin")["model_state"]
        new_stat = {}
        for k, v in old_stat.items():
            # If 'module.' is in the key, remove it
            new_key = k.replace("module.", "") if "module." in k else k
            new_stat[new_key] = v
        self.eval_model.load_state_dict(new_stat)

        self.eval_model.eval()
        if torch.cuda.is_available():
            self.eval_model.to(device)

        self.device = device
        self.feature_offload = feature_offload
        self.feat_dim = None # 用于 offload 模式记录特征维度
        self.feat_dtype = None
        self._init_storage()
    
    def _init_storage(self):
        """内部函数：根据模式初始化存储"""
        if self.feature_offload:
            # === 硬盘卸载模式 ===
            self.temp_dir = tempfile.mkdtemp(prefix="emage_fgd_cache_")
            self.pred_file_path = os.path.join(self.temp_dir, "pred_features.bin")
            self.target_file_path = os.path.join(self.temp_dir, "target_features.bin")
            
            # 以 'wb' 模式打开，准备写入
            self.f_pred = open(self.pred_file_path, "wb")
            self.f_target = open(self.target_file_path, "wb")
            
            self.pred_row_count = 0
            self.target_row_count = 0
        else:
            # === 内存列表模式 (原有逻辑) ===
            self.pred_features = []
            self.target_features = []

    def reset(self):
        """重置评估器状态"""
        if self.feature_offload:
            # 关闭文件句柄
            if hasattr(self, 'f_pred') and not self.f_pred.closed:
                self.f_pred.close()
            if hasattr(self, 'f_target') and not self.f_target.closed:
                self.f_target.close()
            
            # 重新打开清空文件
            self.f_pred = open(self.pred_file_path, "wb")
            self.f_target = open(self.target_file_path, "wb")
            self.pred_row_count = 0
            self.target_row_count = 0
        else:
            self.pred_features = []
            self.target_features = []

    def get_feature(self, data):
        assert len(data.shape) == 3
        if data.shape[1] % 32 != 0:
            drop_len = data.shape[1] % 32 
            data = data[:, :-drop_len]
            # print(data.shape)
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.to(self.device)
            feature = self.eval_model.map2latent(data).cpu().numpy()
        # print(feature.shape)
        return feature

    def update(self, pred, target):
        if self.feature_offload:
            # === 硬盘卸载模式 ===
            feat_pred = self.get_feature(pred)
            feat_target = self.get_feature(target)

            # 记录特征维度 (仅第一次)
            if self.feat_dim is None:
                self.feat_dim = feat_pred.shape[-1]
                self.feat_dtype = feat_pred.dtype # 动态检测 float32 还是 float64
            
            # 3. 复刻原有 compute 中的 reshape 逻辑
            # 原有逻辑: x.reshape(-1, x.shape[-1])
            # 我们在这里提前执行这一步，确保写入磁盘的数据结构与原逻辑最终合并的数据结构一致
            feat_pred_reshaped = feat_pred.reshape(-1, feat_pred.shape[-1])
            feat_target_reshaped = feat_target.reshape(-1, feat_target.shape[-1])
            
            # 4. 写入硬盘 (使用原始 dtype)
            feat_pred_reshaped.tofile(self.f_pred)
            feat_target_reshaped.tofile(self.f_target)
            
            self.pred_row_count += feat_pred_reshaped.shape[0]
            self.target_row_count += feat_target_reshaped.shape[0]
            
            del feat_pred, feat_target, feat_pred_reshaped, feat_target_reshaped
        else:
            self.pred_features.append(self.get_feature(pred))
            self.target_features.append(self.get_feature(target))

    def compute(self):
        if self.feature_offload:
            # === 硬盘卸载模式 ===
            self.f_pred.flush()
            self.f_target.flush()
            
            if self.pred_row_count == 0 or self.target_row_count == 0:
                raise ValueError(f"No features to compute FGD, " \
                    f"pred_row_count: {self.pred_row_count}, target_row_count: {self.target_row_count}")

            # 建立内存映射 (不占实际 RAM)
            pred_features = np.memmap(
                self.pred_file_path, dtype=self.feat_dtype, mode='r', 
                shape=(self.pred_row_count, self.feat_dim)
            )
            target_features = np.memmap(
                self.target_file_path, dtype=self.feat_dtype, mode='r', 
                shape=(self.target_row_count, self.feat_dim)
            )
            
            # 计算完成后，memmap 会自动释放对文件的锁定
            score = self.frechet_distance(pred_features, target_features)
            
            # 手动解除引用
            del pred_features, target_features
            return score
        else:
            pred_features = np.concatenate([x.reshape(-1, x.shape[-1]) for x in self.pred_features], axis=0)
            target_features = np.concatenate([x.reshape(-1, x.shape[-1]) for x in self.target_features], axis=0)
            # print(pred_features.shape, target_features.shape)
            return self.frechet_distance(pred_features, target_features)

    @staticmethod
    def frechet_distance(samples_A, samples_B, eps=1e-6):
        mu1 = np.mean(samples_A, axis=0)
        sigma1 = np.cov(samples_A, rowvar=False)
        mu2 = np.mean(samples_B, axis=0)
        sigma2 = np.cov(samples_B, rowvar=False)
        diff = mu1 - mu2
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    
    def __del__(self):
        """清理临时文件 (仅在卸载模式下生效)"""
        if self.feature_offload:
            # 关闭文件
            if hasattr(self, 'f_pred') and hasattr(self.f_pred, 'close'):
                self.f_pred.close()
            if hasattr(self, 'f_target') and hasattr(self.f_target, 'close'):
                self.f_target.close()
            
            # 删除临时目录
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp dir {self.temp_dir}: {e}")
