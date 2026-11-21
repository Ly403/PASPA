import os
from tqdm import tqdm
import torch
from diffusion.metrics.utils.motion_rep_transfer import get_motion_rep_numpy, get_motion_rep_tensor
import diffusion.utils.rotation_conversions as rc
from diffusion.metrics.frechet_audio_distance.fad import FrechetAudioDistance
import numpy as np
import librosa
import re
from diffusion.metrics.emage_evaltools import FGD, BC, L1div, LVDFace, MSEFace
from accelerate.utils import gather_object
from abc import ABC, abstractmethod
from scipy import linalg
import torch.distributed as dist

class Metric(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        """更新状态，用于累积计算指标"""
        pass
    
    @abstractmethod
    def compute(self, *args, **kwargs):
        """计算指标并返回结果"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置状态"""
        pass
    
    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
    
    def cpu(self):
        pass
    
    def cuda(self):
        pass
    
class EmageMetric(Metric):
    def __init__(
        self,
        download_path,
        smplx,
        accelerator,
        fad_config,
        audio_sr,
        pose_fps,
        evaluate_on_disk=False,
        keep_every_epoch=False,
        use_audio=True,
        device="cuda",
        fgd_feature_offload=False,
    ):
        # emage_eval_tools = importlib.import_module(emage_eval_tools_path)
        self.fgd_evaluator = FGD(
            download_path=download_path, device=device, feature_offload=fgd_feature_offload
        )
        self.bc_evaluator = BC(
            download_path=download_path,
        )
        self.l1div_evaluator = L1div()
        self.lvd_evaluator = LVDFace()
        self.mse_evaluator = MSEFace()
        self.use_audio = use_audio
        if use_audio:
            self.fad_evaluator = FrechetAudioDistance(
                **fad_config
                # model_name="vggish",
                # sample_rate=sample_rate,
                # use_pca=False, 
                # use_activation=False,
                # verbose=False,
                # local_load_path=local_load_fad_path
            )
        self.evaluate_on_disk = evaluate_on_disk
        self.keep_every_epoch = keep_every_epoch
        self.device = device
        self.smplx = smplx
        self.accelerator = accelerator
        self.audio_sr = audio_sr
        self.pose_fps = pose_fps
       
    def cpu(self):
        self.fgd_evaluator.eval_model.to("cpu")
    
    def cuda(self):
        self.fgd_evaluator.eval_model.to(self.device)
    
    def reset(self):
        self.fgd_evaluator.reset()
        self.bc_evaluator.reset()
        self.l1div_evaluator.reset()
        self.lvd_evaluator.reset()
        self.mse_evaluator.reset()
        if self.use_audio:
            self.fad_evaluator.reset()
    
    @torch.no_grad()
    def update(self, results):
        return results
    
    def compute(self, results):
        """
        results: 当前 Rank 上的结果字典
        """
        gathered_results_list = gather_object([results])
        if self.accelerator.is_main_process:
            final_results = {}
            ref_key = 'gt_motion' 
            
            if len(gathered_results_list) > 0:
                for k in gathered_results_list[0].keys():
                    final_results[k] = []

            # 遍历每个 Rank 的字典
            for rank_res in gathered_results_list:
                if not rank_res or ref_key not in rank_res:
                    raise ValueError(f"Invalid rank_res: {rank_res}, which has no {ref_key} data.")
                num_samples = len(rank_res[ref_key])
                
                for i in range(num_samples):
                    for k in final_results.keys():
                        if k in rank_res:
                            final_results[k].append(rank_res[k][i])
            results = final_results
            print(results.keys())
            print(len(results["gt_motion"]))
            
            self.reset()    
            gt_list = results["gt_motion"]
            pred_list = results["motion"]
            pred_audio_list = results["audio"] if self.use_audio else results["gt_audio"]
            gt_audio_list = results["gt_audio"]
            pred_expression_list = results["expression"]
            gt_expression_list = results["gt_expression"]
            betas_list = results["betas"]
            
            for (gt_motion, gt_expression, gt_audio, pred_motion, pred_expression, betas) in tqdm(zip(
                    gt_list, gt_expression_list, gt_audio_list, pred_list, pred_expression_list, betas_list), desc="Evaluation", disable=os.getenv(
                    "DPM_TQDM", "True") == "True"):
                
                if self.evaluate_on_disk:
                    gt_dict = np.load(gt_motion)
                    pred_dict = np.load(pred_motion)
                    gt_motion = gt_dict["poses"]
                    gt_expression = gt_dict["expressions"]
                    pred_motion = pred_dict["poses"]
                    pred_expression = pred_dict["expressions"]
                    betas = gt_dict["betas"]
                    
                    gt_match = re.search("sr(\d+)_", gt_audio)
                    if gt_match:
                        gt_audio_sr = int(gt_match.group(1))
                    else:
                        raise ValueError(f"Invalid gt_audio: {gt_audio}")
                    
                    gt_audio = librosa.load(gt_audio, sr=gt_audio_sr)[0]
                
                
                ########################### 下面这段逻辑是EMAGE算motion相关指标的 #######################################
                t = gt_motion.shape[0]
                # bc and l1 require position representation
                motion_position_pred = get_motion_rep_numpy(
                    self.smplx, pred_motion, device=self.device, betas=betas)["position"] # t*55*3
                motion_position_pred = motion_position_pred.reshape(t, -1)
                # ignore the start and end 2s, this may for beat dataset only
                # audio_beat = self.bc_evaluator.load_audio(
                #     gt_audio, t_start=2 * 16000, t_end=int((t-60)/30*16000),
                #     without_file=True
                # ) # TODO，这里需要修改，因为gt_audio要用wav，最好是未经过mel转换的，这里这个gt_audio实际上先转mel，再转wav的
                # motion_beat = self.bc_evaluator.load_motion(
                #     motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, 
                #     without_file=True
                # )
                # self.bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)
                audio_beat = self.bc_evaluator.load_audio(
                    gt_audio, t_start=0 * self.audio_sr, t_end=int((t-0)/self.pose_fps*self.audio_sr), without_file=True)
                motion_beat = self.bc_evaluator.load_motion(
                    motion_position_pred, t_start=0, t_end=t-0, pose_fps=self.pose_fps, without_file=True)
                self.bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=self.pose_fps)
                

                self.l1div_evaluator.compute(motion_position_pred)
                # audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=0 * 16000, t_end=int((t-0)/30*16000))
                # motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=0, t_end=t-0, pose_fps=30, without_file=True)
                # bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=30)

                # self.l1_evaluator.compute(motion_position_pred)
            
                face_position_pred = get_motion_rep_numpy(
                    self.smplx, 
                    pred_motion, 
                    device=self.device, 
                    expressions=pred_expression, 
                    expression_only=True, 
                    betas=betas
                )["vertices"] # t -1
                face_position_gt = get_motion_rep_numpy(
                    self.smplx,
                    gt_motion,
                    device=self.device,
                    expressions=gt_expression, 
                    expression_only=True, 
                    betas=betas
                )["vertices"]
                self.lvd_evaluator.compute(face_position_pred, face_position_gt)
                self.mse_evaluator.compute(face_position_pred, face_position_gt)
            
                # fgd requires rotation 6d representaiton
                gt_motion = torch.from_numpy(gt_motion).to(self.device).unsqueeze(0)
                pred_motion = torch.from_numpy(pred_motion).to(self.device).unsqueeze(0)
                gt_motion = rc.axis_angle_to_rotation_6d(gt_motion.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
                pred_motion = rc.axis_angle_to_rotation_6d(pred_motion.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
                self.fgd_evaluator.update(pred_motion.float(), gt_motion.float())
                ########################### 上面这段逻辑是EMAGE算motion相关指标的 #######################################

                ########################### 下面需要写关于audio的指标 #######################################
                # 暂放，TODO
                ###########################################################################################
            
            metrics = {}
            metrics["fgd"] = self.fgd_evaluator.compute()
            metrics["bc"] = self.bc_evaluator.avg()
            metrics["l1"] = self.l1div_evaluator.avg()
            metrics["lvd"] = self.lvd_evaluator.avg()
            metrics["mse"] = self.mse_evaluator.avg()
            if self.use_audio:
                metrics["fad"] = self.fad_evaluator.score(
                    background_data=np.stack(gt_audio_list, axis=0),
                    eval_data=np.stack(pred_audio_list, axis=0),
                    background_embds_path=None,
                    eval_embds_path=None,
                    dtype="float32"
                )
        else:
            metrics = {}
        self.accelerator.wait_for_everyone()
        return metrics


class DistributedEmageMetric(Metric):
    """
        完全为加速写的类，用在validation的过程中，注意：这个指标因为是运行中计算，难以保证完全准确，主要做个参考
        完全准确的指标应该通过独立的脚本计算，因为计算量太大，不能在validation过程中计算
    """
    def __init__(
        self,
        download_path,
        smplx,
        accelerator,
        audio_sr,
        pose_fps,
        device="cuda",
        fgd_feature_offload=False,
    ):
        # 暂不支持FAD
        # emage_eval_tools = importlib.import_module(emage_eval_tools_path)
        self.fgd_evaluator = FGD(download_path=download_path, device=device, feature_offload=fgd_feature_offload)
        self.bc_evaluator = BC(download_path=download_path)
        self.l1div_evaluator = L1div()
        self.lvd_evaluator = LVDFace()
        self.mse_evaluator = MSEFace()
        self.device = device
        self.smplx = smplx
        self.accelerator = accelerator
        self.audio_sr = audio_sr
        self.pose_fps = pose_fps
    
    def cpu(self):
        self.fgd_evaluator.eval_model.to("cpu")
    
    def cuda(self):
        self.fgd_evaluator.eval_model.to(self.device)
        
    def reset(self):
        self.fgd_evaluator.reset()
        self.bc_evaluator.reset()
        self.l1div_evaluator.reset()
        self.lvd_evaluator.reset()
        self.mse_evaluator.reset()
    
    @torch.no_grad()
    def update(self, results):
        pred_motion_pts = results.pop("motion")
        gt_motion_pts = results.pop("gt_motion")
        pred_expression_pts = results.pop("expression")
        gt_expression_pts = results.pop("gt_expression")
        betas_pts = results.pop("betas")
        
        gt_audios = results.pop("gt_audio").detach().cpu().numpy()
        
        motion_position_preds = get_motion_rep_tensor(
            self.smplx, pred_motion_pts, device=self.device, betas=betas_pts)["position"].detach().cpu().numpy()
        face_position_preds = get_motion_rep_tensor(
            self.smplx, 
            pred_motion_pts, 
            device=self.device, 
            expressions=pred_expression_pts, 
            expression_only=True, 
            betas=betas_pts
        )["vertices"].detach().cpu().numpy()
        face_position_gts = get_motion_rep_tensor(
            self.smplx,
            gt_motion_pts,
            device=self.device,
            expressions=gt_expression_pts, 
            expression_only=True, 
            betas=betas_pts
        )["vertices"].detach().cpu().numpy()
        n = pred_motion_pts.shape[0]
        t = pred_motion_pts.shape[1]
    
    
        gt_motion_pts = rc.axis_angle_to_rotation_6d(gt_motion_pts.reshape(n, t, 55, 3)).reshape(n, t, 55*6)
        pred_motion_pts = rc.axis_angle_to_rotation_6d(pred_motion_pts.reshape(n, t, 55, 3)).reshape(n, t, 55*6)
        self.fgd_evaluator.update(pred_motion_pts.float(), gt_motion_pts.float())
        for gt_audio, motion_position_pred, face_position_pred, face_position_gt \
            in zip(gt_audios, motion_position_preds, face_position_preds, face_position_gts):
        
            ########################### 下面这段逻辑是EMAGE算motion相关指标的 #######################################
            motion_position_pred = motion_position_pred.reshape(t, -1)
            audio_beat = self.bc_evaluator.load_audio(
                gt_audio, t_start=0 * self.audio_sr, t_end=int((t-0)/self.pose_fps*self.audio_sr), without_file=True)
            motion_beat = self.bc_evaluator.load_motion(
                motion_position_pred, t_start=0, t_end=t-0, pose_fps=self.pose_fps, without_file=True)
            self.bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=self.pose_fps)
            

            self.l1div_evaluator.compute(motion_position_pred)
            self.lvd_evaluator.compute(face_position_pred, face_position_gt)
            self.mse_evaluator.compute(face_position_pred, face_position_gt)
        
            # # fgd requires rotation 6d representaiton
            # gt_motion_pt = gt_motion_pt.unsqueeze(0)
            # pred_motion_pt = pred_motion_pt.unsqueeze(0)
            
            # self.fgd_evaluator.update(pred_motion_pt.float(), gt_motion_pt.float())
        
        return results
    
    @staticmethod
    def _calculate_fd_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        核心矩阵运算逻辑，与 FGD.frechet_distance 中的后半部分完全一致。
        """
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        # 检查是否为有限值，进行数值稳定处理
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        # Frechet Distance 公式
        # FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    
    
    def compute(self, all_results):
        # 不能在开头reset，这个不同，应该在最后reset，且所有进程都要做
        # 需要主进程收集所有结果，然后计算指标，这里需要做个进程同步
        # 该类没有东西需要通过all_results计算，他进来的时候理应是个空字典，这样写只是为了兼容性
        # 因为有些类，比如上面那个EmageMetric，他需要通过all_results计算指标，然后返回给主进程
        # trainer忽视这些冲突
        # 指标计算只分两类，一类update，一类最后通过all_results里面的结果计算，update的结果最后也要处理
        # 1. Gather 标量统计量 (BC, L1, LVD, MSE)
        # ... (保持不变) ...
        bc_stats = gather_object([(self.bc_evaluator.sum, self.bc_evaluator.counter)])
        l1_stats = gather_object([(self.l1div_evaluator.sum, self.l1div_evaluator.counter)])
        lvd_stats = gather_object([(self.lvd_evaluator.sum, self.lvd_evaluator.counter)])
        mse_stats = gather_object([(self.mse_evaluator.sum, self.mse_evaluator.counter)])

        # =========================================================
        # 2. FGD Two-Pass 计算 (核心修改)
        # =========================================================
        
        # 通用数据读取器 (兼容 Offload 和 Memory)
        def get_data_reader(is_pred):
            if self.fgd_evaluator.feature_offload:
                if is_pred:
                    if hasattr(self.fgd_evaluator, 'f_pred'): self.fgd_evaluator.f_pred.flush()
                    file_path = self.fgd_evaluator.pred_file_path
                    row_count = self.fgd_evaluator.pred_row_count
                else:
                    if hasattr(self.fgd_evaluator, 'f_target'): self.fgd_evaluator.f_target.flush()
                    file_path = self.fgd_evaluator.target_file_path
                    row_count = self.fgd_evaluator.target_row_count
                
                dim, dtype = self.fgd_evaluator.feat_dim, self.fgd_evaluator.feat_dtype
                if row_count == 0 or dim is None: return None, 0, 0
                return np.memmap(file_path, dtype=dtype, mode='r', shape=(row_count, dim)), row_count, dim
            else:
                feature_list = self.fgd_evaluator.pred_features if is_pred else self.fgd_evaluator.target_features
                if not feature_list: return None, 0, 0
                # 即使是 float32 数据，拼接成 numpy 也没问题
                data = np.concatenate([x.reshape(-1, x.shape[-1]) for x in feature_list], axis=0)
                return data, data.shape[0], data.shape[1]

        def compute_stats_two_pass(is_pred):
            """
            使用 Two-Pass 算法计算均值和协方差，数值稳定性等同于 np.cov
            即使输入是 float32，累加过程必须强制使用 float64
            """
            data_source, N_local, dim = get_data_reader(is_pred)
            if N_local == 0:
                return None, None, torch.tensor(0.0, device=self.device, dtype=torch.float64)

            # --- Pass 1: 计算局部 Sum (用于求 Global Mean) ---
            local_sum = torch.zeros(dim, device=self.device, dtype=torch.float64) # 强制 double
            chunk_size = 5000
            for i in range(0, N_local, chunk_size):
                end = min(i + chunk_size, N_local)
                chunk_np = data_source[i:end]
                # 转 Tensor -> GPU -> Double
                chunk = torch.from_numpy(chunk_np).to(self.device).double()
                local_sum += torch.sum(chunk, dim=0)
            
            # AllReduce Sum 和 N
            global_sum = local_sum.clone()
            dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
            
            global_N = torch.tensor(N_local, device=self.device, dtype=torch.float64)
            dist.all_reduce(global_N, op=dist.ReduceOp.SUM)
            
            # 得到精确的全局均值
            global_mean = global_sum / global_N
            
            # --- Pass 2: 计算局部中心化协方差 Sum((x-mu)(x-mu)^T) ---
            # 这里的数学意义是：(X - global_mean)^T (X - global_mean)
            local_centered_cov_sum = torch.zeros((dim, dim), device=self.device, dtype=torch.float64) # 强制 double
            
            for i in range(0, N_local, chunk_size):
                end = min(i + chunk_size, N_local)
                chunk_np = data_source[i:end]
                chunk = torch.from_numpy(chunk_np).to(self.device).double()
                
                # 关键：先减去全局均值 (Centering)
                centered = chunk - global_mean 
                # 再计算外积累加
                local_centered_cov_sum += centered.t() @ centered
            
            # 清理 memmap
            if self.fgd_evaluator.feature_offload: del data_source
            
            # AllReduce 协方差 sum
            global_cov_sum = local_centered_cov_sum.clone()
            dist.all_reduce(global_cov_sum, op=dist.ReduceOp.SUM)
            
            # 最终 Sigma = Sum / (N-1)
            sigma = global_cov_sum / (global_N - 1)
            
            return global_mean.cpu().numpy(), sigma.cpu().numpy(), global_N

        # 执行 Two-Pass 计算
        mu_p, sigma_p, N_p = compute_stats_two_pass(is_pred=True)
        mu_t, sigma_t, N_t = compute_stats_two_pass(is_pred=False)

        # 5. Final Compute
        metric = {}
        if self.accelerator.is_main_process:
            # ... (标量指标合并逻辑保持不变) ...
            def reduce_avg(stats_list):
                total_sum = sum(item[0] for item in stats_list)
                total_cnt = sum(item[1] for item in stats_list)
                return total_sum / total_cnt if total_cnt > 0 else 0.0

            # FGD
            if mu_p is not None and mu_t is not None:
                metric["fgd"] = self._calculate_fd_from_stats(mu_p, sigma_p, mu_t, sigma_t)
            else:
                metric["fgd"] = 0.0
            metric["bc"] = reduce_avg(bc_stats)
            metric["l1"] = reduce_avg(l1_stats)
            metric["lvd"] = reduce_avg(lvd_stats)
            metric["mse"] = reduce_avg(mse_stats)

        self.accelerator.wait_for_everyone()
        self.reset()
        return metric