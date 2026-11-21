import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
import lmdb 
import pickle 
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

class AudioFeatureExtractor:
    def __init__(
        self, 
        model_type="wavlm",
        model_id=None,
    ):
        super().__init__()
        if not model_id:
            raise ValueError("model_id must point to a local audio model")
        if model_type == "wavlm":
            from transformers import Wav2Vec2Processor, WavLMModel
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
        elif model_type == "hubert":
            try:
                from modelscope import AutoProcessor, HubertModel
            except:
                from transformers import AutoProcessor, HubertModel
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = HubertModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def __call__(self, raw_audio, seq_size=None):
        inputs = self.processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            features = self.model(inputs).last_hidden_state
            if seq_size != None:
                features = F.interpolate(
                    features.permute((0, 2, 1)), seq_size, mode='linear', align_corners=True)\
                        .permute((0, 2, 1))
        return features.cpu().numpy()

SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]

def LocalToGlobalAA(local_aa, parent_idx):
    N = local_aa.shape[0]
    nJoints = len(parent_idx)
    local_rot = R.from_rotvec(local_aa.reshape(-1, 3)).as_matrix().reshape((N, nJoints, 3, 3))
    global_rot = np.zeros_like(local_rot)
    for i in range(nJoints):
        if parent_idx[i] == -1:
            global_rot[:, i] = local_rot[:, i]
        else:
            global_rot[:, i] = np.einsum('nij,njk->nik', global_rot[:, parent_idx[i]], local_rot[:, i])
    global_aa = R.from_matrix(global_rot.reshape(-1, 3, 3)).as_rotvec().reshape(N, -1)
    return global_aa

def parse_train_test_split_csv(csv_file):
    tables = pd.read_csv(csv_file)
    file_ids = tables['id']
    file_types = tables['type']
    train_files = []
    test_files =[]
    val_files = []
    additional_files = []
    for n in range(len(file_ids)):
        if file_types[n] == 'train':
            train_files.append(file_ids[n])
        elif file_types[n] == 'test':
            test_files.append(file_ids[n])
        elif file_types[n] == 'val':
            val_files.append(file_ids[n])
        elif file_types[n] == 'additional':
            additional_files.append(file_ids[n])
    return {
        "train": train_files,
        "test": test_files,
        "val": val_files,
        "additional": additional_files,
    }

# -----------------------------------------------------------------------------
# 核心修改函数
# -----------------------------------------------------------------------------
def create_lmdb(data_folder, lmdb_path, seq_size, stride_size, 
                local_rotation=True, split='train', model_type="wavlm", model_id=""):
    
    smplx_data_folder = os.path.join(data_folder, "smplxflame_30")
    audio_data_folder = os.path.join(data_folder, "wave16k")
    
    if split =='all':
        npz_files = glob(os.path.join(smplx_data_folder, "*.npz"))
    else:
        split_results = parse_train_test_split_csv(os.path.join(data_folder, "train_test_split.csv"))
        npz_files = [os.path.join(smplx_data_folder, f + ".npz") for f in split_results[split]]

    audio_feature_extractor = AudioFeatureExtractor(model_type=model_type, model_id=model_id)

    # 开辟 1TB 映射空间
    env = lmdb.open(lmdb_path, map_size=1024**4, subdir=False)
    txn = env.begin(write=True)

    dp_id = 0
    commit_interval = 500

    for npz_file in tqdm(npz_files):
        # 基础信息处理（同原逻辑）
        pid = int(os.path.basename(npz_file).split('_')[0])
        audio_file = os.path.join(audio_data_folder, os.path.basename(npz_file).replace(".npz", ".wav"))
        
        try:
            raw_audio, sr = librosa.load(audio_file, sr=16000)
            raw_audio = raw_audio.astype(np.float32)
            smplx_data = np.load(npz_file, allow_pickle=True)
            betas = smplx_data["betas"].astype(np.float32)
            poses = smplx_data["poses"].astype(np.float32)
            expressions = smplx_data["expressions"].astype(np.float32)
            trans = smplx_data["trans"].astype(np.float32)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        # 准备数据 List
        data_chunks = []
        file_name = os.path.splitext(os.path.basename(npz_file))[0]

        if split != "test":
            num_frames = min(len(poses), int(len(raw_audio)/sr * 30))
            num_windows = (num_frames - seq_size) // stride_size + 1
            if num_windows <= 0: continue

            poses_lst = [poses[i*stride_size : i*stride_size+seq_size] for i in range(num_windows)]
            if not local_rotation:
                poses_lst = [LocalToGlobalAA(p, SMPLX_PARENTS).astype(np.float32) for p in poses_lst]
            
            expressions_lst = [expressions[i*stride_size : i*stride_size+seq_size] for i in range(num_windows)]
            trans_lst = [trans[i*stride_size : i*stride_size+seq_size] for i in range(num_windows)]
            
            audio_stride = seq_size * sr // 30
            audio_lst = [raw_audio[(i*stride_size)*sr//30 : (i*stride_size)*sr//30 + audio_stride] for i in range(num_windows)]

            for i in range(num_windows):
                if len(poses_lst[i]) != seq_size or len(audio_lst[i]) != sr * seq_size // 30: continue
                huberts = audio_feature_extractor(audio_lst[i], seq_size=seq_size)
                
                data_chunks.append({
                    "betas": betas, # betas 形状通常是 (10,) 或 (1, 10)，这里存原始的即可
                    "poses": poses_lst[i],
                    "expressions": expressions_lst[i],
                    "trans": trans_lst[i],
                    "audio": audio_lst[i],
                    "huberts": huberts,
                })
        else:
            # Test 模式逻辑
            seq_size_real = len(poses)
            target_audio_len = (16000 * seq_size_real) // 30
            if len(raw_audio) > target_audio_len: raw_audio = raw_audio[:target_audio_len]
            elif len(raw_audio) < target_audio_len: raw_audio = np.pad(raw_audio, (0, target_audio_len - len(raw_audio)), mode='constant')
            huberts = audio_feature_extractor(raw_audio, seq_size=seq_size_real)
            
            data_chunks.append({
                "betas": betas,
                "poses": poses,
                "expressions": expressions,
                "trans": trans,
                "audio": raw_audio,
                "huberts": huberts,
            })

        # --- 核心写入逻辑 ---
        for chunk in data_chunks:
            key_base = f"{dp_id:010d}"
            
            # 1. 构造 Meta 数据 (存 scalar 和 shapes)
            # 注意：betas通常对于同一个 ID 是一样的，但为了 dataset 简单，我们还是每条都存
            meta = {
                "pid": pid,
                "ori_vid": file_name,
                "shape_betas": chunk["betas"].shape,
                "shape_poses": chunk["poses"].shape,
                "shape_expressions": chunk["expressions"].shape,
                "shape_trans": chunk["trans"].shape,
                "shape_audio": chunk["audio"].shape,
                "shape_huberts": chunk["huberts"].shape,
                # 记录 dtype 也是好习惯，虽然我们默认 float32
            }
            txn.put(f"{key_base}_meta".encode('ascii'), pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

            # 2. 存大数组 (Zero-Copy 准备：转 bytes)
            # 使用 float32 保证精度并节省空间（相对于 float64）
            txn.put(f"{key_base}_betas".encode('ascii'), chunk["betas"].tobytes())
            txn.put(f"{key_base}_poses".encode('ascii'), chunk["poses"].tobytes())
            txn.put(f"{key_base}_expressions".encode('ascii'), chunk["expressions"].tobytes())
            txn.put(f"{key_base}_trans".encode('ascii'), chunk["trans"].tobytes())
            txn.put(f"{key_base}_audio".encode('ascii'), chunk["audio"].tobytes())
            txn.put(f"{key_base}_huberts".encode('ascii'), chunk["huberts"].tobytes())
            
            dp_id += 1
            if dp_id % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.put('__len__'.encode('ascii'), str(dp_id).encode('ascii'))
    txn.commit()
    env.close()
    print(f"Optimized LMDB created at {lmdb_path} with {dp_id} samples.")

def create():
    parser = argparse.ArgumentParser(description="Create Optimized LMDB dataset")
    parser.add_argument("data_folder", type=str, help="Path to the smplxflame_30")
    parser.add_argument("--seq_size", type=int, default=60, help="Sequence size")
    parser.add_argument("--stride_size", type=int, default=20, help="Stride size")
    parser.add_argument("--lmdb_path", type=str, default="", help="Path to save")
    parser.add_argument("--local_rotation", type=int, default=0, help="1 or 0")
    parser.add_argument("--split", type=str, default="train", choices=['all', 'train', 'test', 'val', 'additional'])
    parser.add_argument("--model_type", type=str, default="wavlm", choices=['wavlm', 'hubert'])
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Local Hugging Face audio-model directory",
    )
    args = parser.parse_args()

    if args.lmdb_path == "":
        local_global = "local" if args.local_rotation!=0 else "global"
        args.lmdb_path = f"./{args.split}_seq_size_{args.seq_size}_stride_size_{args.stride_size}_{local_global}_optimized.lmdb"

    os.makedirs(os.path.dirname(args.lmdb_path) if os.path.dirname(args.lmdb_path) else '.', exist_ok=True)

    create_lmdb(
        data_folder=args.data_folder, 
        lmdb_path=args.lmdb_path, 
        seq_size=args.seq_size, 
        stride_size=args.stride_size,
        local_rotation=args.local_rotation!=0,
        split=args.split,
        model_type=args.model_type,
        model_id=args.model_id
    )
if __name__ == "__main__":
    create()
