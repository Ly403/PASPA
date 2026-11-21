import os
import numpy as np
import torch
from torch.utils.data import Dataset
from diffusion.utils.rotation_conversions import axis_angle_to_rotation_6d
from scipy.io import loadmat
from transformers import Wav2Vec2Processor, WavLMModel
import librosa
from diffusion.utils.rotation_conversions import LocalToGlobalAA

class OneDataSet(Dataset):
    def __init__(
        self,
        smplx_data_path,
        audio_path,
        audio_feature_model_path=None,
        rotation_method="rot6d",
        local=False
    ):
        self.smplx_data_path = smplx_data_path
        self.audio_path = audio_path
        self.smplx_data = np.load(smplx_data_path, allow_pickle=True)
        self.audio = librosa.load(audio_path, sr=16000)[0]
        # WavLM feature extraction is optional — the BLD inference pipeline
        # re-extracts features from raw audio in context_encoding(), so loading
        # the model here is redundant for predict mode.
        if audio_feature_model_path is not None:
            self.audio_feature_processor = Wav2Vec2Processor.from_pretrained(audio_feature_model_path)
            self.audio_feature_model = WavLMModel.from_pretrained(audio_feature_model_path)
        else:
            self.audio_feature_processor = None
            self.audio_feature_model = None
        self.local = local
        self.num = 1

        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        for k, v in loadmat(os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.joints = 55

                
    def __len__(self):
        return self.num


    def __getitem__(self, idx):
        beta = self.smplx_data["betas"]
        pose = self.smplx_data["poses"]
        expression = self.smplx_data["expressions"]
        trans = self.smplx_data["trans"]
        ori_vid = self.smplx_data_path.split("/")[-1].split(".")[0]
        audio = self.audio
        ori_vid = os.path.splitext(os.path.basename(self.smplx_data_path))[0]
        pid = int(ori_vid.split("_")[0])


        if self.rotation_method == "rot6d":
            pose = axis_angle_to_rotation_6d(pose).astype(np.float32)
        elif self.rotation_method == "axisangle":
            pose = pose
        
        beta = torch.from_numpy(beta).to(torch.float32)
        pose = torch.from_numpy(pose).to(torch.float32)
        expression = torch.from_numpy(expression).to(torch.float32)
        trans = torch.from_numpy(trans).to(torch.float32)
        if len(audio) > 16000 * len(pose)//30:
            audio = audio[:16000 * len(pose)//30]
        elif len(audio) < 16000 * len(pose)//30:
            audio = np.pad(audio, (0, 16000 * len(pose)//30 - len(audio)), mode='constant', constant_values=0)
        if self.audio_feature_processor is not None:
            huberts = self.audio_feature_processor(audio, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                huberts = self.audio_feature_model(huberts).last_hidden_state
        else:
            # Dummy placeholder — predict path extracts features via context_encoding
            huberts = torch.zeros(1, len(pose), 1024)

        audio = torch.from_numpy(audio).to(torch.float32)
          
        J = self.rest_joint_position[pid]
        if not self.local:
            pose = LocalToGlobalAA(pose)
        return {
            "beta": beta.unsqueeze(0).repeat(pose.shape[0], 1),
            "pose": pose,
            "facial": expression,
            "trans":trans,
            "audio":audio,
            "audio_feat":huberts[0],
            "ori_vid": ori_vid,
            "id":pid,
            "J": J
        }


# Example usage
if __name__ == "__main__":
    smplx_data_path = "/remote-home/liuyi/PhysicallyPlausibleKeyPosesGuidedGeneration/Data/2_scott_0_1_1.npz"
    audio_path = "/remote-home/liuyi/PhysicallyPlausibleKeyPosesGuidedGeneration/Data/2_scott_0_1_1.wav"
    audio_feature_model_path = "/remote-home/liuyi/Sana/tools/wavlm-libri-clean-100h-large"
    ds = OneDataSet(smplx_data_path, audio_path, audio_feature_model_path, rotation_method="axisangle")

    print('#dataset', len(ds))

    data_id = 4569
    betas = ds[data_id]["beta"].numpy()
    huberts = ds[data_id]["audio_feat"].numpy()
    poses = ds[data_id]["pose"].numpy()
    audio = ds[data_id]["audio"].numpy()
    expressions = ds[data_id]["facial"].numpy()
    trans = ds[data_id]["trans"].numpy()

    print(poses.shape, huberts.shape)

