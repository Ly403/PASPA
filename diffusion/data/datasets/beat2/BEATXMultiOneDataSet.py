"""
Batch predict dataset: reads a clip list file with (ori_vid, start_frame, end_frame)
per line, returns each clip as a separate dataset item (same format as OneDataSet).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from diffusion.utils.rotation_conversions import axis_angle_to_rotation_6d
from scipy.io import loadmat
import librosa
from diffusion.utils.rotation_conversions import LocalToGlobalAA


class MultiOneDataSet(Dataset):
    def __init__(
        self,
        clip_list_file,
        data_folder,
        rotation_method="axisangle",
        local=False,
        **kwargs,
    ):
        """
        Args:
            clip_list_file: text file, each line: "ori_vid start_frame end_frame"
                e.g. "2_scott_0_1_1 300 2160"
            data_folder: BEAT2 root containing smplxflame_30/ and wave16k/
            rotation_method: "rot6d" or "axisangle"
            local: if False, convert local AA to global AA
            **kwargs: absorbs extra params from base config (smplx_data_path, etc.)
        """
        self.clips = []
        with open(clip_list_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    self.clips.append((parts[0], int(parts[1]), int(parts[2])))
                elif len(parts) == 1 and parts[0]:
                    # Backward compat: bare ori_vid = full session
                    self.clips.append((parts[0], 0, -1))

        self.data_folder = data_folder
        self.rotation_method = rotation_method
        assert rotation_method in ("rot6d", "axisangle")
        self.local = local

        self.rest_joint_position = {}
        for k, v in loadmat(
            os.path.join(os.path.dirname(__file__), "rest_joint_position.mat")
        ).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.joints = 55

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        ori_vid, start, end = self.clips[idx]
        npz_path = os.path.join(
            self.data_folder, "smplxflame_30", ori_vid + ".npz"
        )
        wav_path = os.path.join(
            self.data_folder, "wave16k", ori_vid + ".wav"
        )

        smplx_data = np.load(npz_path, allow_pickle=True)
        beta = smplx_data["betas"]
        pose = smplx_data["poses"]
        expression = smplx_data["expressions"]
        trans = smplx_data["trans"]
        pid = int(ori_vid.split("_")[0])

        audio_full = librosa.load(wav_path, sr=16000)[0]

        # Clip to [start, end)
        if end <= 0:
            end = len(pose)
        pose = pose[start:end]
        expression = expression[start:end]
        trans = trans[start:end]

        audio_start = int(start * 16000 / 30)
        audio_end = int(end * 16000 / 30)
        audio = audio_full[audio_start:audio_end]

        if self.rotation_method == "rot6d":
            pose = axis_angle_to_rotation_6d(pose).astype(np.float32)

        beta = torch.from_numpy(beta).to(torch.float32)
        pose = torch.from_numpy(pose).to(torch.float32)
        expression = torch.from_numpy(expression).to(torch.float32)
        trans = torch.from_numpy(trans).to(torch.float32)

        # Align audio length to pose length
        expected_audio_len = 16000 * len(pose) // 30
        if len(audio) > expected_audio_len:
            audio = audio[:expected_audio_len]
        elif len(audio) < expected_audio_len:
            audio = np.pad(
                audio,
                (0, expected_audio_len - len(audio)),
                mode="constant",
                constant_values=0,
            )

        # Dummy placeholder — predict path extracts features via context_encoding
        huberts = torch.zeros(len(pose), 1024)

        audio = torch.from_numpy(audio).to(torch.float32)

        J = self.rest_joint_position[pid]
        if not self.local:
            pose = LocalToGlobalAA(pose)

        # clip_id distinguishes different clips from the same session
        clip_id = f"{ori_vid}_{start}" if start > 0 else ori_vid

        return {
            "beta": beta.unsqueeze(0).repeat(pose.shape[0], 1),
            "pose": pose,
            "facial": expression,
            "trans": trans,
            "audio": audio,
            "audio_feat": huberts,
            "ori_vid": clip_id,
            "id": pid,
            "J": J,
        }
