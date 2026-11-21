import torch
import torch.nn as nn
from transformers import PreTrainedModel
from diffusion.model.vae.models.new_emage.configuration_emage_audio import EmageVQVAEConvConfig, EmageVAEConvConfig
from diffusion.model.vae.models.new_emage.processing_emage_audio import (
    Quantizer,
    VQEncoderV5,
    VQDecoderV5,
    recover_from_mask_ts,
    rotation_6d_to_axis_angle,
    velocity2position,
    axis_angle_to_rotation_6d
)


def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t

class EmageVAEConv(PreTrainedModel):
    config_class = EmageVAEConvConfig
    base_model_prefix = "emage_vaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.decoder = VQDecoderV5(config)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        rec_pose = self.decoder(pre_latent)
        return {
            "rec_pose": rec_pose
            }

class EmageVQVAEConv(PreTrainedModel):
    config_class = EmageVQVAEConvConfig
    base_model_prefix = "emage_vqvaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.quantizer = Quantizer(config.vae_codebook_size, config.vae_length, config.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(config)
    
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {"poses_feat":vq_latent,"embedding_loss":embedding_loss,"perplexity":perplexity,"rec_pose": rec_pose}
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def get_latent_from_index(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def get_index_from_latent(self, latent):
        index = self.quantizer.map2index(latent)
        return index
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    
    def decode_from_latent(self, latent):
        # print(latent.shape)
        z_flattened = latent.contiguous().view(-1, self.quantizer.e_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.quantizer.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.quantizer.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        # print(min_encoding_indices.shape)
        indices = min_encoding_indices.view(latent.shape[0], latent.shape[1])
        z_q = self.quantizer.get_codebook_entry(indices)
        rec_pose = self.decoder(z_q)
        return rec_pose

class EmageVQModel(nn.Module):
    def __init__(
        self, 
        pretrained_path="H-Liu1997/emage_audio", 
        subfolder_face="emage_vq/face", 
        subfolder_upper="emage_vq/upper", 
        subfolder_lower="emage_vq/lower", 
        subfolder_hands="emage_vq/hands", 
        subfolder_global="emage_vq/global"
    ):
        super().__init__()
        face_model = EmageVQVAEConv.from_pretrained(pretrained_path, subfolder=subfolder_face)
        upper_model = EmageVQVAEConv.from_pretrained(pretrained_path, subfolder=subfolder_upper)
        lower_model = EmageVQVAEConv.from_pretrained(pretrained_path, subfolder=subfolder_lower)
        hands_model = EmageVQVAEConv.from_pretrained(pretrained_path, subfolder=subfolder_hands)
        global_model = EmageVAEConv.from_pretrained(pretrained_path, subfolder=subfolder_global)
    
        self.face_e_dim = face_model.quantizer.e_dim
        self.upper_e_dim = upper_model.quantizer.e_dim
        self.lower_e_dim = lower_model.quantizer.e_dim
        self.hands_e_dim = hands_model.quantizer.e_dim
    
        self.joint_mask_upper = [
          False, False, False, True, False, False, True, False, False, True,
          False, False, True, True, True, True, True, True, True, True,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.joint_mask_lower = [
          True, True, True, False, True, True, False, True, True, False,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.vq_model_face = face_model
        self.vq_model_upper = upper_model
        self.vq_model_hands = hands_model
        self.vq_model_lower = lower_model
        self.global_motion = global_model

    def spilt_inputs(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        bs, t, j6 = smplx_body_rot6d.shape
        smplx_body_rot6d = smplx_body_rot6d.reshape(bs, t, j6//6, 6)
        jaw_rot6d = smplx_body_rot6d[:, :, 22:23, :].reshape(bs, t, 6)
        face = torch.cat([jaw_rot6d, expression], dim=2)
        upper_rot6d = smplx_body_rot6d[:, :,self.joint_mask_upper, :].reshape(bs, t, 78)
        hands_rot6d = smplx_body_rot6d[:, :,25:55, :].reshape(bs, t, 180)
        lower_rot6d = smplx_body_rot6d[:, :,self.joint_mask_lower, :].reshape(bs, t, 54)
        tar_contact = torch.zeros(bs, t, 4, device=smplx_body_rot6d.device) if tar_contact is None else tar_contact
        tar_trans = torch.zeros(bs, t, 3, device=smplx_body_rot6d.device) if tar_trans is None else tar_trans
        lower = torch.cat([lower_rot6d, tar_trans, tar_contact], dim=2)
        return dict(face=face, upper=upper_rot6d, hands=hands_rot6d, lower=lower)
    
    def map2index(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression, tar_contact=tar_contact, tar_trans=tar_trans)
        face_index = self.vq_model_face.map2index(inputs["face"])
        upper_index = self.vq_model_upper.map2index(inputs["upper"])
        hands_index = self.vq_model_hands.map2index(inputs["hands"])
        lower_index = self.vq_model_lower.map2index(inputs["lower"])
        return dict(face=face_index, upper=upper_index, hands=hands_index, lower=lower_index)
    
    def map2latent(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression,tar_contact=tar_contact, tar_trans=tar_trans)
        face_latent = self.vq_model_face.map2latent(inputs["face"])
        upper_latent = self.vq_model_upper.map2latent(inputs["upper"])
        hands_latent = self.vq_model_hands.map2latent(inputs["hands"])
        lower_latent = self.vq_model_lower.map2latent(inputs["lower"])
        return dict(face=face_latent, upper=upper_latent, hands=hands_latent, lower=lower_latent)
    
    def encode(self, motion, expression):
        tar_trans = motion[..., -7:-4]
        tar_contact = motion[..., -4:]
        motion = motion[..., :-7]
        inputs = self.spilt_inputs(motion, expression, tar_contact=tar_contact, tar_trans=tar_trans)
        face_index = self.vq_model_face.map2index(inputs["face"])
        face = self.vq_model_face.get_latent_from_index(face_index)
        upper_index = self.vq_model_upper.map2index(inputs["upper"])
        upper = self.vq_model_upper.get_latent_from_index(upper_index)
        hands_index = self.vq_model_hands.map2index(inputs["hands"])
        hands = self.vq_model_hands.get_latent_from_index(hands_index)
        lower_index = self.vq_model_lower.map2index(inputs["lower"])
        lower = self.vq_model_lower.get_latent_from_index(lower_index)
        
        latent_motion = torch.cat([upper, hands, lower], dim=-1)
        latent_motion_index = torch.cat([upper_index.unsqueeze(-1), \
            hands_index.unsqueeze(-1), lower_index.unsqueeze(-1)], dim=-1)
        latent_expression = face
        latent_expression_index = face_index
        return dict(
            motion=latent_motion,
            expression=latent_expression,
            motion_index=latent_motion_index,
            expression_index=latent_expression_index
        )
    
    def decode(self, motion, expression, motion_index=None, expression_index=None,\
        get_global_motion=False, ref_trans=None):
        face_index = expression_index
        face_latent = expression
        upper_index = motion_index[..., 0] if motion_index is not None else None
        hands_index = motion_index[..., 1] if motion_index is not None else None
        lower_index = motion_index[..., 2] if motion_index is not None else None
        upper_latent = motion[..., :self.upper_e_dim]
        hands_latent = motion[..., self.upper_e_dim:self.upper_e_dim+self.hands_e_dim]
        lower_latent = motion[..., self.upper_e_dim+self.hands_e_dim:]
        
        for input_tensor in [face_index, upper_index, hands_index, lower_index,\
            face_latent, upper_latent, hands_latent, lower_latent]:
            if input_tensor is not None:
                bs, t = input_tensor.shape[:2]
                break
  
        if face_index is not None:
            face_mix = self.vq_model_face.decode(face_index) # bs, t, 106
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        elif face_latent is not None:
            face_mix = self.vq_model_face.decode_from_latent(face_latent)
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        else:
            face_jaw = torch.zeros(bs, t, 3, device=self.vq_model_face.device)
            expression = torch.zeros(bs, t, 100, device=self.vq_model_face.device)

        if upper_index is not None:
            # print(upper_index)
            upper_6d = self.vq_model_upper.decode(upper_index) # bs, t, 78
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif upper_latent is not None:
            upper_6d = self.vq_model_upper.decode_from_latent(upper_latent)
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            upper = torch.zeros(bs, t, 39, device=self.vq_model_upper.device)

        if hands_index is not None:
            hands_6d = self.vq_model_hands.decode(hands_index)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif hands_latent is not None:
            hands_6d = self.vq_model_hands.decode_from_latent(hands_latent)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            hands = torch.zeros(bs, t, 90, device=self.vq_model_hands.device)
        
        if lower_index is not None:
            lower_mix = self.vq_model_lower.decode(lower_index)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif lower_latent is not None:
            lower_mix = self.vq_model_lower.decode_from_latent(lower_latent)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            lower = torch.zeros(bs, t, 27, device=self.vq_model_lower.device)
            transfoot = torch.zeros(bs, t, 7, device=self.vq_model_lower.device)
            lower_6d = axis_angle_to_rotation_6d(lower.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
            lower_mix = torch.cat([lower_6d, transfoot], dim=-1)

        upper2all = recover_from_mask_ts(upper, self.joint_mask_upper)
        hands2all = recover_from_mask_ts(hands, [False]*25+[True]*30)
        lower2all = recover_from_mask_ts(lower, self.joint_mask_lower)
        
        all_motion_axis_angle = upper2all + hands2all + lower2all
        all_motion_axis_angle[:, :, 22*3:22*3+3] = face_jaw
        all_motion_rot6d = axis_angle_to_rotation_6d(all_motion_axis_angle.reshape(bs, t, 55, 3)).reshape(bs, t, 55*6)

        all_motion4inference = torch.cat([all_motion_rot6d, transfoot], dim=2) # 330 + 3 + 4
        
        global_motion = None
        if get_global_motion:
            global_motion = self.get_global_motion(lower_mix, ref_trans)
        return dict(
            motion=all_motion4inference,
            expression=expression,
            motion_axis_angle=all_motion_axis_angle,
            trans=global_motion
        )
    
    def get_global_motion(self, lower_body, ref_trans):
        global_motion = self.global_motion(lower_body)
        rec_trans_v_s = global_motion["rec_pose"][:, :, 54:57]
        if len(ref_trans.shape) == 2:
            ref_trans = ref_trans.unsqueeze(0).repeat(rec_trans_v_s.shape[0], 1, 1)
        
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1/30, ref_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1/30, ref_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        global_motion = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        return global_motion


if __name__ == "__main__":
    model = EmageVQModel(pretrained_path="/remote-home/liuyi/Sana/tools/EMAGE/pretrained_vqvae")
    motion = torch.randn(1, 100, 55*6 + 7)
    expression = torch.randn(1, 100, 100)
    
    results = model.encode(motion, expression)
    print("motion shape: ", results["motion"].shape)
    print("expression shape: ", results["expression"].shape)
    print("motion_index shape: ", results["motion_index"].shape)
    print("expression_index shape: ", results["expression_index"].shape)
    
    motion_index = results["motion_index"]
    expression_index = results["expression_index"]
    results = model.decode(results["motion"], results["expression"], motion_index, expression_index)
    print("motion shape: ", results["motion"].shape)
    print("expression shape: ", results["expression"].shape)