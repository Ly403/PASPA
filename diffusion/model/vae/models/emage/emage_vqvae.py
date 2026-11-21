import numpy as np
import torch
import torch.nn as nn
import copy
from diffusion.model.vae.models.emage.motion_encoder import * 
from diffusion.model.vae.models.emage.quantizer import *
from diffusion.model.vae.models.utils.network import load_checkpoints
from omegaconf import DictConfig
from diffusion.utils.logger import get_root_logger
import os
# ----------- AE, VAE ------------- #
class VAEConvZero(nn.Module):
    def __init__(self, args):
        super(VAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        # embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(pre_latent)
        return {
            # "poses_feat":vq_latent,
            # "embedding_loss":embedding_loss,
            # "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.decoder = VQDecoderV3(args)
        self.fc_mu = nn.Linear(args.vae_length, args.vae_length)
        self.fc_logvar = nn.Linear(args.vae_length, args.vae_length)
        self.variational = args.variational
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu, logvar = None, None
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        rec_pose = self.decoder(pre_latent)
        return {
            "poses_feat":pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
            }
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        return pre_latent
    
    def decode(self, pre_latent):
        rec_pose = self.decoder(pre_latent)
        return rec_pose

class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)
        
class VAEConvMLP(VAEConv):
    def __init__(self, args):
        super(VAEConvMLP, self).__init__(args)
        self.encoder = PoseEncoderConv(args.vae_test_len, args.vae_test_dim, feature_length=args.vae_length)
        self.decoder = PoseDecoderConv(args.vae_test_len, args.vae_test_dim, feature_length=args.vae_length)
 
# class VAELSTM(VAEConv):
#     def __init__(self, args):
#         super(VAELSTM, self).__init__(args)
#         pose_dim = args.vae_test_dim
#         feature_length = args.vae_length
#         self.encoder = PoseEncoderLSTM_Resnet(pose_dim, feature_length=feature_length)
#         self.decoder = PoseDecoderLSTM(pose_dim, feature_length=feature_length)

class VAETransformer(VAEConv):
    def __init__(self, args):
        super(VAETransformer, self).__init__(args)
        self.encoder = Encoder_TRANSFORMER(args)
        self.decoder = Decoder_TRANSFORMER(args)

# ----------- VQVAE --------------- #
class VQVAEConv(nn.Module):
    def __init__(self, args):
        super(VQVAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV3(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose

class VQVAESKConv(VQVAEConv):
    def __init__(self, args):
        super(VQVAESKConv, self).__init__(args)
        smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)


class VQVAEConvStride(nn.Module):
    def __init__(self, args):
        super(VQVAEConvStride, self).__init__()
        self.encoder = VQEncoderV4(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV4(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose

class VQVAEConvZero(nn.Module):    
    def __init__(self, args):
        super(VQVAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def get_index_from_latent(self, latent):
        index = self.quantizer.map2index(latent)
        return index
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    

class VAEConvZero(nn.Module):
    def __init__(self, args):
        super(VAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        # embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(pre_latent)
        return {
            # "poses_feat":vq_latent,
            # "embedding_loss":embedding_loss,
            # "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    # def map2index(self, inputs):
    #     pre_latent = self.encoder(inputs)
    #     index = self.quantizer.map2index(pre_latent)
    #     return index
    
    # def map2latent(self, inputs):
    #     pre_latent = self.encoder(inputs)
    #     index = self.quantizer.map2index(pre_latent)
    #     z_q = self.quantizer.get_codebook_entry(index)
    #     return z_q
    
    # def decode(self, index):
    #     z_q = self.quantizer.get_codebook_entry(index)
    #     rec_pose = self.decoder(z_q)
    #     return rec_pose


class VQVAEConvZero3(nn.Module):
    def __init__(self, args):
        super(VQVAEConvZero3, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose

class VQVAEConvZero2(nn.Module):
    def __init__(self, args):
        super(VQVAEConvZero2, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV7(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose

class VQVAE2(nn.Module):
    def __init__(self, args):
        super(VQVAE2, self).__init__()
        # Bottom-level encoder and decoder
        args_bottom = copy.deepcopy(args)
        args_bottom.vae_layer = 2
        self.bottom_encoder = VQEncoderV6(args_bottom)
        self.bottom_quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        args_bottom.vae_test_dim = args.vae_test_dim
        self.bottom_decoder = VQDecoderV6(args_bottom)
        
        # Top-level encoder and decoder
        args_top = copy.deepcopy(args)
        args_top.vae_layer = 3
        args_top.vae_test_dim = args.vae_length
        self.top_encoder = VQEncoderV3(args_top)  # Adjust according to the top level's design
        self.quantize_conv_t = nn.Conv1d(args.vae_length+args.vae_length, args.vae_length, 1)
        self.top_quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        # self.upsample_t_up = nn.Upsample(scale_factor=2, mode='nearest')
        layers = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample_t= nn.Sequential(*layers)
        self.top_decoder = VQDecoderV3(args_top)  # Adjust to handle top level features appropriately

    def forward(self, inputs):
        # Bottom-level processing
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        #print(enc_b.shape, enc_t.shape)
        top_embedding_loss, quant_t, _, top_perplexity = self.top_quantizer(enc_t)
        #print(quant_t.shape)
        dec_t = self.top_decoder(quant_t)
        #print(dec_t.shape)
        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        #print("5",quant_b.shape)
        bottom_embedding_loss, quant_b, _, bottom_perplexity = self.bottom_quantizer(quant_b)
        #print("6",quant_b.shape)
        upsample_t = self.upsample_t(quant_t.permute(0,2,1)).permute(0,2,1)
        #print("7",upsample_t.shape)
        quant = torch.cat([upsample_t, quant_b], 2)
        rec_pose = self.bottom_decoder(quant)
        # print(quant_t.shape, quant_b.shape, rec_pose.shape)
        return {
            "poses_feat_top": quant_t,
            "pose_feat_bottom": quant_b,
            "embedding_loss":top_embedding_loss+bottom_embedding_loss,
            #"perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        
        _, quant_t, _, _ = self.top_quantizer(enc_t)
        top_index = self.top_quantizer.map2index(enc_t)
        dec_t = self.top_decoder(quant_t)

        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        # quant_b = self.quantize_conv_t(enc_b)
        bottom_index = self.bottom_quantizer.map2index(quant_b)
        return top_index, bottom_index
    
    def get_top_laent(self, top_index):
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        return z_q_top
    
    def map2latent(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        
        _, quant_t, _, _ = self.top_quantizer(enc_t)
        top_index = self.top_quantizer.map2index(enc_t)
        dec_t = self.top_decoder(quant_t)

        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        # quant_b = self.quantize_conv_t(enc_b)
        bottom_index = self.bottom_quantizer.map2index(quant_b)
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        z_q_bottom = self.bottom_quantizer.get_codebook_entry(bottom_index)
        return z_q_top, z_q_bottom
    
    def map2latent_top(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        top_index = self.top_quantizer.map2index(enc_t)
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        return z_q_top
    
    def decode(self, top_index, bottom_index):
        quant_t = self.top_quantizer.get_codebook_entry(top_index)
        quant_b = self.bottom_quantizer.get_codebook_entry(bottom_index)
        upsample_t = self.upsample_t(quant_t.permute(0,2,1)).permute(0,2,1)
        #print("7",upsample_t.shape)
        quant = torch.cat([upsample_t, quant_b], 2)
        rec_pose = self.bottom_decoder(quant)      
        return rec_pose


class EmageVAE(nn.Module):
    def __init__(
        self, 
        vae_length=256,
        vae_test_dim=141,
        vae_test_len=34,
        vae_test_stride=10,
        vae_codebook_size=1024,
        vae_quantizer_lambda=1.,
        vae_layer=2,
        vae_grow=[1,1,2,1],
        model_path=None,
    ):
        super().__init__()
        args = DictConfig({
            "vae_length": vae_length,
            "vae_test_dim": vae_test_dim,
            "vae_test_len": vae_test_len,
            "vae_test_stride": vae_test_stride,
            "vae_codebook_size": vae_codebook_size,
            "vae_quantizer_lambda": vae_quantizer_lambda,
            "vae_layer": vae_layer,
            "vae_grow": vae_grow,
            "model_path": model_path,
        })
        self.args = args
        self.logger = get_root_logger()
        self.args.vae_test_dim = 106
        self.vq_model_face = VQVAEConvZero(self.args)
        
        self.args.vae_test_dim = 78
        self.vq_model_upper = VQVAEConvZero(self.args)
        
        self.args.vae_test_dim = 180
        self.vq_model_hands = VQVAEConvZero(self.args)
        
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.vq_model_lower = VQVAEConvZero(self.args)
        
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.global_motion = VQVAEConvZero(self.args)
        
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240
        self.initialize_weights()
    
    def initialize_weights(self):
        if self.args.model_path is None or self.args.model_path == '':
            self.logger.info("Would not load model weights for the motion encoder since model_path is None")
            return
        load_checkpoints(self.vq_model_face, os.path.join(self.args.model_path,"emage_vq/face/pytorch_model.bin"))
        load_checkpoints(self.vq_model_upper, os.path.join(self.args.model_path,"emage_vq/upper/pytorch_model.bin"))
        load_checkpoints(self.vq_model_hands, os.path.join(self.args.model_path,"emage_vq/hands/pytorch_model.bin"))
        load_checkpoints(self.vq_model_lower, os.path.join(self.args.model_path,"emage_vq/lower/pytorch_model.bin"))
        load_checkpoints(self.global_motion, os.path.join(self.args.model_path,"emage_vq/global/pytorch_model.bin"))
    
    def map2index(self, tar_pose_face, tar_pose_upper, tar_pose_hands, tar_pose_lower):
        tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face) # bs*n/4
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper) # bs*n/4
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower) # bs*n/4
        return tar_index_value_face_top, tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top
    
    def map2latent(self, tar_pose_face, tar_pose_upper, tar_pose_hands, tar_pose_lower):
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        return latent_face_top, latent_upper_top, latent_hands_top, latent_lower_top
    
    def encode(self, input):
        tar_pose_face = input["face"]
        tar_pose_upper = input["upper"]
        tar_pose_hands = input["hands"]
        tar_pose_lower = input["lower"]
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        tar_index_value_face_top = self.vq_model_face.get_index_from_latent(latent_face_top)
        tar_index_value_upper_top = self.vq_model_upper.get_index_from_latent(latent_upper_top)
        tar_index_value_hands_top = self.vq_model_hands.get_index_from_latent(latent_hands_top) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.get_index_from_latent(latent_lower_top) # bs*n/4
        # return latent_face_top, latent_upper_top, latent_hands_top, latent_lower_top, tar_index_value_face_top, \
        #     tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top
        return {
            "face": latent_face_top,
            "upper": latent_upper_top,
            "hands": latent_hands_top,
            "lower": latent_lower_top,
            "face_index": tar_index_value_face_top,
            "upper_index": tar_index_value_upper_top,
            "hands_index": tar_index_value_hands_top,
            "lower_index": tar_index_value_lower_top,
        }
    
    def decode(self, input):
        latent_face_top = input["face"]
        latent_upper_top = input["upper"]
        latent_hands_top = input["hands"]
        latent_lower_top = input["lower"]
        tar_index_value_face_top = self.vq_model_face.get_index_from_latent(latent_face_top)
        tar_index_value_upper_top = self.vq_model_upper.get_index_from_latent(latent_upper_top)
        tar_index_value_hands_top = self.vq_model_hands.get_index_from_latent(latent_hands_top) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.get_index_from_latent(latent_lower_top) # bs*n/4
        value_face = self.vq_model_face.decode(tar_index_value_face_top)
        value_upper = self.vq_model_upper.decode(tar_index_value_upper_top)
        value_hands = self.vq_model_hands.decode(tar_index_value_hands_top)
        value_lower = self.vq_model_lower.decode(tar_index_value_lower_top)
        # return value_face, value_upper, value_hands, value_lower
        return {
            "face": value_face,
            "upper": value_upper,
            "hands": value_hands,
            "lower": value_lower,
        }
 
if __name__ == "__main__":
    model = EmageVAE(
        vae_test_len=32,
        vae_test_dim=330,
        vae_test_stride=20,
        vae_length=240,
        vae_codebook_size=256,
        vae_layer=4,
        vae_grow=[1,1,2,1],
        model_path=None,
    )
    tar_pose_face = torch.rand(1, 32, 106) 
    tar_pose_upper = torch.rand(1, 32, 78)
    tar_pose_hands = torch.rand(1, 32, 180)
    tar_pose_lower = torch.rand(1, 32, 61)
    results = model.encode({
        "face": tar_pose_face,
        "upper": tar_pose_upper,
        "hands": tar_pose_hands,
        "lower": tar_pose_lower,
    })
    for k, result in results.items():
        print(result.shape)