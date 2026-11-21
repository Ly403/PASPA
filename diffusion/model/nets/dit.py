import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import Summer, get_emb
import numpy as np
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from diffusion.model.nets.basic_modules import MPFourierTimeEmbedding
from diffusion.utils.config import instantiate_from_config
'''
* is it feasible to improve the generation quality by using both local and global rotations?
* directly predicting the poses can result in jittering between two inference chuncks.
    is it possible to predict incremental rotations instead of absolute rotations?
* can we generate both audio and gestures together from text? this can reduce the latency of the system. 
* can we improve the generation quality by using a SKEL, which is a biomechanics model
* can bone direction loss improve the generation quality?
* can we use similarity loss on latent structure to improve the generation quality?
* is it feasible to improve the performance by using bone direction loss?
* in the future i shall consider using latent diffusion models to generate the poses.
* i shall try intermediate supervision to improve the generation quality.
* decompose the parameter space into three parts: expression, hand joints, and other body joints.
'''

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Rotary Position Embedding (RoPE)
        
        参数:
            dim (int): 输入特征的维度（必须是偶数）
        """
        super().__init__()
        assert dim % 2 == 0, "维度必须是偶数"
        self.dim = dim
        
        # 初始化频率参数 theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 缓存cos和sin值
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = None

    def _update_cos_sin_cache(self, seq_len, device):
        """更新cos和sin缓存"""
        if self.max_seq_len is None or seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            # 生成位置索引 [0, 1, ..., seq_len-1]
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            
            # 计算频率矩阵 shape: (seq_len, dim//2)
            freqs = torch.outer(t, self.inv_freq)
            
            # 拼接相同频率用于后续广播 shape: (seq_len, dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # 缓存cos和sin值
            self.cos_cached = emb.cos()[None, :, None, :]  # 形状: (1, seq_len, 1, dim)
            self.sin_cached = emb.sin()[None, :, None, :]

    def rotate_half(self, x):
        """将输入的后半部分旋转并取反"""
        x = x.view(*x.shape[:-1], -1, 2)        # 分组成二维向量
        x1 = x[..., 0]                          # 前半部分
        x2 = x[..., 1]                          # 后半部分
        x_rot = torch.stack((-x2, x1), dim=-1)   # 旋转操作
        return x_rot.view(*x.shape[:-2], -1)     # 恢复原始形状

    def forward(self, x, offset=0):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状为 (batch, seq_len, n_head, dim)
            offset (int): 位置偏移量（用于增量解码）
        
        返回:
            Tensor: 旋转后的张量
        """
        batch, seq_len, n_head, _ = x.size()
        self._update_cos_sin_cache(seq_len + offset, x.device)
        
        # 获取对应位置的cos和sin值
        cos = self.cos_cached[:, offset:offset+seq_len]
        sin = self.sin_cached[:, offset:offset+seq_len]
        
        # 应用旋转位置编码
        x_rot = (x * cos) + (self.rotate_half(x) * sin)
        return x_rot

class PositionalEncoding1D(nn.Module):
    # 我真服了，这个sb代码有个抽象且隐秘的bug
    # ddp的情况下，不同卡如果处理的不等长的序列，会导致不同卡间的cached_penc不同尺寸
    # 从而导致ddp广播出错，这个错误非常难以发现，debug花了我好长时间
    # 不知道搞个cache有什么意义，这个库的代码质量不过关，下面是去掉cache后的版本
    # 你就算要写cache，你也要区分training和eval两种状态啊，这点意识都没有吗
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return cached_penc


class MyMultiheadAttention(nn.Module):
    '''
    A multi-head attention module that supports different input dimensions for queries, keys, and values.'''
    def __init__(self, embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 attention_dim=None,
                 qdim=None, kdim=None, vdim=None, out_dim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.out_dim = out_dim if out_dim is not None else embed_dim
        self.attention_dim = attention_dim or embed_dim


        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.attention_dim // num_heads
        assert self.head_dim * num_heads == self.attention_dim, "embed_dim must be divisible by num_heads"


        self.q_proj_weight = Parameter(torch.empty((self.attention_dim, self.qdim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.attention_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))



        self.in_proj_bias_q = Parameter(torch.empty(self.attention_dim, **factory_kwargs))
        self.in_proj_bias_k = Parameter(torch.empty(self.attention_dim, **factory_kwargs))
        self.in_proj_bias_v = Parameter(torch.empty(embed_dim, **factory_kwargs))

        self.out_proj = nn.Linear(embed_dim, self.out_dim, bias=True, **factory_kwargs)


        self._reset_parameters()

    def _reset_parameters(self):

        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)


        constant_(self.in_proj_bias_q, 0.)
        constant_(self.in_proj_bias_k, 0.)
        constant_(self.in_proj_bias_v, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v):
        q = torch.einsum("blj,ij->bli", q, self.q_proj_weight)
        k = torch.einsum("blj,ij->bli", k, self.k_proj_weight)
        v = torch.einsum("blj,ij->bli", v, self.v_proj_weight)

        B, L = q.shape[:2]
        q =  q + self.in_proj_bias_q
        k = k + self.in_proj_bias_k
        v = v + self.in_proj_bias_v

        q = q.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        k = k.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        v = v.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        res = F.scaled_dot_product_attention(q, k,v, dropout_p=self.dropout).transpose(-3, -2)
        res = res.reshape((B, L, -1))
        res = self.out_proj(res)
        return res

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        集成Rotary Position Embedding的多头注意力机制

        参数:
            embed_dim (int): 输入嵌入维度
            num_heads (int): 注意力头数
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 初始化QKV投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 初始化Rotary位置编码
        self.rotary_pe = RotaryPositionEmbedding(self.head_dim)

        # 最终输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, offset=0):
        """
        前向传播

        参数:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            key_padding_mask: (batch_size, seq_len) 可选
            attn_mask: (seq_len, seq_len) 可选
            offset: 位置偏移（用于增量解码）

        返回:
            attn_output: (batch_size, seq_len, embed_dim)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 投影到Q, K, V
        q = self.q_proj(query)  # (B, T, E)
        k = self.k_proj(key)    # (B, T, E)
        v = self.v_proj(value)  # (B, T, E)

        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)

        # 应用Rotary位置编码到Q和K
        q = self.rotary_pe(q.permute(0, 2, 1, 3), offset=offset).permute(0, 2, 1, 3)  # 保持形状(B, H, T, D)
        k = self.rotary_pe(k.permute(0, 2, 1, 3), offset=offset).permute(0, 2, 1, 3)

        # 计算注意力分数 (B, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 处理掩码
        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        if attn_mask is not None:
            attn_scores += attn_mask

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # 合并多头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # (B, T, E)

        # 最终投影
        attn_output = self.out_proj(attn_output)
        return attn_output

class CrossAttentionLayerWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.attention = MultiHeadAttentionWithRoPE(embed_dim, num_heads)
    
    def forward(self, x, key, value, key_padding_mask=None, attn_mask=None, offset=0):
        x = self.ln1(x)
        key = self.ln2(key)
        value = self.ln3(value)
        attn_output = self.attention(x, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, offset=offset)
        x = x + attn_output
        x = self.ln4(x)
        x = x + self.ff(x)
        return x


class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_persons, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        return self.mlp(t)

class SimpleSpeechModel(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModel, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out


class SimpleSpeechModelWav2Sem(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModelWav2Sem, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64 + 1024, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid, wav2sem):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        wav2sem = wav2sem.unsqueeze(1).expand(-1, huberts.shape[1], -1)
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed, wav2sem], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out





class DiTBlock(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x, **kwargs):
        time_embed = kwargs['time_embed']
        x0 = x
        x = self.ln1(x)
        gate1, scale1, shift1, gate2, scale2, shift2 = self.mlp(time_embed).chunk(6, dim=-1)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln3(x)
        return x


class DiTBlock2(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed



    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)



        # time_style_embed
        x0 = x
        x = self.ln1(x)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        # ff
        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        
        x = self.ln3(x)
        return x

class DiTBlock2PostNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed



    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)



        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x



class DiTBlock2PostNormGuider(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 12)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_poses_prompt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_poses_prompt  = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)


        self.ln_guider = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_guider  = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn_guider = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        poses_prompt = kwargs['poses_prompt']
        prompt_mask = kwargs['prompt_mask']
        guider = kwargs['guider']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3, gate4, scale4, shift4 = self.mlp(
            time_emb).chunk(12, dim=-1)
        # poses prompt
        poses_prompt0 = poses_prompt
        poses_prompt = self.ln_poses_prompt(poses_prompt)
        poses_prompt = poses_prompt * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        poses_prompt = self.ff_poses_prompt(poses_prompt) * gate3.unsqueeze(1) + poses_prompt0
        x = self.cross_attn(x, poses_prompt, poses_prompt, key_padding_mask=prompt_mask)[0] + x


        # guider
        guider0 = guider
        guider = self.ln_guider(guider)
        guider = guider * (1 + scale4.unsqueeze(1)) + shift4.unsqueeze(1)
        guider = self.ff_guider(guider) * gate4.unsqueeze(1) + guider0
        x = self.cross_attn_guider(x, guider, guider)[0] + x


        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x




class DiTBlock2PostNormGuider2(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_poses_prompt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_poses_prompt  = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)


        self.fuse_guider = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        poses_prompt = kwargs['poses_prompt']
        prompt_mask = kwargs['prompt_mask']
        guider = kwargs['guider']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # poses prompt
        poses_prompt0 = poses_prompt
        poses_prompt = self.ln_poses_prompt(poses_prompt)
        poses_prompt = poses_prompt * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        poses_prompt = self.ff_poses_prompt(poses_prompt) * gate3.unsqueeze(1) + poses_prompt0
        x = self.cross_attn(x, poses_prompt, poses_prompt, key_padding_mask=prompt_mask)[0] + x


        # guider
        x = self.fuse_guider(torch.cat([x, guider], dim=-1))


        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim, num_freqs=64):
        """
        时间编码模块，用于将连续时间 t ∈ [0,1] 映射为高维嵌入。

        参数：
        - dim: 输出嵌入的维度
        - num_freqs: 使用的频率数量（每个频率对应一个 sin 和 cos）
        """
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.num_freqs = num_freqs

        # 频率项，使用可学习的 buffer
        inv_freq = 1.0 / (10000 ** (torch.arange(0, num_freqs, dtype=torch.float32) / num_freqs))
        self.register_buffer('inv_freq', inv_freq)

        # MLP 将傅里叶特征映射到目标维度
        self.mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        输入：
        - t: (batch_size,)，表示时间（通常归一化在 [0,1] 范围内）

        输出：
        - time_embedding: (batch_size, dim)
        """
        # 构造傅里叶特征
        x = torch.outer(t, self.inv_freq)  # (batch_size, num_freqs)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        x = torch.cat([sin_x, cos_x], dim=-1)  # (batch_size, 2 * num_freqs)

        # 通过 MLP 映射到目标维度
        time_embedding = self.mlp(x)
        return time_embedding


class DiffusionDITNetPartsFixedExpressions2PostNorm(nn.Module):
    def __init__(self, hidden_size=384,  num_layers=6, num_person=31, num_embedding_channels=768):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 , hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6 + 100, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 100, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6)


        # self.time_embed = nn.Sequential(
        #     nn.Linear(1, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size - 64),
        # )
        self.time_embed = TimeEmbedding(hidden_size, num_embedding_channels=num_embedding_channels)
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6, hidden_size)



        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['speaker_id']
        poses_prompt = kwargs['poses_prompt']
        prompt_mask = kwargs['prompt_mask']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps)
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        hand_poses_prompt = self.seed_prj_hand(poses_prompt[..., hand_indices])
        body_poses_prompt = self.seed_prj_body(poses_prompt[..., body_indices])


        hand_poses_prompt = self.pos_enc(hand_poses_prompt)
        body_poses_prompt = self.pos_enc(body_poses_prompt)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)


        

        kwargs = {'poses_prompt': hand_poses_prompt, 'time_embed': time_style_embed, 'audio_embeding': huberts, 'prompt_mask': prompt_mask}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'poses_prompt': body_poses_prompt, 'time_embed': time_style_embed, 'audio_embeding': huberts, 'prompt_mask': prompt_mask}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x
    
    def sample(self, noise,  normalize_func=None, **kwargs):
        B, L = noise.shape[:2]
        device = noise.device
        n_timesteps = 10
        time_steps = torch.linspace(0, 1, steps=n_timesteps + 1, device=device)
        x0 = noise
        xt = x0
        for i in range(n_timesteps):
            ts, te = time_steps[i], time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward(xt,  ts, **kwargs)
            if normalize_func is not None:
                pred_x1 = normalize_func(pred_x1[...,:330])
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]

        return xt


class DiffusionDITNetPartsFixedExpressions2PostNormGuider(nn.Module):
    def __init__(self, hidden_size=384,num_layers=6, num_person=31, num_embedding_channels=384):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 , hidden_size)
        self.in_trans = nn.Linear(3, hidden_size)

        self.in_x_hand = nn.Linear(55 * 6 + 3 + 100, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 3 + 100, hidden_size)
        self.in_x_trans = nn.Linear(55 * 6 + 3 + 100, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.trans_fuse = nn.Linear(hidden_size * 2, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2PostNormGuider(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2PostNormGuider(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_trans = nn.ModuleList([DiTBlock2PostNormGuider(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6)
        self.trans = nn.Linear(hidden_size, 3)


        # self.time_embed = nn.Sequential(
        #     nn.Linear(1, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size - 64),
        # )
        self.time_embed = MPFourierTimeEmbedding(hidden_size, num_embedding_channels=num_embedding_channels)
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6, hidden_size)
        self.seed_prj_trans = nn.Linear(3, hidden_size)



        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self, 
        motion, 
        timestep,
        expression,
        audio = None,
        audio_mask = None,
        txt = None,
        txt_mask = None,
        **kwargs
    ):
        x = motion
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_joint_indices = kwargs['face_joint_indices']
        trans_indices = kwargs['trans_indices']
        pid = kwargs['speaker_id']
        poses_prompt = kwargs['pose_prompt']
        prompt_mask = kwargs['pose_prompt_mask']
        trans_prompt = kwargs['trans_prompt']
        trans_mask = kwargs['trans_prompt_mask']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = motion

        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        trans = motion[...,trans_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        trans = self.in_trans(trans)
        
        x = torch.cat([motion, expression], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)
        xtrans = self.in_x_trans(x)



        huberts = self.hubert_prj(audio)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))
        trans = self.trans_fuse(torch.cat([trans, xtrans], dim=-1))


        time_embed = self.time_embed(timestep)
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(motion.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        hand_poses_prompt = self.seed_prj_hand(poses_prompt[..., hand_indices])
        body_poses_prompt = self.seed_prj_body(poses_prompt[..., body_indices])
        trans_prompt = self.seed_prj_trans(trans_prompt)


        hand_poses_prompt = self.pos_enc(hand_poses_prompt)
        body_poses_prompt = self.pos_enc(body_poses_prompt)
        trans_prompt = self.pos_enc(trans_prompt)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)
        trans = self.pos_enc(trans)


        pre_hand = hand
        pre_body = body
        for hand_blk, body_blk, trans_blk in zip(self.transformer_hand, self.transformer_body, self.transformer_trans):

            kwargs = {'poses_prompt': hand_poses_prompt, 'time_embed': time_style_embed, 'prompt_mask': prompt_mask, "guider":pre_body}
            hand = hand_blk(hand, **kwargs)

            kwargs = {'poses_prompt': body_poses_prompt, 'time_embed': time_style_embed, 'prompt_mask': prompt_mask, "guider":pre_hand}
            body = body_blk(body, **kwargs)

            kwargs = {'poses_prompt': trans_prompt, 'time_embed': time_style_embed, 'prompt_mask': trans_mask, "guider":pre_body}
            trans = trans_blk(trans, **kwargs)
            pre_hand = hand
            pre_body = body



        hand = self.hand(hand)
        body = self.body(body)
        trans = self.trans(trans)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        x[..., trans_indices] = trans
        
        return {
            "motion": x,
        }
    
    def sample(self, noise,  normalize_func=None, **kwargs):
        B, L = noise.shape[:2]
        device = noise.device
        n_timesteps = 10
        time_steps = torch.linspace(0, 1, steps=n_timesteps + 1, device=device)
        x0 = noise
        xt = x0
        for i in range(n_timesteps):
            ts, te = time_steps[i], time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward(xt,  ts, **kwargs)
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]

        return xt

class DiffusionDITNetPartsFixedExpressions2PostNormNotrans(nn.Module):
    def __init__(self, hidden_size=384, num_layers=6, num_person=31, num_embedding_channels=384):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.in_x_hand = nn.Linear(55 * 6 + 3 + 100, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 3 + 100, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        # self.time_embed = MPFourierTimeEmbedding(hidden_size, num_embedding_channels=num_embedding_channels)
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.n_seed = 8

    def forward(
        self, 
        motion, 
        timestep,
        expression,
        audio = None,
        audio_mask = None,
        txt = None,
        txt_mask = None,
        **kwargs
    ):
        x = motion
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_joint_indices = kwargs['face_joint_indices']
        trans_indices = kwargs['trans_indices']
        pid = kwargs['speaker_id']
        poses_prompt = kwargs['pose_prompt']
        prompt_mask = kwargs['pose_prompt_mask']
        trans_prompt = kwargs['trans_prompt']
        poses_prompt = torch.cat([poses_prompt, trans_prompt], dim=-1)
        
        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)
        x0 = motion

        hand = motion[..., hand_indices]
        body = motion[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        
        x = torch.cat([motion, expression], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)

        huberts = self.hubert_prj(audio)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))

        
        time_embed = self.time_embed(timestep.reshape(-1, 1))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(motion.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(poses_prompt[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(poses_prompt[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)


        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)

        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        
        return {
            "motion": x,
        }        

if __name__ == "__main__":
    net = SimpleSpeechModel().cuda()
    huberts = torch.randn(2, 60, 1024).cuda()
    pid = torch.randint(0, 31, (2, )).cuda()
    print(huberts.shape, pid.shape)
    
    r = net(huberts, pid)
    print(r.shape)