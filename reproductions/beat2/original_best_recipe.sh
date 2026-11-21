#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash tests/bash/test_eval_pth_dir_causal_geoloss_emage_deltanet_block_fuse_tmod_min_snr.sh <CHECKPOINT_DIR> [HYDRA_EXTRA_OVERRIDES...]"
    exit 1
fi

CKPT_DIR="$1"
shift

export CUDA_VISIBLE_DEVICES=5,
export TRITON_PRINT_AUTOTUNING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HYDRA_FULL_ERROR=1
# DISABLE_FLEX_COMPILE=1 disables torch.compile wrapper for flex_attention
export DISABLE_FLEX_COMPILE="1"
# Enable FLA (flash-linear-attention) kernels if available
export USE_FLA="1"
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_TO=tensorboard
name=causal_geoloss_emage_deltanet_mono_logvar_detach_naive_lognorm6_audio_cfg_ucg01_hist_p07_n01
python scripts/eval_pth_dir.py \
    name=$name \
    report_to=$REPORT_TO \
    trainer=bld \
    metrics=distributed_emage \
    metrics.evaluator.fgd_feature_offload=false \
    model=causal_emage_deltanet \
    model.load_from=$CKPT_DIR \
    +model.model_wrapper.params.return_logvar=true \
    +model.model_wrapper.params.logvar_channels=128 \
    model.local_attn_size=256 \
    model.norm1_type=liger-rms \
    model.model.params.pos_embed_type=liger-rope \
    model.model.params.block_config.params.attn_type=naive \
    model.model.params.block_config.params.deltanet_use_triton_kernel=true \
    model.max_attention_size=256 \
    +model.model.params.block_config.params.history_cfg_enabled=true \
    +model.model.params.block_config.params.history_cfg_p_clean=0.7 \
    +model.model.params.block_config.params.history_cfg_p_null=0.1 \
    +model.model.params.block_config.params.deltanet_use_learnable_init_state=false \
    +model.model.params.block_config.params.history_cfg_null_state_mode=learned \
    model.guider.guider_type=classifier-free-bld \
    model.guider.ucg_prob=0.1 \
    model.guider.cfg_audio_scale=1.5 \
    model.guider.cfg_history_scale=1.5 \
    +model.guider.cfg_mode=interaction \
    model.guider.cfg_schedule_type=interval \
    +model.guider.cfg_schedule_kwargs.start=100 \
    +model.guider.cfg_schedule_kwargs.end=600 \
    +model.guider.cfg_schedule_kwargs.outside_scale=1.0 \
    train.ema_update=false \
    train.debug_gradient_interval=-1 \
    train.lr=0.001 \
    train.lr_schedule=piecewise_constant \
    +train.lr_schedule_args.step_rules=\"1:20000,0.1\" \
    data=beatx_global_256_lmdb \
    data.train_batch_size=72 \
    data.validation_batch_size=1 \
    data.test_batch_size=1 \
    data.num_workers=8 \
    data.audio_sr=16000 \
    data.audio_fps=16000 \
    data.pose_length=256 \
    train.visualize_log_interval=1 \
    train.visualize=false \
    scheduler.sampler.params.steps=10 \
    train.task=audio2motion \
    train.normalize=false \
    scheduler=causal_euler_geoloss_joint \
    scheduler.weighting_scheme=logit_normal \
    scheduler.logit_mean=-1.0 \
    scheduler.overlap_length=8 \
    scheduler.scheduler.loss_type=emage_mse \
    scheduler.scheduler.logvar_clamp_min=none \
    scheduler.scheduler.logvar_clamp_max=none \
    scheduler.monotonic_blocks=true \
    scheduler.weak_monotonic_enable=false \
    scheduler.use_trans_for_fk=true \
    +scheduler.scheduler.loss_weighting_config=null \
    +scheduler.scheduler.other_lambdas.lambda_rot6d=1.0 \
    scheduler.scheduler.other_lambdas.lambda_translation=1.0 \
    scheduler.scheduler.other_lambdas.lambda_struct_loss=1.0 \
    scheduler.scheduler.other_lambdas.lambda_rotation=0.0 \
    scheduler.scheduler.other_lambdas.lambda_latent_loss=0.0 \
    scheduler.scheduler.other_lambdas.lambda_vel_rcxyz=0.0 \
    scheduler.scheduler.other_lambdas.lambda_acc_rcxyz=0.0 \
    scheduler.scheduler.other_lambdas.lambda_rcxyz=1.0 \
    scheduler.scheduler.other_lambdas.lambda_vel_trans=0.0 \
    scheduler.scheduler.other_lambdas.lambda_acc_trans=0.0 \
    scheduler.scheduler.other_lambdas.lambda_block_boundary=1.0 \
    scheduler.scheduler.geometric_loss_logvar_mode=detach \
    train.eval=true \
    train/optimizer=torchmuon \
    train.optimizer_constructor.params.nonhidden_names=[style_embed,hand_in,clean_hand_in,body_in,clean_body_in,xhand_in,xbody_in,hand_fuse,body_fuse,hand_final_layer,body_final_layer,motion_t_embedder,context_embedder,logvar_embed,logvar_pred,null_state,cross_attn,cross_attn2] \
    +train.teacher_forcing=true \
    +train.num_frame_per_block=64 \
    +train.independent_first_frame=false \
    data.load_audio_feat=true \
    data.data_path_1=/mnt/disk4T/ly/Sana/tools/EMAGE/emage_evaltools/ \
    data.train.params.lmdb_path=/mnt/disk4T/ly/Sana/.cache/train_seq_size_256_stride_size_20_global.lmdb \
    data.validation.params.lmdb_path=/mnt/disk4T/ly/Sana/.cache/test_seq_size_256_stride_size_20_global_ids_2.lmdb \
    data.test.params.lmdb_path=/mnt/disk4T/ly/Sana/.cache/test_seq_size_256_stride_size_20_global_ids_2.lmdb \
    data.predict.params.smplx_data_path=/mnt/disk4T/ly/BEAT2/beat_english_v2.0.0/smplxflame_30/16_jorge_1_3_3.npz \
    data.predict.params.audio_path=/mnt/disk4T/ly/BEAT2/beat_english_v2.0.0/wave16k/16_jorge_1_3_3.wav \
    metrics.evaluator.download_path=/mnt/disk4T/ly/Sana/tools/EMAGE/emage_evaltools/ \
    model.speech_model_ckpt_path=/mnt/disk4T/ly/Sana/tools/best.pt \
    audio.audio_encoder.local_model_path=/mnt/disk4T/ly/Sana/tools/wavlm-libri-clean-100h-large \
    +eval.cleanup_non_best=no \
    "$@"