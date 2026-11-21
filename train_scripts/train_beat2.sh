#!/usr/bin/env bash
# Main PASPA training recipe for BEAT2/BEAT-X.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

required_env=(
    BEAT2_TRAIN_LMDB
    BEAT2_VALIDATION_LMDB
    BEAT2_TEST_LMDB
)
for variable_name in "${required_env[@]}"; do
    if [[ -z "${!variable_name:-}" || "${!variable_name}" != /* ]]; then
        echo "${variable_name} must be an absolute path" >&2
        exit 2
    fi
    if [[ ! -f "${!variable_name}" ]]; then
        echo "Required LMDB does not exist: ${!variable_name}" >&2
        exit 2
    fi
done

SPEECH_MODEL_CKPT="${SPEECH_MODEL_CKPT:-$REPO_ROOT/tools/best.pt}"
WAVLM_DIR="${WAVLM_DIR:-$REPO_ROOT/tools/wavlm-libri-clean-100h-large}"
EMAGE_EVALTOOLS_DIR="${EMAGE_EVALTOOLS_DIR:-$REPO_ROOT/tools/EMAGE/emage_evaltools}"
EMAGE_EVALTOOLS_DIR="${EMAGE_EVALTOOLS_DIR%/}/"
for required_file in \
    "$SPEECH_MODEL_CKPT" \
    "$WAVLM_DIR/config.json" \
    "$EMAGE_EVALTOOLS_DIR/AESKConv_240_100.bin" \
    "$EMAGE_EVALTOOLS_DIR/mean_vel_smplxflame_30.npy" \
    "$EMAGE_EVALTOOLS_DIR/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required asset does not exist: $required_file" >&2
        exit 2
    fi
done
export SPEECH_MODEL_CKPT WAVLM_DIR EMAGE_EVALTOOLS_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_PRINT_AUTOTUNING="${TRITON_PRINT_AUTOTUNING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
# DISABLE_FLEX_COMPILE=1 disables torch.compile wrapper for flex_attention
export DISABLE_FLEX_COMPILE="1"
# Enable FLA (flash-linear-attention) kernels if available
export USE_FLA="1"
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-72}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RUN_NAME="${RUN_NAME:-paspa_beat2}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

for positive_integer_name in \
    NUM_PROCESSES MAIN_PROCESS_PORT TRAIN_BATCH_SIZE NUM_EPOCHS NUM_WORKERS; do
    positive_integer_value="${!positive_integer_name}"
    if [[ ! "$positive_integer_value" =~ ^[0-9]+$ ]]; then
        echo "${positive_integer_name} must be an integer: ${positive_integer_value}" >&2
        exit 2
    fi
done
if (( NUM_PROCESSES < 1 || MAIN_PROCESS_PORT < 1 \
    || MAIN_PROCESS_PORT > 65535 || TRAIN_BATCH_SIZE < 1 \
    || NUM_EPOCHS < 1 )); then
    echo "process/batch/epoch counts and port must be positive and valid" >&2
    exit 2
fi
IFS=',' read -r -a VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#VISIBLE_DEVICE_LIST[@]} != NUM_PROCESSES )); then
    echo "CUDA_VISIBLE_DEVICES must contain NUM_PROCESSES=${NUM_PROCESSES} devices: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

accelerate launch \
    --num_machines=1 \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port="$MAIN_PROCESS_PORT" \
    main.py \
    name="$RUN_NAME" \
    report_to="$REPORT_TO" \
    trainer=bld \
    metrics=distributed_emage \
    metrics.evaluator.download_path="$EMAGE_EVALTOOLS_DIR" \
    metrics.evaluator.fgd_feature_offload=false \
    model=causal_emage_deltanet \
    model.speech_model_ckpt_path="$SPEECH_MODEL_CKPT" \
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
    data.data_path_1="$EMAGE_EVALTOOLS_DIR" \
    data.train.params.lmdb_path="$BEAT2_TRAIN_LMDB" \
    data.validation.params.lmdb_path="$BEAT2_VALIDATION_LMDB" \
    data.test.params.lmdb_path="$BEAT2_TEST_LMDB" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.validation_batch_size=1 \
    data.test_batch_size=1 \
    data.num_workers="$NUM_WORKERS" \
    data.audio_sr=16000 \
    data.audio_fps=16000 \
    audio.audio_encoder.local_model_path="$WAVLM_DIR" \
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
    train.save_model_epochs=10 \
    train.check_eval_n_epochs=5 \
    train.keep_last=false \
    train.sanity_check_steps=0 \
    train.num_epochs="$NUM_EPOCHS" \
    data.load_audio_feat=true \
    _train=true \
    _validation=true \
    _test=true \
    _predict=false \
    ++hydra.run.dir="outputs/${RUN_NAME}_${TIMESTAMP}" \
    "$@"
