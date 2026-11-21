#!/usr/bin/env bash
#
# Train, validate, or test PASPA on TalkSHOW with a frozen WavLM Large face head.
#
# This is an opt-in TalkSHOW launcher.  It does not modify or source any BEAT
# config, path, or checkpoint.  Every external TalkSHOW dependency is supplied
# explicitly through the environment so Hydra cannot silently pick up a
# developer-machine path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Select one atomic data/BLD scale contract. The default is 64-frame blocks
# with 256-frame sequences.
TALKSHOW_PROFILE="${TALKSHOW_PROFILE:-block64_seq256}"
TALKSHOW_MODE="${TALKSHOW_MODE:-train}"
STRIDE_SIZE=20
case "$TALKSHOW_PROFILE" in
    block16_seq64)
        BLOCK_SIZE=16
        SEQ_SIZE=64
        DEFAULT_RUN_NAME=paspa_talkshow_block16_seq64
        ;;
    block32_seq128)
        BLOCK_SIZE=32
        SEQ_SIZE=128
        DEFAULT_RUN_NAME=paspa_talkshow_block32_seq128
        ;;
    block64_seq256)
        BLOCK_SIZE=64
        SEQ_SIZE=256
        DEFAULT_RUN_NAME=paspa_talkshow_block64_seq256
        ;;
    *)
        echo "Unsupported TALKSHOW_PROFILE: ${TALKSHOW_PROFILE}" >&2
        echo "Choose block16_seq64, block32_seq128, or block64_seq256" >&2
        exit 2
        ;;
esac

case "$TALKSHOW_MODE" in
    train)
        TALKSHOW_TEST_ONLY=0
        TALKSHOW_VALIDATION_ONLY=0
        ;;
    validation)
        TALKSHOW_TEST_ONLY=0
        TALKSHOW_VALIDATION_ONLY=1
        ;;
    test)
        TALKSHOW_TEST_ONLY=1
        TALKSHOW_VALIDATION_ONLY=0
        ;;
    *)
        echo "Unsupported TALKSHOW_MODE: ${TALKSHOW_MODE}" >&2
        echo "Choose train, validation, or test" >&2
        exit 2
        ;;
esac

required_env=(
    SHOW_LMDB_DIR
    TALKSHOW_FACE_CKPT
    WAVLM_DIR
    SHOW_REST_JOINT_PATH
    EMAGE_EVALTOOLS_DIR
)
for variable_name in "${required_env[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Missing required environment variable: ${variable_name}" >&2
        exit 2
    fi
    if [[ "${!variable_name}" != /* ]]; then
        echo "${variable_name} must be an absolute path: ${!variable_name}" >&2
        exit 2
    fi
done

# FGD currently concatenates this path with an asset filename in one code path,
# so retain exactly one trailing slash.  Other paths are normalized only for
# predictable Hydra snapshots; no file is created or modified here.
EMAGE_EVALTOOLS_DIR="${EMAGE_EVALTOOLS_DIR%/}/"
SHOW_LMDB_DIR="${SHOW_LMDB_DIR%/}"
WAVLM_DIR="${WAVLM_DIR%/}"
DEFAULT_TALKSHOW_VALIDATION_LMDB="$SHOW_LMDB_DIR/show_val_seq_size_${SEQ_SIZE}_stride_size_${STRIDE_SIZE}_global.lmdb"
TALKSHOW_VALIDATION_LMDB_OVERRIDDEN=0
if [[ -n "${TALKSHOW_VALIDATION_LMDB:-}" ]]; then
    TALKSHOW_VALIDATION_LMDB_OVERRIDDEN=1
fi
TALKSHOW_VALIDATION_LMDB="${TALKSHOW_VALIDATION_LMDB:-$DEFAULT_TALKSHOW_VALIDATION_LMDB}"
TALKSHOW_VALIDATION_SPLIT="${TALKSHOW_VALIDATION_SPLIT:-val}"
if [[ "$TALKSHOW_VALIDATION_LMDB" != /* ]]; then
    echo "TALKSHOW_VALIDATION_LMDB must be an absolute path: $TALKSHOW_VALIDATION_LMDB" >&2
    exit 2
fi
case "$TALKSHOW_VALIDATION_SPLIT" in
    val)
        ;;
    test)
        if (( TALKSHOW_VALIDATION_LMDB_OVERRIDDEN == 0 )); then
            echo "TALKSHOW_VALIDATION_SPLIT=test requires an explicit TALKSHOW_VALIDATION_LMDB" >&2
            exit 2
        fi
        echo "WARNING: validation is intentionally using an LMDB declared split=test; checkpoint selection will observe test data." >&2
        ;;
    *)
        echo "TALKSHOW_VALIDATION_SPLIT must be val or test: $TALKSHOW_VALIDATION_SPLIT" >&2
        exit 2
        ;;
esac
export \
    EMAGE_EVALTOOLS_DIR \
    SHOW_LMDB_DIR \
    WAVLM_DIR \
    TALKSHOW_VALIDATION_LMDB \
    TALKSHOW_VALIDATION_SPLIT

required_files=(
    "$TALKSHOW_FACE_CKPT"
    "$SHOW_REST_JOINT_PATH"
    "$WAVLM_DIR/config.json"
    "$WAVLM_DIR/preprocessor_config.json"
    "$EMAGE_EVALTOOLS_DIR/AESKConv_240_100.bin"
    "$EMAGE_EVALTOOLS_DIR/mean_vel_smplxflame_30.npy"
    "$EMAGE_EVALTOOLS_DIR/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
)
if [[ "$TALKSHOW_TEST_ONLY" == "1" ]]; then
    required_files+=(
        "$SHOW_LMDB_DIR/show_test_seq_size_${SEQ_SIZE}_stride_size_${STRIDE_SIZE}_global.lmdb"
    )
elif [[ "$TALKSHOW_VALIDATION_ONLY" == "1" ]]; then
    required_files+=(
        "$TALKSHOW_VALIDATION_LMDB"
    )
else
    required_files+=(
        "$SHOW_LMDB_DIR/show_train_seq_size_${SEQ_SIZE}_stride_size_${STRIDE_SIZE}_global.lmdb"
        "$TALKSHOW_VALIDATION_LMDB"
        "$SHOW_LMDB_DIR/show_test_seq_size_${SEQ_SIZE}_stride_size_${STRIDE_SIZE}_global.lmdb"
    )
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file does not exist: ${required_file}" >&2
        exit 2
    fi
done
for required_directory in "$WAVLM_DIR" "$EMAGE_EVALTOOLS_DIR"; do
    if [[ ! -d "$required_directory" ]]; then
        echo "Required directory does not exist: ${required_directory}" >&2
        exit 2
    fi
done
if [[ \
    ! -f "$WAVLM_DIR/model.safetensors" \
    && ! -f "$WAVLM_DIR/pytorch_model.bin" \
    && ! -f "$WAVLM_DIR/model.safetensors.index.json" \
    && ! -f "$WAVLM_DIR/pytorch_model.bin.index.json" \
]]; then
    echo "WAVLM_DIR contains no recognized model weights: $WAVLM_DIR" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_PRINT_AUTOTUNING="${TRITON_PRINT_AUTOTUNING:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export DISABLE_FLEX_COMPILE="${DISABLE_FLEX_COMPILE:-1}"
export USE_FLA="${USE_FLA:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-72}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
RUN_CONTRACT_PREFLIGHT="${RUN_CONTRACT_PREFLIGHT:-1}"
TALKSHOW_CFG_AUDIO_SCALE="${TALKSHOW_CFG_AUDIO_SCALE:-1.5}"
TALKSHOW_CFG_HISTORY_SCALE="${TALKSHOW_CFG_HISTORY_SCALE:-1.5}"
TALKSHOW_CFG_MODE="${TALKSHOW_CFG_MODE:-interaction}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

for positive_integer_name in \
    NUM_MACHINES NUM_PROCESSES MAIN_PROCESS_PORT TRAIN_BATCH_SIZE \
    NUM_EPOCHS NUM_WORKERS; do
    positive_integer_value="${!positive_integer_name}"
    if [[ ! "$positive_integer_value" =~ ^[0-9]+$ ]]; then
        echo "${positive_integer_name} must be an integer: ${positive_integer_value}" >&2
        exit 2
    fi
done
if (( NUM_MACHINES < 1 || NUM_PROCESSES < 1 || MAIN_PROCESS_PORT < 1 \
    || MAIN_PROCESS_PORT > 65535 || TRAIN_BATCH_SIZE < 1 \
    || NUM_EPOCHS < 1 )); then
    echo "machine/process/batch/epoch counts and port must be positive and valid" >&2
    exit 2
fi
if (( NUM_MACHINES != 1 )); then
    echo "This launcher supports NUM_MACHINES=1 only" >&2
    exit 2
fi
IFS=',' read -r -a VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#VISIBLE_DEVICE_LIST[@]} != NUM_PROCESSES )); then
    echo "CUDA_VISIBLE_DEVICES must contain NUM_PROCESSES=${NUM_PROCESSES} devices: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi
for visible_device in "${VISIBLE_DEVICE_LIST[@]}"; do
    if [[ -z "${visible_device//[[:space:]]/}" ]]; then
        echo "CUDA_VISIBLE_DEVICES contains an empty device entry: ${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
done
if [[ "$RUN_CONTRACT_PREFLIGHT" != "0" && "$RUN_CONTRACT_PREFLIGHT" != "1" ]]; then
    echo "RUN_CONTRACT_PREFLIGHT must be 0 or 1" >&2
    exit 2
fi
for cfg_scale_name in TALKSHOW_CFG_AUDIO_SCALE TALKSHOW_CFG_HISTORY_SCALE; do
    cfg_scale_value="${!cfg_scale_name}"
    if [[ ! "$cfg_scale_value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "${cfg_scale_name} must be a non-negative number: ${cfg_scale_value}" >&2
        exit 2
    fi
done
case "$TALKSHOW_CFG_MODE" in
    joint|independent|audio_first|state_first|interaction)
        ;;
    *)
        echo "Unsupported TALKSHOW_CFG_MODE: ${TALKSHOW_CFG_MODE}" >&2
        exit 2
        ;;
esac

# Scale-critical fields must come from TALKSHOW_PROFILE.  Otherwise the shell
# could validate one LMDB while Hydra silently trains with another sequence or
# block contract supplied at the end of the command line.
RESUME_CHECKPOINT=""
MODEL_LOAD_CHECKPOINT=""
for cli_override in "$@"; do
    # Hydra accepts plain, '+' and '++' override prefixes.  Normalize both
    # additions before matching so none can bypass the preflight contract.
    normalized_override="${cli_override#+}"
    normalized_override="${normalized_override#+}"
    case "$normalized_override" in
        talkshow_profile=*|~talkshow_profile|~talkshow_profile.*|\
        talkshow_profile.*=*|\
        talkshow_test_only=*|~talkshow_test_only|\
        talkshow_validation_only=*|~talkshow_validation_only|\
        data=*|~data|data.pose_length=*|~data.pose_length|\
        data.train.params.lmdb_path=*|\
        ~data.train.params.lmdb_path|\
        data.validation.params.lmdb_path=*|\
        ~data.validation.params.lmdb_path|\
        data.test.params.lmdb_path=*|\
        ~data.test.params.lmdb_path|\
        data.train.params.expected_split=*|\
        ~data.train.params.expected_split|\
        data.validation.params.expected_split=*|\
        ~data.validation.params.expected_split|\
        data.test.params.expected_split=*|\
        ~data.test.params.expected_split|\
        data.train.params.expected_seq_size=*|\
        ~data.train.params.expected_seq_size|\
        data.validation.params.expected_seq_size=*|\
        ~data.validation.params.expected_seq_size|\
        data.test.params.expected_seq_size=*|\
        ~data.test.params.expected_seq_size|\
        data.train.params.expected_pids=*|\
        ~data.train.params.expected_pids|\
        data.validation.params.expected_pids=*|\
        ~data.validation.params.expected_pids|\
        data.test.params.expected_pids=*|\
        ~data.test.params.expected_pids|\
        data.train.params.expected_pid_offset=*|\
        ~data.train.params.expected_pid_offset|\
        data.validation.params.expected_pid_offset=*|\
        ~data.validation.params.expected_pid_offset|\
        data.test.params.expected_pid_offset=*|\
        ~data.test.params.expected_pid_offset|\
        data.train.params.model_pid_offset=*|\
        ~data.train.params.model_pid_offset|\
        data.validation.params.model_pid_offset=*|\
        ~data.validation.params.model_pid_offset|\
        data.test.params.model_pid_offset=*|\
        ~data.test.params.model_pid_offset|\
        model=*|~model|\
        model.model.params.num_person=*|\
        ~model.model.params.num_person|\
        model.resume_from=*|model.resume_from.*=*|\
        ~model.resume_from|~model.resume_from.*|\
        model.load_from.*=*|\
        ~model.load_from|~model.load_from.*|\
        model.speech_model=*|model.speech_model.*=*|\
        ~model.speech_model|~model.speech_model.*|\
        model.speech_model_ckpt_path=*|~model.speech_model_ckpt_path|\
        trainer=*|~trainer|\
        train.num_frame_per_block=*|\
        ~train.num_frame_per_block|\
        model.speech_model.params.expected_pose_length=*|\
        ~model.speech_model.params.expected_pose_length|\
        model.speech_model.params.expected_block_size=*|\
        ~model.speech_model.params.expected_block_size)
            echo "Scale-critical override is locked by TALKSHOW_PROFILE: ${cli_override}" >&2
            exit 2
            ;;
        resume_from=*)
            resume_value="${normalized_override#resume_from=}"
            if [[ -n "$resume_value" && "$resume_value" != "null" ]]; then
                RESUME_CHECKPOINT="$resume_value"
            fi
            ;;
        model.load_from=*)
            if [[ \
                "$TALKSHOW_TEST_ONLY" != "1" \
                && "$TALKSHOW_VALIDATION_ONLY" != "1" \
            ]]; then
                echo "Scale-critical override is locked by TALKSHOW_PROFILE: ${cli_override}" >&2
                exit 2
            fi
            model_load_value="${normalized_override#model.load_from=}"
            if [[ "$model_load_value" != /*.pth ]]; then
                echo "TalkSHOW ${TALKSHOW_MODE} mode requires an absolute .pth model.load_from path: ${model_load_value}" >&2
                exit 2
            fi
            if [[ -n "$MODEL_LOAD_CHECKPOINT" ]]; then
                echo "TalkSHOW ${TALKSHOW_MODE} mode accepts exactly one model.load_from checkpoint" >&2
                exit 2
            fi
            MODEL_LOAD_CHECKPOINT="$model_load_value"
            ;;
    esac
done

if [[ "$TALKSHOW_TEST_ONLY" == "1" ]]; then
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        echo "TALKSHOW_MODE=test forbids resume_from; use model.load_from=/absolute/path/to/checkpoint.pth" >&2
        exit 2
    fi
    if [[ -z "$MODEL_LOAD_CHECKPOINT" ]]; then
        echo "TALKSHOW_MODE=test requires model.load_from=/absolute/path/to/checkpoint.pth" >&2
        exit 2
    fi
fi
if [[ "$TALKSHOW_VALIDATION_ONLY" == "1" ]]; then
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        echo "TALKSHOW_MODE=validation forbids resume_from; use model.load_from=/absolute/path/to/checkpoint.pth" >&2
        exit 2
    fi
    if [[ -z "$MODEL_LOAD_CHECKPOINT" ]]; then
        echo "TALKSHOW_MODE=validation requires model.load_from=/absolute/path/to/checkpoint.pth" >&2
        exit 2
    fi
fi

CHECKPOINT_TO_VERIFY="$RESUME_CHECKPOINT"
if [[ -n "$MODEL_LOAD_CHECKPOINT" ]]; then
    CHECKPOINT_TO_VERIFY="$MODEL_LOAD_CHECKPOINT"
fi
if [[ -n "$CHECKPOINT_TO_VERIFY" ]]; then
    python tools/show/verify_talkshow_resume_profile.py \
        --checkpoint "$CHECKPOINT_TO_VERIFY" \
        --profile "$TALKSHOW_PROFILE" \
        --expected-seq-size "$SEQ_SIZE" \
        --expected-block-size "$BLOCK_SIZE"
fi

echo "TalkSHOW mode/profile: ${TALKSHOW_MODE}/${TALKSHOW_PROFILE} (seq=${SEQ_SIZE}, block=${BLOCK_SIZE}, stride=${STRIDE_SIZE}, epochs=${NUM_EPOCHS})"
echo "TalkSHOW CFG: audio=${TALKSHOW_CFG_AUDIO_SCALE}, history=${TALKSHOW_CFG_HISTORY_SCALE}, mode=${TALKSHOW_CFG_MODE}"
if [[ "$TALKSHOW_TEST_ONLY" == "1" ]]; then
    echo "TalkSHOW data access: test only (train/validation LMDBs are not opened)"
elif [[ "$TALKSHOW_VALIDATION_ONLY" == "1" ]]; then
    echo "TalkSHOW data access: validation only (train/test LMDBs are not opened)"
    echo "TalkSHOW validation LMDB: ${TALKSHOW_VALIDATION_LMDB} (declared split=${TALKSHOW_VALIDATION_SPLIT})"
else
    echo "TalkSHOW validation LMDB: ${TALKSHOW_VALIDATION_LMDB} (declared split=${TALKSHOW_VALIDATION_SPLIT})"
fi

if [[ "$RUN_CONTRACT_PREFLIGHT" == "1" ]]; then
    python tools/show/verify_talkshow_face_condition.py \
        --checkpoint "$TALKSHOW_FACE_CKPT" \
        --wavlm-dir "$WAVLM_DIR" \
        --required-face-loss corrected
    preflight_splits=(train val test)
    if [[ "$TALKSHOW_TEST_ONLY" == "1" ]]; then
        preflight_splits=(test)
    elif [[ "$TALKSHOW_VALIDATION_ONLY" == "1" ]]; then
        preflight_splits=(val)
    fi
    for split_name in "${preflight_splits[@]}"; do
        split_lmdb_path="$SHOW_LMDB_DIR/show_${split_name}_seq_size_${SEQ_SIZE}_stride_size_${STRIDE_SIZE}_global.lmdb"
        expected_lmdb_split="$split_name"
        if [[ "$split_name" == "val" ]]; then
            split_lmdb_path="$TALKSHOW_VALIDATION_LMDB"
            expected_lmdb_split="$TALKSHOW_VALIDATION_SPLIT"
        fi
        python tools/show/verify_lmdb_contract.py \
            --lmdb-path "$split_lmdb_path" \
            --expected-split "$expected_lmdb_split" \
            --expected-seq-size "$SEQ_SIZE" \
            --expected-pids 21 22 23 24 \
            --expected-pid-offset 1 \
            --expected-max-av-duration-delta-frames 3 \
            --rest-joint-path "$SHOW_REST_JOINT_PATH" \
            --wavlm-dir "$WAVLM_DIR"
    done
fi

# New runs use a descriptive top-level directory. On resume, leave
# hydra.run.dir untouched so configs/config.yaml can resolve the checkpoint's
# experiment root and allocate its normal sibling outputN/ directory. Pure
# model.load_from test runs also leave it to an explicit caller override (the
# checkpoint sweep) or the config's normal new-run resolver.
HYDRA_RUN_ARGS=("++hydra.run.dir=outputs/${RUN_NAME}_${TIMESTAMP}")
if [[ -n "$RESUME_CHECKPOINT" || -n "$MODEL_LOAD_CHECKPOINT" ]]; then
    HYDRA_RUN_ARGS=()
fi
TALKSHOW_MODE_ARGS=()
if [[ "$TALKSHOW_TEST_ONLY" == "1" ]]; then
    TALKSHOW_MODE_ARGS=(
        "++talkshow_test_only=true"
        "train.restore_rng_state=false"
        "_train=false"
        "_validation=false"
        "_test=true"
        "_predict=false"
    )
elif [[ "$TALKSHOW_VALIDATION_ONLY" == "1" ]]; then
    TALKSHOW_MODE_ARGS=(
        "++talkshow_validation_only=true"
        "train.restore_rng_state=false"
        "_train=false"
        "_validation=true"
        "_test=false"
        "_predict=false"
    )
fi

accelerate launch \
    --num_machines="$NUM_MACHINES" \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port="$MAIN_PROCESS_PORT" \
    main.py \
    name="$RUN_NAME" \
    report_to="$REPORT_TO" \
    trainer=bld \
    metrics=distributed_emage \
    metrics.evaluator.download_path="$EMAGE_EVALTOOLS_DIR" \
    metrics.evaluator.fgd_feature_offload=false \
    +talkshow_profile="$TALKSHOW_PROFILE" \
    model=show_causal_emage_deltanet_talkshow_face \
    model.speech_model.params.expected_pose_length="$SEQ_SIZE" \
    model.speech_model.params.expected_block_size="$BLOCK_SIZE" \
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
    model.guider.cfg_audio_scale="$TALKSHOW_CFG_AUDIO_SCALE" \
    model.guider.cfg_history_scale="$TALKSHOW_CFG_HISTORY_SCALE" \
    +model.guider.cfg_mode="$TALKSHOW_CFG_MODE" \
    model.guider.cfg_schedule_type=interval \
    +model.guider.cfg_schedule_kwargs.start=100 \
    +model.guider.cfg_schedule_kwargs.end=600 \
    +model.guider.cfg_schedule_kwargs.outside_scale=1.0 \
    train.ema_update=false \
    train.debug_gradient_interval=-1 \
    train.lr=0.001 \
    train.lr_schedule=piecewise_constant \
    +train.lr_schedule_args.step_rules=\"1:20000,0.1\" \
    data.data_path_1="$EMAGE_EVALTOOLS_DIR" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.validation.params.lmdb_path="$TALKSHOW_VALIDATION_LMDB" \
    data.validation.params.expected_split="$TALKSHOW_VALIDATION_SPLIT" \
    data.validation_batch_size=1 \
    data.test_batch_size=1 \
    data.num_workers="$NUM_WORKERS" \
    data.audio_sr=16000 \
    data.audio_fps=16000 \
    audio.audio_encoder.local_model_path="$WAVLM_DIR" \
    train.visualize_log_interval=1 \
    train.visualize=false \
    scheduler.sampler.params.steps=10 \
    train.task=audio2motion \
    train.normalize=false \
    train.finetune_speech_model=false \
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
    train.teacher_forcing=true \
    train.num_frame_per_block="$BLOCK_SIZE" \
    train.independent_first_frame=false \
    train.save_model_epochs=10 \
    train.check_eval_n_epochs=5 \
    train.keep_last=false \
    train.sanity_check_steps=0 \
    train.num_epochs="$NUM_EPOCHS" \
    data.load_audio_feat=true \
    _train=true \
    _validation=false \
    _test=true \
    _predict=false \
    "${HYDRA_RUN_ARGS[@]}" \
    "$@" \
    "${TALKSHOW_MODE_ARGS[@]}"
