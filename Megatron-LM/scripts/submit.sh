#= Prelude =#
# Some constants.
SCRIPT_VERSION=repro
SEQ_LEN=4096
TOKENIZER=mistralai/Mistral-Nemo-Base-2407
CODE_PATH=$FOG_PATH/Megatron-LM

ACTIVATION=swiglu
QK_IMPL=apex
DYT_ALPHA=1.0
NODES=1
TIME=1-00:00:00
MARGIN=0
FP8LEN=1024
WEIGHT_DECAY=0.1
MIN_LR=1e-8
NORM=rms

# Precision aware optimizer.
DEF_GRAD_DTYPE=fp32
DEF_PARAM_DTYPE=fp32
DEF_M1_DTYPE=fp32
DEF_M2_DTYPE=fp32
GRAD_DTYPE=$DEF_GRAD_DTYPE
PARAM_DTYPE=$DEF_PARAM_DTYPE
M1_DTYPE=$DEF_M1_DTYPE
M2_DTYPE=$DEF_M2_DTYPE

# Usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 300/1/8"
	echo "Options:"
	# Misc settings.
	echo " --nodes <nodes>: How many nodes to use"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --time (default=$TIME): Change the sbatch time limit"
	# FP8 settings.
	echo " --fp8: Enables fp8"
	echo " --fp8-dpa: Enables fp8 DPA"
	echo " --fp8-margin: fp8 margin"
	echo " --fp8-len <int>: fp8 history length"
	echo " --fp8-lmhead: Enables fp8 lmhead"
	# Precision aware optimizer.
	echo " --dtype-grad <fp32/bf16> (default=$DEF_GRAD_DTYPE): Main gradient dtype"
	echo " --dtype-param <fp32/fp16> (default=$DEF_PARAM_DTYPE): Main parameter dtype"
	echo " --dtype-m1 <fp32/fp16/fp8> (default=$DEF_M1_DTYPE): Optimizer first moment dtype"
	echo " --dtype-m2 <fp32/fp16/fp8> (default=$DEF_M2_DTYPE): Optimizer second moment dtype"
	# Training settings..
	echo " --tokens <int>: Amount of tokens to train with (in B)."
	echo " --lr <float>: Learning rate."
	echo " --cooldown-wd: When set, weight decay will be cooldowned."
	# Architecture settings.
	echo " --init <float>: Change init std."
	echo " --no-prenorm: Disables pre-layernorm."
	echo " --finalln: Enables final layer-norm."
	echo " --postnorm: Enables post-layernorm."
	echo " --qknorm: Enables qk norm."
	echo " --qkimpl <te|apex|torch>: QK Implementation."
	echo " --qkinit <float>: Sets qk norm initialization to this value."
	echo " --qk-dyt: Enables QK DyT (instead of RMS)"
	echo " --dyt-alpha <float>: QK DyT alpha initialization"
	echo " --softmax-scale: Sets attention softmax scale."
	echo " --layerscale: Enables layerscale."
	echo " --layerscale-value <float>: Layerscale init value."
	echo " --input-upscale: Upscales input by 1/std."
	echo " --alpha: MLP alpha."
	echo " --activation (default=$ACTIVATION): MLP activation. Choices=[swiglu, gelu, xielu, sswiglu]."
	# Opt settings.
	echo " --decay-qkgains: Decay qk layernorm gains"
	echo " --no-train-qk-gains: Don't train QK layernorm gains"
	# Logs.
	echo " --wandb-name <str>: Specify wandb name."
}

if [[ $# -eq 0 ]]; then
	>&2 echo Invalid argument count: $#
	usage
	exit 1
fi

# Define some variables depending on size.
EXTRA_ARGS=()
TP=1
PP=1
UNTIE=true
INIT_STD=0.02
if [[ $1 -eq 300 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS=8
	GBS=128
	ITERS=2000  # 1BT.
	LR=0.001
	SIZE=390M
	SAVE_FREQ=10000
	DEF_TOKENS=50
	UNTIE=false
	INTERMEDIATE_METRICS_INTERVAL=10
elif [[ $1 -eq 1 ]]; then 
	# batch_size: ~1.05M.
	LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS=4
	GBS=256
	ITERS=1000  # 1BT.
	LR=0.00025
	SIZE=1.5B
	SAVE_FREQ=5000
	DEF_TOKENS=125
	INTERMEDIATE_METRICS_INTERVAL=100
elif [[ $1 -eq 8 ]]; then 
	# batch_size: ~2.1M.
	TP=1
	LAYERS=32
	HIDDEN_SIZE=4096
	FFN_SIZE=14336
	NUM_HEADS=32
	NUM_QUERY_GROUPS=8
	MBS=1
	GBS=512
	ITERS=500  # 1BT.
	LR=0.0003
	SIZE=8B
	SAVE_FREQ=2500
	DEF_TOKENS=125
	INTERMEDIATE_METRICS_INTERVAL=200
else
	>&2 echo "Invalid model size: $1"
	usage
	exit 1
fi
shift

# Now get the general options.
TOKENS=$DEF_TOKENS
ENVS=""
SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--nodes)
			NODES=$2; shift 2;;
		--time)
			TIME=$2; shift 2;;
		--fp8)
			FP8=true; shift;;
		--fp8-dpa)
			FP8DPA=true; shift;;
		--fp8-margin)
			MARGIN=$2; shift 2;;
		--fp8-len)
			FP8LEN=$2; shift 2;;
		--fp8-lmhead)
			FP8_LMHEAD=true; shift;; 
		--dtype-grad)
			GRAD_DTYPE=$2; shift 2;;
		--dtype-param)
			PARAM_DTYPE=$2; shift 2;;
		--dtype-m1)
			M1_DTYPE=$2; shift 2;;
		--dtype-m2)
			M2_DTYPE=$2; shift 2;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--tokens)
			TOKENS=$2; shift 2;;
		--init)
			NEW_INIT_STD=$2; shift 2;;
		--lr)
			LR=$2; 
			CHANGED_LR=true
			shift 2;;
		--cooldown-wd)
			COOLDOWN_WD=true; shift;;
		--no-prenorm)
			PRENORM=false; shift;;
		--finalln)
			FINALNORM=true; shift;;
		--postnorm)
			POSTNORM=true; shift;;
		--qknorm)
			QKNORM=true; shift;;
		--qkimpl)
			QK_IMPL=$2; shift 2;;
		--qkinit)
			QK_INIT=$2; shift 2;;
		--qk-dyt)
			QK_DYT=true; shift;;
		--dyt-alpha)
			DYT_ALPHA=$2; shift 2;;
		--softmax-scale)
			SOFTMAX_SCALE=$2; shift 2;;
		--layerscale)
			LAYERSCALE=true; shift;;
		--layerscale-value)
			LAYERSCALE_VALUE=$2; shift 2;;
		--input-upscale)
			INPUT_UPSCALE=true; shift;;
		--activation)
			ACTIVATION=$2; shift 2;;
		--alpha)
			ALPHA=$2; shift 2;;
		--decay-qkgains)
			DECAY_GAINS=true; shift;;
		--no-train-qk-gains)
			NOTRAIN_GAINS=true; shift;;
		--wandb-name)
			WANDB_NAME=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

#= MIDDLE: Set up arguments. =#
FP8_ARGS=()
if [[ $FP8 = true ]]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS+=(--fp8-margin $MARGIN --fp8-format hybrid --fp8-amax-history-len $FP8LEN --fp8-amax-compute-algo max)
	if [[ $FP8DPA = true ]]; then
		SUFFIX=$SUFFIX-fp8dpa
		FP8_ARGS+=(--fp8-dot-product-attention)
	fi
	if [[ $MARGIN -ne 0 ]]; then
		SUFFIX=$SUFFIX-fp8m$MARGIN
	fi
	if [[ $FP8LEN -ne 1024 ]]; then
		SUFFIX=$SUFFIX-fp8l$FP8LEN
	fi
	if [[ $FP8_LMHEAD = true ]]; then
		SUFFIX=$SUFFIX-fp8LM
		FP8_ARGS+=(--lm-head-in-fp8)
	fi
fi

if [[ $GRAD_DTYPE != $DEF_GRAD_DTYPE ]] || [[ $PARAM_DTYPE != $DEF_PARAM_DTYPE ]] || [[ $M1_DTYPE != $DEF_M1_DTYPE ]] || [[ $M2_DTYPE != $DEF_M2_DTYPE ]]; then
	if [[ $GRAD_DTYPE = bf16 ]] && [[ $PARAM_DTYPE = fp16 ]] && [[ $M1_DTYPE = fp8 ]] && [[ $M2_DTYPE = fp16 ]]; then
		SUFFIX=$SUFFIX-fp8OPT
	elif [[ $GRAD_DTYPE = bf16 ]] && [[ $PARAM_DTYPE = fp16 ]] && [[ $M1_DTYPE = fp16 ]] && [[ $M2_DTYPE = fp16 ]]; then
		SUFFIX=$SUFFIX-fp16OPT
	else
		SUFFIX=$SUFFIX-dtypeG${GRAD_DTYPE}P${PARAM_DTYPE}M1${M1_DTYPE}M2${M2_DTYPE}
	fi
	FP8_ARGS+=(--use-precision-aware-optimizer --main-grads-dtype $GRAD_DTYPE --main-params-dtype $PARAM_DTYPE --exp-avg-dtype $M1_DTYPE --exp-avg-sq-dtype $M2_DTYPE)
fi

ARCH_ARGS=()
if [[ ! -z ${NEW_INIT_STD+x} ]]; then
	SUFFIX=$SUFFIX-std$NEW_INIT_STD
	INIT_STD=$NEW_INIT_STD
fi
if [[ $PRENORM = false ]]; then
	SUFFIX=$SUFFIX-nopre
	ARCH_ARGS+=(--no-pre-layer-norm)
	if [[ $FINALNORM = true ]]; then
		SUFFIX=$SUFFIX-finalLN
	else
		ARCH_ARGS+=(--no-final-layernorm)
	fi
fi
if [[ $POSTNORM = true ]]; then
	SUFFIX=$SUFFIX-postln
	ARCH_ARGS+=(--post-layer-norm)
fi

if [[ $QKNORM = true ]]; then
	ARCH_ARGS+=(--qk-layernorm)
	if [[ $QK_DYT = true ]]; then
		SUFFIX=$SUFFIX-qkDyT
		if [[ $DYT_ALPHA != 1.0 ]]; then
			SUFFIX=$SUFFIX-DyTalpha$DYT_ALPHA
		fi
		ARCH_ARGS+=(--qk-dyt --dyt-alpha-init $DYT_ALPHA)
	else
		SUFFIX=$SUFFIX-qknorm
		if [[ $QK_IMPL != apex ]]; then
			SUFFIX=$SUFFIX-qk$QK_IMPL
		fi
		ARCH_ARGS+=(--qknorm-impl $QK_IMPL)
		if [[ $QK_IMPL = torch ]]; then
			ARCH_ARGS+=(--no-persist-layer-norm)
		fi
	fi

	if [[ ! -z ${QK_INIT+x} ]]; then
		SUFFIX=$SUFFIX-qkinit$QK_INIT
		ARCH_ARGS+=(--qknorm-init $QK_INIT)
	fi
fi
if [[ ! -z ${SOFTMAX_SCALE+x} ]]; then
	SUFFIX=$SUFFIX-softmax$SOFTMAX_SCALE
	ARCH_ARGS+=(--softmax-scale $SOFTMAX_SCALE)
fi
if [[ $LAYERSCALE = true ]]; then
	SUFFIX=$SUFFIX-ls
	if [[ ! -z ${LAYERSCALE_VALUE+x} ]]; then
		BETA=$LAYERSCALE_VALUE
		SUFFIX=$SUFFIX$BETA
	else
		BETA=$(echo "print(1/$LAYERS**0.5)" | python3)
	fi
	if [[ $POSTNORM = true ]] && [[ $PRENORM = false ]]; then
		ARCH_ARGS+=(--layernorm-init $BETA)
	else
		ARCH_ARGS+=(--layer-scale $BETA)
	fi
fi
if [[ $INPUT_UPSCALE = true ]]; then
	SUFFIX=$SUFFIX-is
	MULT=$(echo "print(1/$INIT_STD)" | python3)
	ARCH_ARGS+=(--input-embeddings-multiplier $MULT)
fi

if [[ $ACTIVATION = gelu ]]; then
	FFN_SIZE=$((3*$FFN_SIZE/2))
	SUFFIX=$SUFFIX-$ACTIVATION
elif [[ $ACTIVATION = xielu ]]; then
	if [[ $CSCS_XIELU = true ]]; then
		MAYBE_INSTALL_XIELU="pip install --no-build-isolation --no-deps git+https://github.com/nickjbrowning/XIELU.git"
	fi
	FFN_SIZE=$((3*$FFN_SIZE/2))
	SUFFIX=$SUFFIX-$ACTIVATION
	ARCH_ARGS+=(--$ACTIVATION)
elif [[ $ACTIVATION = sswiglu ]]; then
	SUFFIX=$SUFFIX-sswiglu
	ARCH_ARGS+=(--scaled-swiglu --swiglu)
elif [[ $ACTIVATION = swiglu ]]; then
	ARCH_ARGS+=(--swiglu)
else
	>&2 echo Unknown activation: $ACTIVATION
	exit 1
fi

if [[ ! -z ${ALPHA+x} ]]; then
	SUFFIX=$SUFFIX-mlp$ALPHA
	ARCH_ARGS+=(--mlp-alpha $ALPHA)
fi

OPT_ARGS=()
if [[ $DECAY_GAINS = true ]]; then
	SUFFIX=$SUFFIX-decayQKgains
	OPT_ARGS+=(--weight-decay-qk-gains)
fi
if [[ $NOTRAIN_GAINS = true ]]; then
	SUFFIX=$SUFFIX-ntQKgain
	OPT_ARGS+=(--no-train-qk-gains)
fi

if [[ $CHANGED_LR = true ]]; then
	SUFFIX=$SUFFIX-lr$LR
fi

WARMUP=$((5*ITERS/2))  # 2.5BT.
EVAL_INTERVAL=$((ITERS/2))  # 500MT.
EVAL_ITERS=$((ITERS/100))  # 10MT.
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
fi
ITERS=$((ITERS*TOKENS))
DECAY_ITERS=$(($ITERS/5))

if [[ $COOLDOWN_WD = true ]]; then
	SUFFIX=$SUFFIX-coolWD
	MIN_COOLDOWN=$(echo "print($MIN_LR*$WEIGHT_DECAY/$LR)" | python3)
	OPT_ARGS+=(--end-weight-decay-cooldown $MIN_COOLDOWN --weight-decay-cooldown-iters $DECAY_ITERS --weight-decay-cooldown-style 1-sqrt)
fi

EXTRA_LOGS=()
EXTRA_OPT=()

SUFFIX=$SUFFIX$EXTRA_NAME
EXTRA_LOGS+=(
	--log-validation-ppl-to-tensorboard
	--log-intermediate-metrics mean rms kurtosis
	--log-intermediate-metrics-interval $INTERMEDIATE_METRICS_INTERVAL
	--log-params-norm-per-param
	--log-num-zeros-in-grad
	--log-params-norm
	--log-weight-decay
)

# Final preparations.
NAME=llama${SIZE}$SUFFIX
JOB_NAME=$NAME
ROOT_PATH=$TRAIN_ROOT/$NAME
SAVE_PATH=$ROOT_PATH/checkpoints
DIFFS_PATH=$ROOT_PATH/diffs
TRIGGER_DIR=$ROOT_PATH/triggerdir
TRIGGER_LOCK=$TRIGGER_DIR/training_lock
mkdir -p $SAVE_PATH
mkdir -p $DIFFS_PATH
mkdir -p $TRIGGER_DIR


if [[ -z ${WANDB_NAME+x} ]]; then
	WANDB_NAME=$NAME
fi

if [[ ${#WANDB_NAME} -gt 128 ]]; then
	>&2 echo "WANDB_NAME is too long (it shouldn't exceed 128 characters): $WANDB_NAME"
	exit 1
fi

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#
ENVS="$ENVS CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=\\\$SLURM_CPUS_PER_TASK WANDB_RESUME=allow WANDB_RUN_ID=$NAME"
WANDB_PROJECT=op_$SCRIPT_VERSION

LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	$MAYBE_VIRTUAL
	--seq-length $SEQ_LEN
	--max-position-embeddings $SEQ_LEN
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER
	--normalization RMSNorm
	--position-embedding-type rope
	--attention-softmax-in-fp32
	--disable-bias-linear
	--transformer-impl transformer_engine
	--num-layers $LAYERS
	--hidden-size $HIDDEN_SIZE
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--rotary-base 500000
	--rotary-percent 1.0
	--use-rope-scaling
	--bf16
	--adam-eps 0.00000001
	--norm-epsilon 0.00001
)
if [[ $UNTIE = true ]]; then
	LLAMA_ARGS+=(--untie-embeddings-and-output-weights)
fi

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--weight-decay $WEIGHT_DECAY
	--adam-beta1 0.9 
	--adam-beta2 0.95
	--init-method-std $INIT_STD
	--clip-grad 1.0
	--lr $LR
	--min-lr $MIN_LR
	--trigger-path $TRIGGER_DIR
	--exit-signal-handler
)

DATA_ARGS=(
	--data-path ${TOKENIZED_DATA_PATH}_text_document
	--data-cache-path $DATA_CACHE_PATH
	--split 99,1,0
)

LOGGING=(
	--log-interval 1
	--save-interval $SAVE_FREQ
	--save $SAVE_PATH
	--load $SAVE_PATH
	--tensorboard-dir $ROOT_PATH/tensorboard
	--eval-interval $EVAL_INTERVAL
	--eval-iters $EVAL_ITERS
	--wandb-project $WANDB_PROJECT
	--wandb-exp-name $WANDB_NAME
	--wandb-save-dir $ROOT_PATH/wandb
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-throughput
	--log-timers-to-tensorboard
	--log-progress
	--log-memory-to-tensorboard
)
LOGGING=(${LOGGING[@]} ${EXTRA_LOGS[@]})

SCHEDULER_ARGS=(
	--lr-decay-style WSD
	--lr-wsd-decay-style 1-sqrt
	--lr-wsd-decay-iters $DECAY_ITERS
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--overlap-grad-reduce
	--async-save
 	--use-distributed-optimizer
)
EXTRA_ARGS=(${EXTRA_ARGS[@]} ${EXTRA_OPT[@]})

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]} ${ARCH_ARGS[@]} ${OPT_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="python pretrain_gpt.py $ARGS"

mkdir -p $ROOT_PATH
cat > $ROOT_PATH/submission.sbatch <<- EOM
#!/bin/bash
#SBATCH --time=$TIME
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$ROOT_PATH/slurmlogs/%j.out
#SBATCH --error=$ROOT_PATH/slurmlogs/%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=460000
#SBATCH --environment=$CONTAINER_PATH
#SBATCH --exclusive
$MAYBE_SIGNAL

echo Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

# Log git status.
cd $CODE_PATH
echo ---------
echo git status:
git status
echo git log:
git log -n 1
echo ---------
git diff > $DIFFS_PATH/\$SLURM_JOBID.diff

export MASTER_ADDR=\$(hostname)
export WORLD_SIZE=\$SLURM_NPROCS
export MASTER_PORT=25678

if [[ -f $TRIGGER_LOCK ]]; then
	# An ongoing trining is running, let's send the exit and save trigger.
	touch $TRIGGER_DIR/save
	touch $TRIGGER_DIR/exit
	while [[ -f $TRIGGER_LOCK ]]; do
		echo Ongoing training detected... Waiting a bit...
		sleep 10
	done
	# At this point the other run deleted the TRIGGER_LOCK, so let's also delete the exit signal and reinit the trigger lock for future jobs.
	echo Previous training run finished, proceeding
	rm $TRIGGER_DIR/save
	rm $TRIGGER_DIR/exit
fi
echo Creating trigger lock: $TRIGGER_LOCK
touch $TRIGGER_LOCK

srun -lu --cpus-per-task \$SLURM_CPUS_PER_TASK bash -c "
	cd $CODE_PATH
	$MAYBE_INSTALL_XIELU
	export PYTHONPATH=\$PWD
	export RANK=\\\$SLURM_PROCID
	export LOCAL_RANK=\\\$SLURM_LOCALID
	export $ENVS
	numactl --membind=0-3 env $CMD
"

# Let's indicate that there is no training run in progress anymore.
echo Removing trigger lock
rm $TRIGGER_LOCK
echo Goodbye
EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"

sbatch $ROOT_PATH/submission.sbatch
