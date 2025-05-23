#!/bin/bash


MODEL_FOLDER_NAME=$1
i=$2

# Unchanged
TRANSFORMERS_DIR=$FOG_PATH/fogs-transformers
EVAL_DIR=$SFOG_PATH/eval
LM_EVAL_DIR=$EVAL_DIR/lm-evaluation-harness/
RESULT_DIR=$EVAL_DIR/output/$MODEL_FOLDER_NAME
LOGS_DIR=$EVAL_DIR/logs
HF_CHECKPOINT_DIR=$TRAIN_ROOT/hf_ckpts
HF_CHECKPOINT_PATH=$HF_CHECKPOINT_DIR/$MODEL_FOLDER_NAME
# Other
TOKENIZER=mistralai/Mistral-Nemo-Base-2407
ATTN_IMPL=flash_attention_2
DTYPE=bfloat16
TASKS=fog_eval
mkdir -p $LOGS_DIR
mkdir -p $RESULT_DIR
COMMON_EVAL_ARGS="--trust_remote_code --batch_size=auto --tasks=$TASKS --output=$RESULT_DIR --max_batch_size 256"

# necessary 
MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_ADDR
export HF_HOME=$TRAIN_ROOT/HF_cache
mkdir -p $HF_HOME


## Custom packages
cd $TRANSFORMERS_DIR
# pip install -e .
echo "not installing custom transformers because done in conversion..."
echo "If not done yet, stop and save default.yaml (with the list of benchmarks) in lm-evaluation-harness/lm_eval/tasks/fog_eval/, after cloning lm-eval !"
cd $LM_EVAL_DIR
# if i passed as $2 is 0 then do not install
if [[ $i == 0 ]]; then
    echo "installing lm evaluation harness"
    pip install -e .[api]
else
    echo "not installing custom transformers because done in conversion..."
fi



echo "Running evaluation for model $MODEL_FOLDER_NAME"
## Launch
accelerate launch -m lm_eval \
    --cache_requests true \
    --model=hf \
    --model_args="pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER,max_length=4096,attn_implementation=$ATTN_IMPL,dtype=$DTYPE" \
    $COMMON_EVAL_ARGS 


# Save the current sbatch script in the RESULT_DIR
cp $0 $RESULT_DIR/$(basename $0)
echo "Script has been copied to $RESULT_DIR"