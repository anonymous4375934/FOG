#!/bin/bash
# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.


MODEL_NAME=$1
i=$2

# Avoid multiple installations
if [[ $i == 0 ]]; then
    echo "installing custom transformers"
    cd $FOG_PATH/fog_transformers
    pip install -e .
    echo "-------------"
fi

CKPT_PATH=$TRAIN_ROOT/$MODEL_NAME/checkpoints/
MEGATRON_PATH=$FOG_PATH/Megatron-LM
export PYTHONPATH=$MEGATRON_PATH
cd $MEGATRON_PATH

# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_PATH/scripts/conversion/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=$TRAIN_ROOT/torch_ckpts/$MODEL_NAME
mkdir -p $TORCH_CKPT_SAVE_PATH
# [core (torch) --> HF] dependencies
HF_SAVE_DIR=$TRAIN_ROOT/hf_ckpts/
SAVE_DIR=$HF_SAVE_DIR/$MODEL_NAME
mkdir -p $SAVE_DIR
LOADER=core
SAVER=fog_hf

# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH
else
    LOAD_DIR=$CKPT_PATH
    echo "Skipping torch_dist --> torch conversion..."
fi

# Run core --> HF
echo "Running core --> HF conversion..."
python $MEGATRON_PATH/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader  $LOADER \
    --saver $SAVER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --hf-tokenizer mistralai/Mistral-Nemo-Instruct-2407 \
    --test-logits


cp $0 $SAVE_DIR/
echo "This conversion script has been copied to $SAVE_DIR for reproducibility"
echo "----------------------------------------------"