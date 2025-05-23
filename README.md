# FOG

This repository includes the following implementations:
 - FOG architectures (postnorm, layerscale, xielu...) in both `Megatron-LM` and `fog_transformers`.
 - Other architectures used in some experiments (Llama3 w/ and w/o Smooth SwiGLU, OP...)
 - Metric tracking.
 - Evaluation pipeline.
 
This README includes step-by-step instructions to reproduce the main results presented.

## Infrastructure

The infrastructure is assumed to be slurm-based with FP8-compatible NVIDIA GPUs (e.g. H100s).
Across the presented scripts, we will make use of the following environmental variables, which you should adapt to your particular setup (e.g. paths you have write access to, etc):
- `FOG_PATH`: Set this to the path of this repository once cloned.
- `CONTAINER_PATH`: The `.toml` environment path for your container.
- `RAW_DATA_PATH`: Path where the raw data will be stored (jsonl format).
- `TOKENIZED_DATA_PATH`: Path path where the raw data will be stored.
- `TRAIN_ROOT`: Directory where all the checkpoints and logs of your main runs will be at.
- `DATA_CACHE_PATH`: Path that Megatron-LM can use as data cache.


Initial steps:

1. **Prepare a suitable container**.
   Our container is based on the [ngc/py-25-01](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html) pytorch container.
   Additionally, we add the following dependencies:
   ```
   python -m pip install datasets tqdm wandb git+https://github.com/nickjbrowning/XIELU.git
   cd $FOG_PATH/fog_transformers
   python -m pip install -e .  # install transformers library with FOG implementation.
   ```

1. The tokenizer used is the [Mistral-Nemo](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) tokenizer.
   You should have access to its huggingface repository, or change the `TOKENIZER` references in the scripts.

## Getting the data

The following slurm submission script downloads and tokenizes FineWeb-Edu data.
Make sure to change `--name` to `sample-350BT` or `CC-MAIN-2024-10` if you want to train on a larger amount of tokens.

```
#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --time=01:30:00
#SBATCH --job-name=download
#SBATCH --output=slurmlogs/%j.out
#SBATCH --error=slurmlogs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug

TOKENIZER=mistralai/Mistral-Nemo-Base-2407
MEGATRON_PATH=$FOG_PATH/Megatron-LM
cd $MEGATRON_PATH
mkdir -p $TOKENIZED_DATA_PATH

# First download the dataset to .jsonl format.
python scripts/download.py HuggingFaceFW/fineweb-edu $RAW_DATA_PATH --name=sample-10BT

# Now we can tokenize.
python tools/preprocess_data.py \
	--input=$RAW_DATA_PATH \
	--output-prefix=$TOKENIZED_DATA_PATH \
	--tokenizer-type=HuggingFaceTokenizer \
	--tokenizer-model=$TOKENIZER \
	--workers=64
```

## Launching

Using `scripts/submit.sh` you can launch the training runs. <br>
For instance, the main 1.5B experiments can be launched using the following commands:

```
# Llama3 1.5B bf16 baseline.
bash scripts/submit.sh 1

# Llama3 1.5B fp8dpa run.
bash scripts/submit.sh 1 --fp8 --fp8-dpa

# Llama3 1.5B with SmoothSwiGLU activation fp8dpa run.
bash scripts/submit.sh 1 --activation sswiglu --fp8 --fp8-dpa

# FOG-strong 1.5B fp8dpa.
bash scripts/submit.sh 1 \
	--fp8 --fp8-dpa --dtype-grad bf16 --dtype-param fp16 --dtype-m1 fp16 --dtype-m2 fp16 \
	--no-prenorm --postnorm --qknorm --softmax-scale 0.125 --layerscale --input-upscale --activation xielu --no-train-qk-gains
```

You should also export your `WANDB_API_KEY` to upload to w&b.
The checkpoints will be saved under `TRAIN_ROOT`, each experiment in a different directory.
Run `scripts/submit.sh --help` to get more information on other optional arguments you can specify.

## Evaluating

To evaluate a trained (Megatron) checkpoint, the easiest way is to convert it to the HF `transformers` format before using `lm-eval-harness`. <br>
First, clone [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository:
```
mkdir -p $FOG_PATH/eval/
cd $FOG_PATH/eval/
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```
Then place [default.yaml](./eval/default.yaml) in `FOG/eval/lm-evaluation-harness/lm_eval/tasks/fog_eval/`.

The following script will run the [Megatron -> HF] conversion with a logits' test, followed by evaluations of the checkpoints on the `fog_eval` set of benchmarks. <br>
`MODEL_NAME_LIST` should contain the names of runs saved by the above experiment launcher.
```
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --cpus-per-task=288
#SBATCH --job-name=evals
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=your/preferred/path/to/output/file.out
#SBATCH --error=your/preferred/path/to/error/file.err
#SBATCH --environment=/path/to/ngc_pt_jan.toml
#SBATCH --partition=debug
#SBATCH --time=01:00:00

MODEL_NAME_LIST=(
	"llama1.5B"
	"llama1.5B-fp8-fp8dpa"
	"llama1.5B-fp8-fp8dpa-sswiglu"
	"llama1.5B-fp8-fp8dpa-fp16OPT-nopre-postln-qknorm-softmax0.125-ls-is-xielu-ntQKgain"
)

i=0
for MODEL_NAME in "${MODEL_NAME_LIST[@]}"; do
    echo "Converting checkpoint $MODEL_NAME"
    bash $MEGATRON_PATH/tools/checkpoint/do-convert.sh $MODEL_NAME $i

    echo "Evaluating model $MODEL_NAME"
    bash $TRAIN_ROOT/eval/do-eval.sh $MODEL_NAME $i
    i=$((i + 1))

    echo "Finished evaluating checkpoint $MODEL_NAME"
    echo "Next...?"
    echo "----------------------------------------------"
done

echo "All checkpoints evaluated!"
```

