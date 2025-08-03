#!/bin/bash
#SBATCH --job-name musicgen-large
#SBATCH --output log/out/%j.txt
#SBATCH --error log/err/%j.txt
#SBATCH --mail-type ALL
#SBATCH --time 30-00:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --partition allgroups
#SBATCH --mem 35G
#SBATCH --gres=gpu:a40:4

source .env

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=OFF # change to INFO for more verbose output
export PATH="$PATH:$HOME/ffmpeg:$HOME/.local/bin"

export DORA_PACKAGE="audiocraft"
export AUDIOCRAFT_DORA_DIR="$OUTPATH/dora"
export AUDIOCRAFT_REFERENCE_DIR="$OUTPATH/reference"

srun --mail-user "$EMAIL" ~/miniconda3/bin/conda run -n musicgen \
	dora run -d \
		solver=musicgen/musicgen_base_32khz \
		model/lm/model_scale=large \
		continue_from=//pretrained/facebook/musicgen-large \
		conditioner=text2music \
		dset=audio/tasty_music \
		logging.log_wandb=true \
		wandb.project=musicgen-large \
		dataset.num_workers=1 \
		dataset.valid.num_samples=1 \
		dataset.batch_size=4 \
		schedule.cosine.warmup=8 \
		generate.lm.prompted_samples=False \
		generate.lm.gen_gt_samples=True \
		autocast=false \
        deadlock.use=false \
		fsdp.use=true \
        checkpoint.save_every=2 \
        checkpoint.keep_last=5 \
		optim.lr=1e-4 \
		optim.epochs=50 \
		optim.updates_per_epoch=2000 \
		optim.optimizer=adamw
