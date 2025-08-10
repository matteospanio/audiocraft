#!/bin/bash
#SBATCH --job-name musicgen-small
#SBATCH --output log/out/%j.txt
#SBATCH --error log/err/%j.txt
#SBATCH --mail-type ALL
#SBATCH --time 10-00:00:00
#SBATCH --partition allgroups
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:a40:2
#SBATCH --mem 32G

source .env

spack load ffmpeg

export DORA_PACKAGE="audiocraft"
export AUDIOCRAFT_DORA_DIR="$OUTPATH/dora"
export AUDIOCRAFT_CACHE_DIR="$OUTPATH/cache"
export AUDIOCRAFT_REFERENCE_DIR="$OUTPATH/reference"

srun --mail-user "$EMAIL" ~/miniconda3/bin/conda run -n musicgen \
	dora run -d \
		solver=musicgen/musicgen_base_32khz \
		model/lm/model_scale=small \
		continue_from=//pretrained/facebook/musicgen-small \
		conditioner=text2music \
		dset=audio/tasty_music \
		logging.log_wandb=true \
		wandb.project=musicgen-small \
		dataset.num_workers=8 \
		dataset.batch_size=32 \
		schedule.cosine.warmup=8 \
		generate.lm.prompted_samples=false \
		generate.lm.gen_gt_samples=true \
		autocast=false \
		fsdp.use=true \
		schedule.cosine.warmup=8 \
		checkpoint.save_every=2 \
		checkpoint.keep_last=5 \
		optim.lr=1e-4 \
		optim.epochs=100 \
		optim.updates_per_epoch=2000 \
		optim.optimizer=adamw
