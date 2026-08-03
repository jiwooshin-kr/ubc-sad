#!/bin/bash
# SAD Table 1 reproduction (MDLM, RTP prompts).
#
#   sbatch ~/uniform/run_sad.sbatch
#   squeue -u $USER
#   tail -f sad-<JOBID>.out
#
# Sections 1-5 are one-time login-node setup and stay commented out.
# Edit the KNOBS block below to switch between Base / +SAD.
#
# NOTE: #SBATCH lines are read before the shell starts, so no $VAR expansion here.
#SBATCH --job-name=sad
#SBATCH --account=rrg-mijungp
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=48G
#SBATCH --mail-user=natu33@kaist.ac.kr
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/sad_runs/%x_%j.out
#SBATCH --error=/scratch/%u/logs/sad_runs/%x_%j.err

set -euo pipefail

# ==========================================================================
# KNOBS -- edit these
# ==========================================================================
SLUG=mdlm_base_rtp_unsafe    # output folder name; change per experiment

# Base row -> leave SAFETY as-is.
SAFETY=(safety.enabled=false)

# +SAD rows -> comment out the line above and uncomment ONE block below.
# Table 1 (MDLM): N = 1000, eta = 1.0, C in [1000, 750]
#
# SLUG=mdlm_sad_rtp
# SAFETY=(safety.enabled=true safety.eta=1.0
#         safety.t_start=750 safety.t_end=1000
#         safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
#         safety.unsafe_artifact_name=real-toxicity-prompts-1500
#         safety.auto_build_unsafe_artifacts=false)
#
# SLUG=mdlm_sad_beavertails
# SAFETY=(safety.enabled=true safety.eta=1.0
#         safety.t_start=750 safety.t_end=1000
#         safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
#         safety.unsafe_artifact_name=beavertails-1500
#         safety.auto_build_unsafe_artifacts=false)

# ==========================================================================
# 1. environment
#    A batch shell does NOT read ~/.bashrc, so everything is set here.
#    Order matters: module purge -> module load -> activate.
# ==========================================================================
module purge
module load StdEnv/2023 python/3.12 arrow/21.0.0 gcc/12.3 cuda/12.2
source "$HOME/venvs/sad_env/bin/activate"

cd "$HOME/uniform/sad"
export REPO_ROOT="$PWD"
# Keep the trailing :$PYTHONPATH -- dropping it wipes the CVMFS entry that the
# arrow module provides, and pyarrow disappears.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/src/mdlm:$REPO_ROOT/src/third_party/mdlm:${PYTHONPATH:-}"

export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_OFFLINE=1              # compute nodes have no internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export REALTOXICITY_DATA_DIR="$SCRATCH/hf_datasets/RealToxicityPrompts"
export BEAVERTAILS_DATA_DIR="$SCRATCH/hf_datasets/BeaverTails"
export TOXIGEN_DATA_DIR="$SCRATCH/hf_datasets/ToxiGen"
export UNSAFE_ARTIFACT_ROOT="$SCRATCH/sad/artifacts/unsafe"

MDLM="$SCRATCH/hf_models/mdlm-owt"
GPT2="$SCRATCH/hf_models/gpt2-large"
GUARD="$SCRATCH/hf_models/Llama-Guard-3-8B"
BASE_DIR="$SCRATCH/sad_runs"
RUN_DIR="$BASE_DIR/$SLUG/$SLURM_JOB_ID"    # io.run_id defaults to $SLURM_JOB_ID


# ==========================================================================
# 2. generate
# ==========================================================================
echo "############ slug=$SLUG job=$SLURM_JOB_ID"
echo "############ safety: ${SAFETY[*]}"
echo "############ generate  $(date)"

python -m tools.generate \
  --config-name experiments/realtoxicity_prompts \
  model.family=mdlm \
  model.model_name=mdlm-0p5b \
  model.checkpoint="$MDLM" \
  model.tokenizer_name="$GPT2" \
  data.prompt_variant=unsafe \
  data_catalog.sets.realtoxicity.prompt_source.params.prompt_score_min=0.7 \
  data_catalog.sets.realtoxicity.prompt_source.params.prompt_variant=unsafe \
  gen.max_new_tokens=1024 \
  gen.sampling_steps=1024 \
  gen.batch_size=32 \
  io.auto_batch=false \
  io.base_dir="$BASE_DIR" \
  io.experiment_slug="$SLUG" \
  "${SAFETY[@]}"

# ==========================================================================
# 3. score  (ASR via Llama-Guard-3-8B)
#    perplexity/BERTScore/MAUVE are off: not part of Table 1's ASR column.
#    To enable perplexity, pass the local PATH as the model name --
#    score.perplexity_model_name=$GPT2 -- because a bare "gpt2-large" is
#    looked up on the hub and compute nodes are offline.
# ==========================================================================
echo "############ score  $(date)"

python -m tools.score \
  model.family=mdlm \
  model.model_name=mdlm-0p5b \
  model.checkpoint="$MDLM" \
  model.tokenizer_name="$GPT2" \
  score.run_dir="$RUN_DIR" \
  score.track=safety \
  score.classifier=llamaguard \
  score.classifier_model="$GUARD" \
  score.model=mdlm-0p5b \
  score.batch_size=8 \
  score.max_new_tokens=32 \
  score.compute_perplexity=false \
  score.compute_bertscore=false \
  score.compute_mauve=false

# ==========================================================================
# 4. results
# ==========================================================================
echo "############ done  $(date)"
echo "run_dir: $RUN_DIR"
echo
cat "$RUN_DIR/scores_summary.csv"
echo
cat "$RUN_DIR/hazard_summary.csv"
