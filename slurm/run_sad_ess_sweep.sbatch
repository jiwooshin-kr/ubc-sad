#!/bin/bash
# MDLM ESS-vs-guidance-window sweep (SAD side of the Duo window sweep).
#
# Measures ESS only -- generation runs, scoring does NOT. The number we want is
# `ess_weights` in diagnostics/dist_logs/<tag>/ess_probe.csv, which the repellency
# kernel writes every guided step.
#
#   sbatch slurm/run_sad_ess_sweep.sbatch <T_START> <T_END> [REFSET] [NPROMPT]
#
#     T_START/T_END  guidance window, same convention as safety.t_start/t_end
#     REFSET         rtp | bt          (default rtp -- matches the Duo curve)
#     NPROMPT        prompts to run    (default 64; ESS is a per-step diagnostic,
#                                       so 64 prompts already give thousands of rows)
#
# Two env flags are REQUIRED for the CSV to appear and are easy to forget:
#   SAFE_REPELLENCY_DEBUG=1   -> safe_denoiser.py:1987 gates _log_metrics_to_csv on it
#   SAFE_REPELLENCY_CSV_LOG   -> safe_denoiser.py:90 reads csv_log_path from it
# Without both, the run completes fine and silently produces no ESS data.
#
#SBATCH --job-name=sad_ess
#SBATCH --account=rrg-mijungp
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=48G
#SBATCH --mail-user=natu33@kaist.ac.kr
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/sad_runs/%x_%j.out
#SBATCH --error=/scratch/%u/logs/sad_runs/%x_%j.err

set -euo pipefail

T_START="${1:?usage: sbatch run_sad_ess_sweep.sbatch T_START T_END [rtp|bt] [NPROMPT]}"
T_END="${2:?usage: sbatch run_sad_ess_sweep.sbatch T_START T_END [rtp|bt] [NPROMPT]}"
REFSET="${3:-rtp}"
NPROMPT="${4:-64}"

case "$REFSET" in
  rtp) ARTIFACT=real-toxicity-prompts-1500 ;;
  bt)  ARTIFACT=beavertails-1500 ;;
  *)   echo "REFSET must be rtp or bt, got '$REFSET'" >&2; exit 2 ;;
esac

SLUG="mdlm_ess_${REFSET}_t${T_START}_t${T_END}"

# ==========================================================================
# 1. environment  (a batch shell does not read ~/.bashrc)
# ==========================================================================
module purge
module load StdEnv/2023 python/3.12 arrow/21.0.0 gcc/12.3 cuda/12.2
source "$HOME/venvs/sad_env/bin/activate"

cd "${SAD_REPO:-$HOME/uniform/sad}"
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
export UNSAFE_ARTIFACT_ROOT="$SCRATCH/sad/artifacts/unsafe"

export SAFE_KERNEL_MODE=strict       # Table 1 setting; relaxed changes beta_hat's scale
export SAFE_REPELLENCY_DEBUG=1       # <- without this there is no ess_probe.csv
export SAFE_REPELLENCY_CSV_LOG=ess_probe.csv

MDLM="$SCRATCH/hf_models/mdlm-owt"
GPT2="$SCRATCH/hf_models/gpt2-large"
BASE_DIR="$SCRATCH/sad_runs"
RUN_DIR="$BASE_DIR/$SLUG/$SLURM_JOB_ID"

echo "############ slug=$SLUG job=$SLURM_JOB_ID"
echo "############ window=[$T_START,$T_END] refset=$REFSET($ARTIFACT) nprompt=$NPROMPT"
echo "############ generate  $(date)"

# ==========================================================================
# 2. generate  (no scoring -- ESS comes out of the guidance kernel itself)
# ==========================================================================
python -m tools.generate \
  --config-name experiments/realtoxicity_prompts \
  model.family=mdlm \
  model.model_name=mdlm-0p5b \
  model.checkpoint="$MDLM" \
  model.tokenizer_name="$GPT2" \
  data.prompt_variant=unsafe \
  data_catalog.sets.realtoxicity.prompt_source.params.prompt_score_min=0.7 \
  data_catalog.sets.realtoxicity.prompt_source.params.prompt_variant=unsafe \
  data_catalog.sets.realtoxicity.prompt_limit="$NPROMPT" \
  gen.max_new_tokens=1024 \
  gen.sampling_steps=1024 \
  gen.batch_size=8 \
  io.auto_batch=false \
  io.base_dir="$BASE_DIR" \
  io.experiment_slug="$SLUG" \
  safety.enabled=true \
  safety.eta=1.0 \
  safety.t_start="$T_START" \
  safety.t_end="$T_END" \
  safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe" \
  safety.unsafe_artifact_name="$ARTIFACT" \
  safety.auto_build_unsafe_artifacts=false

# ==========================================================================
# 3. report the ESS straight away so the .out file is self-contained
# ==========================================================================
echo "############ done  $(date)"
echo "run_dir: $RUN_DIR"

python - "$RUN_DIR" "$T_START" "$T_END" "$REFSET" <<'PY'
import csv, statistics, sys
from pathlib import Path

run_dir, t_start, t_end, refset = sys.argv[1:5]
hits = sorted(Path(run_dir).rglob("ess_probe.csv"))
if not hits:
    print(f"NO ess_probe.csv under {run_dir} -- check SAFE_REPELLENCY_DEBUG/CSV_LOG")
    raise SystemExit(1)

for p in hits:
    rows = list(csv.DictReader(open(p)))
    ess = [float(r["ess_weights"]) for r in rows
           if r.get("ess_weights") not in (None, "")]
    if not ess:
        print(f"{p}: no ess_weights column")
        continue
    kernel = "relaxed" if "beta_kernel_mode_paper" in rows[0] else "strict"
    print(f"ESS_RESULT refset={refset} window=t{t_start}-{t_end} "
          f"rows={len(ess)} ESS={statistics.mean(ess):.2f} "
          f"ESS_over_N={statistics.mean(ess)/1500:.5f} "
          f"min={min(ess):.2f} max={max(ess):.2f} kernel={kernel}")
    print(f"  path: {p}")
PY
