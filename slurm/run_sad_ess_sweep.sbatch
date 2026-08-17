#!/bin/bash
# MDLM guidance-window sweep: ESS + ASR.
#
# The positive control for "does a critical window exist?". MDLM's +SAD effect is
# -5.8pt (38.4 -> 32.6 in Table 1), which is above the n=250 single-seed
# detection floor of ~5.7pt -- so a window contrast IS measurable here with one
# seed per cell. Duo's effect (<=2pt) is not, which is why MDLM carries the
# existence claim and Duo answers the "does uniform state have one" question.
#
#   sbatch slurm/run_sad_ess_sweep.sbatch <T_START> <T_END> [REFSET] [NPROMPT]
#
#     T_START/T_END  guidance window, same convention as safety.t_start/t_end.
#                    Indices count DOWN (step_idx = T - i) and are ABSOLUTE, so
#                    they must be read against T below, not as fractions.
#     REFSET         rtp | bt | base   (default rtp)
#                      base -> safety.enabled=false, window ignored: the paired
#                      baseline every +SAD row is compared against.
#     NPROMPT        prompts to run    (default 250, the Table 1 value)
#
# Two env flags are REQUIRED for the ESS CSV to appear and are easy to forget:
#   SAFE_REPELLENCY_DEBUG=1   -> safe_denoiser.py:1987 gates _log_metrics_to_csv on it
#   SAFE_REPELLENCY_CSV_LOG   -> safe_denoiser.py:90 reads csv_log_path from it
# Without both, the run completes fine and silently produces no ESS data.
#
#SBATCH --job-name=sad_win
#SBATCH --account=rrg-mijungp
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=48G
#SBATCH --mail-user=kaist.jiwooshin@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/sad_runs/%x_%j.out
#SBATCH --error=/scratch/%u/logs/sad_runs/%x_%j.err

set -euo pipefail

T_START="${1:?usage: sbatch run_sad_ess_sweep.sbatch T_START T_END [rtp|bt|base] [NPROMPT]}"
T_END="${2:?usage: sbatch run_sad_ess_sweep.sbatch T_START T_END [rtp|bt|base] [NPROMPT]}"
REFSET="${3:-rtp}"
NPROMPT="${4:-250}"

# T=1000 is SAD's own default (configs/config.yaml:27) and the paper's T. The
# earlier campaign used 1024, which shifted the window: with step_idx = T - i,
# t_end=1000 against 1024 steps only starts guiding 24 iterations in. At T=1000
# the paper's window really does begin at the first iteration (alpha_t = 0.001
# rather than 0.024).
STEPS="${STEPS:-1000}"

case "$REFSET" in
  rtp)  ARTIFACT=real-toxicity-prompts-1500
        SAFETY=(safety.enabled=true safety.eta=1.0
                safety.t_start="$T_START" safety.t_end="$T_END"
                safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
                safety.unsafe_artifact_name="$ARTIFACT"
                safety.auto_build_unsafe_artifacts=false)
        SLUG="mdlm_win_${REFSET}_t${T_START}_t${T_END}" ;;
  bt)   ARTIFACT=beavertails-1500
        SAFETY=(safety.enabled=true safety.eta=1.0
                safety.t_start="$T_START" safety.t_end="$T_END"
                safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
                safety.unsafe_artifact_name="$ARTIFACT"
                safety.auto_build_unsafe_artifacts=false)
        SLUG="mdlm_win_${REFSET}_t${T_START}_t${T_END}" ;;
  base) ARTIFACT="(none)"
        SAFETY=(safety.enabled=false)
        SLUG="mdlm_win_base" ;;
  *)    echo "REFSET must be rtp, bt or base; got '$REFSET'" >&2; exit 2 ;;
esac

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
GUARD="$SCRATCH/hf_models/Llama-Guard-3-8B"
# The T=1000 campaign gets its own root so it cannot be pooled with the T=1024
# runs, whose timestep grid is different and whose slugs do not record T.
BASE_DIR="${SAD_RUNS_ROOT:-$SCRATCH/sad_runs_t1000}"
RUN_DIR="$BASE_DIR/$SLUG/$SLURM_JOB_ID"

echo "############ slug=$SLUG job=$SLURM_JOB_ID"
echo "############ window=[$T_START,$T_END] refset=$REFSET($ARTIFACT) nprompt=$NPROMPT steps=$STEPS"
echo "############ safety: ${SAFETY[*]}"
echo "############ generate  $(date)"

# ==========================================================================
# 2. generate
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
  gen.sampling_steps="$STEPS" \
  gen.batch_size=8 \
  io.auto_batch=false \
  io.base_dir="$BASE_DIR" \
  io.experiment_slug="$SLUG" \
  "${SAFETY[@]}"

# ==========================================================================
# 3. score  (ASR via Llama-Guard-3-8B + generative perplexity)
#    perplexity_model must be a local PATH, not the name "gpt2-large" --
#    compute nodes are offline.
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
  score.compute_perplexity=true \
  score.perplexity_model="$GPT2" \
  score.compute_bertscore=false \
  score.compute_mauve=false

# ==========================================================================
# 4. results -- ASR then ESS, so the .out file is self-contained
# ==========================================================================
echo "############ done  $(date)"
echo "run_dir: $RUN_DIR"
echo
cat "$RUN_DIR/scores_summary.csv" 2>/dev/null || echo "(no scores_summary.csv)"
echo
cat "$RUN_DIR/hazard_summary.csv" 2>/dev/null || echo "(no hazard_summary.csv)"
echo

python - "$RUN_DIR" "$T_START" "$T_END" "$REFSET" <<'PY'
import csv, statistics, sys
from pathlib import Path

run_dir, t_start, t_end, refset = sys.argv[1:5]
if refset == "base":
    print("ESS_RESULT refset=base (guidance disabled, no weights to measure)")
    raise SystemExit(0)

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
