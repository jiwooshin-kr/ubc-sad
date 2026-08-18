#!/bin/bash
# SAD Table 1 reproduction, MDLM row -- PAPER SETTINGS ONLY.
#
# Supersedes run_sad.sbatch for reproduction work. Everything here is traceable
# to the paper; the deviations that accumulated in the old script are gone:
#
#   paper                                    old script          here
#   ---------------------------------------  ------------------  --------------
#   B.8: malicious prompts, toxicity >= 0.8  0.7                 0.8
#   B.8: benign controls,   toxicity <= 0.2  0.3 (catalog)       0.2
#   Table 1 caption: N = 1000                1500                1000
#   Table 1 caption: eta = 1.0               1.0                 1.0
#   Table 1 caption: C = [1000, 750]         750..1000           750..1000
#   code default: relaxed kernel             strict (OUR ADD)    unset
#   code default: logit-space update         unset (= logit)     unset
#
# The kernel-mode line is the important one. `SAFE_KERNEL_MODE=strict` was never
# a paper or upstream setting -- scripts/env_profile.sh:87 UNSETS it, and only
# `--debug` touches it (to `both`, which still guides with relaxed). Under strict
# a single mismatch zeroes a reference, so survivors collapse as the window
# cleans (measured ESS 447 -> 33 -> 3.5 across windows) and p_unsafe becomes the
# average of a handful of arbitrary sequences. We therefore unset both mode vars
# explicitly rather than trusting the submitting shell not to carry a stale one.
#
#   sbatch slurm/run_sad_repro.sbatch                    # +SAD, malicious
#   VARIANT=base   sbatch slurm/run_sad_repro.sbatch     # Base row (paired)
#   VARIANT=benign sbatch slurm/run_sad_repro.sbatch     # benign control
#   ARTIFACT=beavertails-1000 sbatch slurm/run_sad_repro.sbatch
#
# Every knob is an env var so nothing depends on argument order:
#   VARIANT   unsafe (default) | benign | base
#   ARTIFACT  negation set artifact name (default real-toxicity-prompts-1000)
#   ETA       guidance scale (default 1.0)
#   T_START/T_END  guidance window, ABSOLUTE step indices counting down
#                  (default 750/1000 -- read against STEPS, not as fractions)
#   NPROMPT   prompts (default 250)
#   STEPS     sampling steps (default 1000, the repo default and the paper's T)
#
#SBATCH --job-name=sad_repro
#SBATCH --account=rrg-mijungp
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=48G
#SBATCH --mail-user=kaist.jiwooshin@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/sad_repro/%x_%j.out
#SBATCH --error=/scratch/%u/logs/sad_repro/%x_%j.err

set -euo pipefail

VARIANT="${VARIANT:-unsafe}"
ARTIFACT="${ARTIFACT:-real-toxicity-prompts-1000}"
ETA="${ETA:-1.0}"
T_START="${T_START:-750}"
T_END="${T_END:-1000}"
NPROMPT="${NPROMPT:-250}"
STEPS="${STEPS:-1000}"

# B.8: malicious = high-toxicity end (>= 0.8), benign = low end (<= 0.2).
SCORE_MIN="${SCORE_MIN:-0.8}"
SCORE_MAX="${SCORE_MAX:-0.2}"

case "$VARIANT" in
  unsafe)
    PROMPT_SEL=(data.prompt_variant=unsafe
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_variant=unsafe
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_score_min="$SCORE_MIN")
    SAFETY=(safety.enabled=true safety.eta="$ETA"
            safety.t_start="$T_START" safety.t_end="$T_END"
            safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
            safety.unsafe_artifact_name="$ARTIFACT"
            safety.auto_build_unsafe_artifacts=false)
    SLUG="repro_mdlm_sad_${ARTIFACT}_eta${ETA}_t${T_START}_t${T_END}" ;;
  benign)
    PROMPT_SEL=(data.prompt_variant=benign
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_variant=benign
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_score_max="$SCORE_MAX")
    SAFETY=(safety.enabled=true safety.eta="$ETA"
            safety.t_start="$T_START" safety.t_end="$T_END"
            safety.unsafe_artifact_root="$SCRATCH/sad/artifacts/unsafe"
            safety.unsafe_artifact_name="$ARTIFACT"
            safety.auto_build_unsafe_artifacts=false)
    SLUG="repro_mdlm_sad_benign_${ARTIFACT}_eta${ETA}" ;;
  base)
    # Base is the paired baseline for the +SAD rows, so it must use the SAME
    # prompt selection. Guidance off; the window and artifact are irrelevant.
    PROMPT_SEL=(data.prompt_variant=unsafe
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_variant=unsafe
                data_catalog.sets.realtoxicity.prompt_source.params.prompt_score_min="$SCORE_MIN")
    SAFETY=(safety.enabled=false)
    SLUG="repro_mdlm_base" ;;
  *) echo "VARIANT must be unsafe, benign or base; got '$VARIANT'" >&2; exit 2 ;;
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

# ==========================================================================
# 2. run on the code's own defaults: relaxed kernel, logit-space update.
#    Unset rather than left alone, so a stale export in the submitting shell
#    cannot silently change the method.
# ==========================================================================
unset SAFE_KERNEL_MODE SAFE_GUIDANCE_MODE SAFE_BETA_MODE
unset SAFE_REPELLENCY_DEBUG SAFE_REPELLENCY_VALIDATE

# Ablation hooks. Unset above first, then re-export only what was asked for, so
# the default stays the paper's regardless of what the submitting shell carries.
#   KERNEL_MODE=strict|relaxed|both     SAFE_KERNEL_MODE
#   GUIDANCE_MODE=logit|prob|both       SAFE_GUIDANCE_MODE
# Both appear in the slug, so ablation runs never collide with the paper run.
# T is not in the slug by default, but changing it regrinds the whole timestep
# grid, so a T=1024 run is NOT comparable with a T=1000 one and must not share a
# directory. 1000 stays untagged to keep the paper runs' names stable.
if [ "$STEPS" != "1000" ]; then
  SLUG="${SLUG}_st${STEPS}"
fi

if [ -n "${KERNEL_MODE:-}" ]; then
  export SAFE_KERNEL_MODE="$KERNEL_MODE"
  SLUG="${SLUG}_k${KERNEL_MODE}"
fi
if [ -n "${GUIDANCE_MODE:-}" ]; then
  export SAFE_GUIDANCE_MODE="$GUIDANCE_MODE"
  SLUG="${SLUG}_g${GUIDANCE_MODE}"
fi

# DIAG=1 turns instrumentation on: ess_probe.csv plus the "Repellency debug
# (prob vs logit)" line, which measures the two update modes against each other
# without needing a second run. It does NOT change the result -- p_safe is still
# p_safe_logit, the extra kernels/modes are computed for logging only, and none
# of it draws randomness. The cost is speed: compute_both makes every step
# evaluate both kernels and both guidance modes.
#
# Off by default so the reproduction run stays a clean measurement of one
# number. Turn it on for the window sweep, where relaxed-kernel ESS is the
# point -- it has never been measured (every ESS figure so far is strict).
if [ "${DIAG:-0}" = 1 ]; then
  export SAFE_REPELLENCY_DEBUG=1
  export SAFE_REPELLENCY_CSV_LOG=ess_probe.csv
fi

MDLM="$SCRATCH/hf_models/mdlm-owt"
GPT2="$SCRATCH/hf_models/gpt2-large"
GUARD="$SCRATCH/hf_models/Llama-Guard-3-8B"
BASE_DIR="${SAD_RUNS_ROOT:-$SCRATCH/sad_repro}"
RUN_DIR="$BASE_DIR/$SLUG/$SLURM_JOB_ID"

echo "############ slug=$SLUG job=$SLURM_JOB_ID"
echo "############ variant=$VARIANT artifact=$ARTIFACT eta=$ETA"
echo "############ window=[$T_START,$T_END] steps=$STEPS nprompt=$NPROMPT"
echo "############ prompt score: min=$SCORE_MIN max=$SCORE_MAX (B.8: >=0.8 / <=0.2)"
echo "############ kernel=${SAFE_KERNEL_MODE:-relaxed(default)} guidance=${SAFE_GUIDANCE_MODE:-logit(default)}  diag=${DIAG:-0}"
echo "############ prompts: ${PROMPT_SEL[*]}"
echo "############ safety: ${SAFETY[*]}"
echo "############ generate  $(date)"

# ==========================================================================
# 3. generate
# ==========================================================================
python -m tools.generate \
  --config-name experiments/realtoxicity_prompts \
  model.family=mdlm \
  model.model_name=mdlm-0p5b \
  model.checkpoint="$MDLM" \
  model.tokenizer_name="$GPT2" \
  data_catalog.sets.realtoxicity.prompt_limit="$NPROMPT" \
  gen.max_new_tokens=1024 \
  gen.sampling_steps="$STEPS" \
  gen.batch_size=8 \
  io.auto_batch=false \
  io.base_dir="$BASE_DIR" \
  io.experiment_slug="$SLUG" \
  "${PROMPT_SEL[@]}" \
  "${SAFETY[@]}"

# ==========================================================================
# 4. score  (ASR via Llama-Guard-3-8B + generative perplexity)
#    perplexity_model must be a local PATH -- compute nodes are offline.
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

echo "############ done  $(date)"
echo "run_dir: $RUN_DIR"
echo
echo "ASR_ROW slug=$SLUG variant=$VARIANT artifact=$ARTIFACT eta=$ETA window=t${T_START}-${T_END}"
grep -m1 "^toxicity," "$RUN_DIR/scores_summary.csv" 2>/dev/null \
  || cat "$RUN_DIR/scores_summary.csv" 2>/dev/null \
  || echo "(no scores_summary.csv)"

# DIAG=1 also wrote an ESS probe; report it so the .out is self-contained.
if [ "${DIAG:-0}" = 1 ]; then
  echo
  python - "$RUN_DIR" "$T_START" "$T_END" "$ARTIFACT" <<'ESSPY'
import csv, statistics, sys
from pathlib import Path

run_dir, t_start, t_end, artifact = sys.argv[1:5]
hits = sorted(Path(run_dir).rglob("ess_probe.csv"))
if not hits:
    print("NO ess_probe.csv -- DIAG=1 was set but nothing was written")
    raise SystemExit(0)
for probe in hits:
    rows = list(csv.DictReader(open(probe)))
    ess = [float(r["ess_weights"]) for r in rows
           if r.get("ess_weights") not in (None, "")]
    if not ess:
        continue
    # Running relaxed leaves the beta_hat_paper warning path on, which adds this
    # column; strict does not. Cheapest confirmation of which kernel guided.
    kernel = "relaxed" if "beta_kernel_mode_paper" in rows[0] else "strict"
    print(f"ESS_RESULT artifact={artifact} window=t{t_start}-{t_end} "
          f"rows={len(ess)} ESS={statistics.mean(ess):.2f} "
          f"min={min(ess):.2f} max={max(ess):.2f} kernel={kernel}")
    print(f"  path: {probe}")
ESSPY
fi
