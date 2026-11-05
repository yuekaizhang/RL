#!/bin/bash
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
python - <<'PY'
from pathlib import Path
import json
assert json.loads(Path('validation/mcore/results/smoke-proof.json').read_text())['status'] == 'passed'
assert json.loads(Path('validation/mcore/config-equivalence.json').read_text())['status'] == 'passed'
assert json.loads(Path('validation/mcore/provider-probe.json').read_text())['status'] == 'passed'
assert not Path('validation/mcore/slurm/submitted.tsv').exists(), 'Jobs already submitted; inspect them instead of duplicating'
PY
mkdir -p validation/mcore/slurm
for lag in 0 1; do
  export SC_MCORE_LOOKAHEAD="$lag"
  previous=''
  for segment in 1 2 3 4; do
    args=()
    if [[ -n "$previous" ]]; then args+=(--dependency="afterok:$previous"); fi
    job=$(sbatch --parsable "${args[@]}" \
      --job-name="yuekaiz-sc4963-mcore-ep16-la${lag}-300-s42-seg${segment}" \
      --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"MCore SC LA0/LA1 KL comparison; CPUs/GPU roles alternate during rollout/refit and conversion"}}' \
      --output="$PWD/validation/mcore/slurm/slurm-%j.out" \
      --error="$PWD/validation/mcore/slurm/slurm-%j.err" \
      validation/mcore/batch.sh)
    printf '%s\t%s\t%s\n' "$lag" "$segment" "$job" | tee -a validation/mcore/slurm/submitted.tsv
    previous="$job"
  done
done
