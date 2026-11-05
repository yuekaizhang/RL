#!/bin/bash
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
exec 9>validation/mcore/slurm/recovery-submit.lock
/usr/bin/flock -n 9
python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('validation/mcore/results/logprob-chunk-check.json').read_text())['status'] == 'passed'
config = json.loads(Path('validation/mcore/config-equivalence.json').read_text())
assert config['status'] == 'passed' and config['logprob_chunk_size'] == 512
assert json.loads(Path('validation/mcore/results/resume-preflight.json').read_text())['status'] == 'passed'
assert not Path('validation/mcore/slurm/submitted-chunk512.tsv').exists(), 'Recovery already submitted'
for lag in (0, 1):
    checkpoint = Path(f'validation/mcore/results/yuekaiz-mcore-la{lag}-300/checkpoints/step_10')
    info = json.loads((checkpoint/'training_info.json').read_text())
    assert info['current_step'] == info['trainer_version'] == 10
    for artifact in ['policy/weights', 'data_plane', 'train_dataloader.pt', 'replay_buffer_metadata.pt']:
        assert (checkpoint/artifact).exists(), (lag, artifact)
PY
for lag in 0 1; do
  export SC_MCORE_LOOKAHEAD="$lag"
  previous=''
  for segment in 1 2 3 4; do
    args=()
    if [[ -n "$previous" ]]; then args+=(--dependency="afterok:$previous"); fi
    job=$(sbatch --parsable "${args[@]}" \
      --job-name="yuekaiz-sc4963-mcore-la${lag}-300-chunk512-r1-s${segment}" \
      --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Resume MCore LA0/LA1 at checkpoint10 after reducing logprob backward workspace"}}' \
      --output="$PWD/validation/mcore/slurm/slurm-%j.out" \
      --error="$PWD/validation/mcore/slurm/slurm-%j.err" \
      validation/mcore/batch.sh)
    printf '%s\t%s\t%s\n' "$lag" "$segment" "$job" | tee -a validation/mcore/slurm/submitted.tsv validation/mcore/slurm/submitted-chunk512.tsv
    previous="$job"
  done
done
