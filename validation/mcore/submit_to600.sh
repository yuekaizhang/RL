#!/bin/bash
set -euo pipefail
[[ "$USER" == yuekaiz ]]
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
exec 9>validation/mcore/slurm/submit-to600.lock
/usr/bin/flock -n 9
python - <<'PY'
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

root = Path('validation/mcore')
assert not (root / 'slurm/submitted-to600.tsv').exists(), 'Continuation already submitted; inspect existing jobs.'
queue = subprocess.run(['squeue', '-u', 'yuekaiz', '-h', '-o', '%i|%j|%T'], capture_output=True, text=True, check=True)
assert not any('yuekaiz-sc4963-mcore' in line for line in queue.stdout.splitlines()), queue.stdout
assert json.loads((root / 'completion-check.json').read_text())['status'] == 'passed'

def load(path):
    cfg = yaml.safe_load(path.read_text())
    default = cfg.pop('defaults', None)
    base = load(path.parent / default) if default else {}
    def merge(a, b):
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(a.get(key), dict):
                merge(a[key], value)
            else:
                a[key] = copy.deepcopy(value)
    merge(base, cfg)
    return base

proof = {'status': 'passed', 'target_steps': 600, 'cases': {}}
for lag in (0, 1):
    source = root / f'results/yuekaiz-mcore-la{lag}-300/checkpoints/step_300'
    info = json.loads((source / 'training_info.json').read_text())
    assert info['current_step'] == info['trainer_version'] == 300
    assert info['consumed_samples'] == 2400
    for artifact in ('data_plane', 'replay_buffer_metadata.pt', 'train_dataloader.pt', 'policy/weights/iter_0000000/train_state.pt'):
        assert (source / artifact).exists(), (lag, artifact)
    shards = list((source / 'policy/weights/iter_0000000').glob('*.distcp'))
    assert len(shards) == 16 and all(p.stat().st_size > 0 for p in shards), (lag, shards)
    previous = load(root / f'la{lag}.yaml')
    current = load(root / f'la{lag}_to600.yaml')
    expected = copy.deepcopy(previous)
    expected['grpo']['max_num_steps'] = 600
    expected['policy']['megatron_cfg']['scheduler'].update(use_checkpoint_opt_param_scheduler=True, override_opt_param_scheduler=False)
    expected['checkpointing']['checkpoint_dir'] = current['checkpointing']['checkpoint_dir']
    expected['logger']['log_dir'] = current['logger']['log_dir']
    expected['logger']['wandb']['resume'] = 'must'
    assert current == expected, 'Unexpected training-recipe change'
    assert current['logger']['wandb']['id'] == f'yuekaiz4963mc{lag}s42'
    assert current['policy']['logprob_chunk_size'] == 512
    assert current['checkpointing']['load_replay_buffer'] is True
    assert current['checkpointing']['checkpoint_dir'] != previous['checkpointing']['checkpoint_dir']
    proof['cases'][str(lag)] = {
        'source_checkpoint': str(source.resolve()),
        'training_info_sha256': hashlib.sha256((source / 'training_info.json').read_bytes()).hexdigest(),
        'weight_shards': len(shards),
        'source_step': 300,
        'wandb_id': current['logger']['wandb']['id'],
        'destination': current['checkpointing']['checkpoint_dir'],
        'only_changes': ['target600', 'restore saved constant scheduler', 'new checkpoint/log root', 'W&B resume must'],
    }
(root / 'results/to600-submission-check.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
PY
if [[ "${1:-}" == --check ]]; then exit 0; fi
[[ -f /lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh ]]
for lag in 0 1; do
  export SC_MCORE_LOOKAHEAD="$lag"
  previous=''
  for segment in 1 2 3 4; do
    args=()
    if [[ -n "$previous" ]]; then args+=(--dependency="afterok:$previous"); fi
    job=$(sbatch --parsable "${args[@]}" \
      --job-name="yuekaiz-sc4963-mcore-ep16-la${lag}-600-s42-seg${segment}" \
      --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Continue MCore LA0/LA1 checkpoint300 to600, full optimizer/scheduler/native replay restore"}}' \
      --output="$PWD/validation/mcore/slurm/slurm-%j.out" \
      --error="$PWD/validation/mcore/slurm/slurm-%j.err" \
      validation/mcore/batch_to600.sh)
    printf '%s\t%s\t%s\n' "$lag" "$segment" "$job" | tee -a validation/mcore/slurm/submitted.tsv validation/mcore/slurm/submitted-to600.tsv
    previous="$job"
  done
done
