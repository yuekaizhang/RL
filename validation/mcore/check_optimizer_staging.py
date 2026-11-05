"""Measure Bridge CPU staging using the image's actual compressed TE Adam."""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from megatron.bridge.training.optim import (
    memory_efficient_precision_aware_optimizer_state_checkpointing,
)
from transformer_engine.pytorch.optimizers import FusedAdam

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
torch.manual_seed(42)
initial = [torch.randn(1048576, dtype=torch.bfloat16) for _ in range(16)]
reference = None
records = []
for enabled in (False, True):
    params = [torch.nn.Parameter(value.cuda()) for value in initial]
    optimizer = FusedAdam(
        params,
        lr=1e-6,
        weight_decay=0.1,
        master_weights=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
        store_param_remainders=True,
        use_decoupled_grad=True,
    )
    for param in params:
        param.decoupled_grad = torch.ones_like(param, dtype=torch.float32)
    optimizer.step()
    wrapper = SimpleNamespace(
        optimizer=optimizer,
        config=SimpleNamespace(
            use_precision_aware_optimizer=True, optimizer_cpu_offload=False
        ),
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with memory_efficient_precision_aware_optimizer_state_checkpointing(
        wrapper, enabled=enabled
    ) as patched:
        portable = optimizer.state_dict()
        optimizer.load_state_dict(portable)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    portable_cpu = {
        key: {name: value.cpu().clone() for name, value in state.items()}
        for key, state in portable["state"].items()
    }
    del portable
    optimizer.step()
    updated = [param.detach().cpu().clone() for param in params]
    if reference is None:
        reference = (portable_cpu, updated)
    else:
        for key, state in portable_cpu.items():
            for name, value in state.items():
                assert torch.equal(value, reference[0][key][name]), (key, name)
        assert all(torch.equal(a, b) for a, b in zip(updated, reference[1]))
    records.append(
        {"cpu_staging": enabled, "patched": patched, "extra_peak_bytes": peak}
    )
    del optimizer, params, wrapper, portable_cpu, updated, param
    gc.collect()
    torch.cuda.empty_cache()
assert records[1]["patched"] == 1
assert records[1]["extra_peak_bytes"] < records[0]["extra_peak_bytes"]
result = {
    "status": "passed",
    "scope": "TE FusedAdam portable state + load + next update; not full EP16 resume",
    "portable_state_and_next_update": "bitwise_equal",
    "checks": records,
}
Path("validation/mcore/results/optimizer-staging-check.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result), flush=True)
