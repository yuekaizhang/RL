"""Check the OOM workaround preserves logprobs/gradients and measure workspace."""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_rl.distributed.model_utils import ChunkedDistributedLogprob

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
assert torch.cuda.is_available()
torch.manual_seed(42)
output = Path("validation/mcore/results/logprob-chunk-check.json")
result = {"status": "passed", "checks": []}
with tempfile.TemporaryDirectory(prefix="yuekaiz-chunk-") as directory:
    dist.init_process_group(
        "nccl", init_method=f"file://{directory}/rendezvous", rank=0, world_size=1
    )
    for length, vocab in [(1537, 4096), (4096, 248320)]:
        logits = torch.randn(
            1, length, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        target = torch.randint(vocab, (1, length), device="cuda")
        weights = torch.randn(1, length, device="cuda")
        records = []
        reference_lp = reference_grad = None
        for chunk in (4096, 512):
            logits.grad = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            lp = ChunkedDistributedLogprob.apply(
                logits, target, 0, vocab, chunk, dist.group.WORLD, False
            )
            (lp * weights).sum().backward()
            torch.cuda.synchronize()
            records.append(
                {
                    "chunk": chunk,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "workspace_above_input_bytes": torch.cuda.max_memory_allocated()
                    - baseline,
                }
            )
            actual_lp = lp.detach().cpu()
            actual_grad = logits.grad.detach().cpu()
            if reference_lp is None:
                reference_lp, reference_grad = actual_lp, actual_grad
            else:
                lp_error = float((actual_lp - reference_lp).abs().max())
                grad_error = float((actual_grad - reference_grad).abs().max())
                print(
                    json.dumps(
                        {
                            "length": length,
                            "vocab": vocab,
                            "lp_max_abs_error": lp_error,
                            "grad_max_abs_error": grad_error,
                        }
                    ),
                    flush=True,
                )
                torch.testing.assert_close(
                    actual_lp, reference_lp, rtol=1e-6, atol=2e-6
                )
                torch.testing.assert_close(
                    actual_grad, reference_grad, rtol=1e-2, atol=2e-6
                )
            del lp
        result["checks"].append(
            {
                "sequence_length": length,
                "vocab_size": vocab,
                "logprob_max_abs_error": lp_error,
                "gradient_max_abs_error": grad_error,
                "logprob_tolerance": {"rtol": 1e-6, "atol": 2e-6},
                "gradient_tolerance": {"rtol": 1e-2, "atol": 2e-6},
                "memory": records,
            }
        )
        del (
            logits,
            target,
            weights,
            actual_lp,
            actual_grad,
            reference_lp,
            reference_grad,
        )
        torch.cuda.empty_cache()
    dist.destroy_process_group()
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
