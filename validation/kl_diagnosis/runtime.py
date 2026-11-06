"""Capture generation/eval/train logprobs on identical actual training tokens."""

import asyncio
import json
import os
from pathlib import Path

import ray
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
from nemo_rl.algorithms.single_controller import SingleControllerActor


class DiagnosticLoss(ClippedPGLossFn):
    def __call__(self, *args, **kwargs):
        import nemo_rl.algorithms.loss.loss_functions as implementation

        original = implementation.clipped_pg_actor_objective
        data = kwargs.get("data", args[1] if len(args) > 1 else None)

        def capture(current, **objective):
            root = Path(os.environ["SC_LP_DIAG_DIR"]) / "train"
            root.mkdir(exist_ok=True)
            rank = torch.distributed.get_rank()
            index = len(list(root.glob(f"{self.diag_step}-{rank}-*.pt")))
            torch.save(
                {
                    "current": current.detach().cpu(),
                    "prev": objective["prev_logprobs"].detach().cpu(),
                    "generation": data["generation_logprobs"][:, 1:].detach().cpu(),
                    "mask": objective["mask"].detach().cpu(),
                    "input_ids": data["input_ids"].detach().cpu(),
                },
                root / f"{self.diag_step}-{rank}-{index}.pt",
            )
            return original(current, **objective)

        implementation.clipped_pg_actor_objective = capture
        try:
            return super().__call__(*args, **kwargs)
        finally:
            implementation.clipped_pg_actor_objective = original


BASE = SingleControllerActor.__ray_metadata__.modified_class


@ray.remote(num_cpus=1, num_gpus=0)
class DiagnosticController(BASE):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ref = self._gen.worker_group.run_single_worker_single_data(
            "diagnostic_extension_status", worker_idx=0
        )
        probes = ray.get(ref)
        assert len(probes) == 8 and all(
            item["probe"] == "ProbeExtension" for item in probes
        ), probes
        (Path(os.environ["SC_LP_DIAG_DIR"]) / "engine-probes.json").write_text(
            json.dumps(probes, indent=2) + "\n"
        )
        loss = DiagnosticLoss.__new__(DiagnosticLoss)
        loss.__dict__.update(self._loss_fn.__dict__)
        self._loss_fn = loss
        begin = self._trainer.begin_train_step

        def tagged_begin(loss_fn):
            loss_fn.diag_step = self._train_steps
            return begin(loss_fn)

        self._trainer.begin_train_step = tagged_begin

    async def _advantage_stage(self, meta):
        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=[
                "input_ids",
                "input_lengths",
                "generation_logprobs",
                "token_mask",
                "prev_logprobs",
            ],
        )
        root = Path(os.environ["SC_LP_DIAG_DIR"])
        torch.save(
            {
                "ids": meta.sample_ids,
                "tags": meta.tags,
                "data": data,
                "trainer_version": self._trainer_version,
                "train_step": self._train_steps,
            },
            root / f"selected-{self._train_steps}.pt",
        )
        (root / f"selected-{self._train_steps}.json").write_text(
            json.dumps({"step": self._train_steps, "tags": meta.tags}, default=str)
        )
        if self._train_steps == 600:
            # All current-step LA0 rollouts have completed. Score them with
            # the same engine before/after a reversible FP32 SSM replacement.
            (root / "score-request").write_text(str(root / "selected-600.pt"))
            ref = self._gen.worker_group.run_single_worker_single_data(
                "reset_prefix_cache_async", worker_idx=0
            )
            await asyncio.to_thread(ray.get, ref)
        return await super()._advantage_stage(meta)
