"""Isolated real-worker audit harness; never imported by convergence runs."""

import contextvars
import json
import os
import threading
import time
import uuid
from pathlib import Path

import ray
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
from nemo_rl.algorithms.single_controller import SingleControllerActor

G_FIELDS = ["input_ids", "input_lengths", "generation_logprobs", "token_mask"]


class AuditedLoss(ClippedPGLossFn):
    """Capture the actual TIS tensor handed to the actor objective in a worker."""

    def __call__(self, *args, **kwargs):
        import nemo_rl.algorithms.loss.loss_functions as implementation

        original = implementation.clipped_pg_actor_objective
        data = kwargs.get("data", args[1] if len(args) > 1 else None)

        def capture(current, **objective):
            assert objective["importance_weights"] is not None
            prev = objective["prev_logprobs"]
            gen = torch.where(
                objective["mask"].bool(), data["generation_logprobs"][:, 1:], 0.0
            )
            expected = torch.exp(prev - gen).nan_to_num(0, 0, 0).clamp(0, 2)
            torch.testing.assert_close(
                objective["importance_weights"], expected, rtol=0, atol=0
            )
            path = Path(self.audit_dir) / "loss"
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "time_ns": time.monotonic_ns(),
                    "step": self.audit_step,
                    "session": self.audit_session,
                    "data": {
                        k: v.detach().cpu()
                        for k, v in data.items()
                        if isinstance(v, torch.Tensor)
                    },
                    "current": current.detach().cpu(),
                    "prev": prev.detach().cpu(),
                    "generation": gen.detach().cpu(),
                    "importance": objective["importance_weights"].detach().cpu(),
                    "mask": objective["mask"].detach().cpu(),
                },
                path / f"{os.getpid()}-{uuid.uuid4().hex}.pt",
            )
            return original(current, **objective)

        assert not self.enable_torch_compile
        implementation.clipped_pg_actor_objective = capture
        try:
            return super().__call__(*args, **kwargs)
        finally:
            implementation.clipped_pg_actor_objective = original


G_BASE = SingleControllerActor.__ray_metadata__.modified_class


@ray.remote(num_cpus=1, num_gpus=0)
class AuditedController(G_BASE):  # pragma: no cover
    """Trace stable group identities at the real controller boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_dir = Path(os.environ["SC_AUDIT_DIR"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_lock = threading.Lock()
        self.audit_targets = {}
        self.audit_active_rpcs = {}
        self.audit_session = uuid.uuid4().hex
        self.audit_group_context = contextvars.ContextVar("audit_group", default=None)
        loss = AuditedLoss.__new__(AuditedLoss)
        loss.__dict__.update(self._loss_fn.__dict__)
        loss.audit_dir = str(self.audit_dir)
        loss.audit_session = self.audit_session
        loss.audit_step = self._train_steps
        self._loss_fn = loss
        self.event("init", restored_step=self._train_steps)
        self._install_hooks()

    def event(self, event, **values):
        row = dict(
            event=event,
            session=self.audit_session,
            time_ns=time.monotonic_ns(),
            step=self._train_steps,
            version=self._trainer_version,
            **values,
        )
        with self.audit_lock:
            with (self.audit_dir / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, default=str) + "\n")

    def _install_hooks(self):
        buffer = self._buffer
        reserve = buffer.reserve

        def traced_reserve(**kwargs):
            group = reserve(**kwargs)
            self.audit_targets[group] = kwargs["target_step"]
            self.event(
                "dispatch",
                group=group,
                target=kwargs["target_step"],
                weight=kwargs["weight_version"],
                buffered=len(buffer.ready_list),
            )
            return group

        buffer.reserve = traced_reserve
        commit = buffer.commit

        async def traced_commit(group, *args, **kwargs):
            meta = await commit(group, *args, **kwargs)
            data = await self._call_dp(
                "get_samples",
                sample_ids=meta.sample_ids,
                partition_id=meta.partition_id,
                select_fields=G_FIELDS,
            )
            torch.save(
                {"ids": meta.sample_ids, "tags": meta.tags, "data": data},
                self.audit_dir / f"published-{group}.pt",
            )
            self.event(
                "ready",
                group=group,
                target=self.audit_targets.get(group),
                ids=meta.sample_ids,
                tags=meta.tags,
                start_weight=kwargs.get(
                    "start_weight_version", args[1] if len(args) > 1 else None
                ),
                end_weight=kwargs.get(
                    "end_weight_version", args[2] if len(args) > 2 else None
                ),
            )
            return meta

        buffer.commit = traced_commit
        manager = self._rollout_manager
        admitted = manager.mark_prompt_group_admitted

        def traced_admitted(cut, group, **kwargs):
            result = admitted(cut, group, **kwargs)
            record = next(
                g for g in manager.recovery_ledger.groups() if g.group_id == group
            )
            self.event(
                "admit",
                group=group,
                target=kwargs["target_step"],
                prompt=record.prompt_id,
            )
            return result

        manager.mark_prompt_group_admitted = traced_admitted
        generate = manager.generate_and_push

        async def traced_generate(*args, **kwargs):
            group = kwargs.get("lineage_group_id")
            token = self.audit_group_context.set(group)
            try:
                return await generate(*args, **kwargs)
            finally:
                self.audit_group_context.reset(token)

        manager.generate_and_push = traced_generate
        backend = manager._impl._policy_generation
        launch = backend.worker_group.run_single_worker_single_data

        def traced_launch(*args, **kwargs):
            result = launch(*args, **kwargs)
            if kwargs.get("method_name") != "generate_async":
                return result
            request = uuid.uuid4().hex
            group = self.audit_group_context.get()
            self.audit_active_rpcs[request] = group
            self.event("rpc_start", request=request, group=group)

            async def tracked():
                try:
                    async for ref in result:
                        self.audit_active_rpcs.pop(request, None)
                        self.event("rpc_output", request=request, group=group)
                        yield ref
                finally:
                    self.audit_active_rpcs.pop(request, None)
                    self.event("rpc_end", request=request, group=group)

            return tracked()

        backend.worker_group.run_single_worker_single_data = traced_launch
        trainer = self._trainer
        train = trainer.train_microbatches_from_meta

        def traced_train(meta, **kwargs):
            self.event("train_start", ids=meta.sample_ids)
            try:
                return train(meta, **kwargs)
            finally:
                self.event("train_end", ids=meta.sample_ids)

        trainer.train_microbatches_from_meta = traced_train
        begin = trainer.begin_train_step

        def traced_begin(loss):
            loss.audit_step = self._train_steps
            return begin(loss)

        trainer.begin_train_step = traced_begin

    async def _advantage_stage(self, meta):
        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=[*G_FIELDS, "prev_logprobs"],
        )
        torch.save(
            {"ids": meta.sample_ids, "tags": meta.tags, "data": data},
            self.audit_dir / f"selected-{self._train_steps}.pt",
        )
        self.event("select", ids=meta.sample_ids, tags=meta.tags)
        return await super()._advantage_stage(meta)

    async def _cleanup_consumed_metas_unlocked(self, cut, metas):
        await super()._cleanup_consumed_metas_unlocked(cut, metas)
        self.event("consume", ids=[sid for meta in metas for sid in meta.sample_ids])

    async def _sync_weights(self, *args, **kwargs):
        self.event("refit_start")
        result = await super()._sync_weights(*args, **kwargs)
        self.event("refit_end")
        return result

    async def _save_data_plane_checkpoint(self, *args, **kwargs):
        # Called while the exclusive checkpoint cut is held: these owners belong
        # to the same snapshot as the replay metadata and native TQ tensors.
        replay = kwargs.get("replay_metadata") or {}
        ready = [group["group_id"] for group in replay.get("groups", [])]
        pending = dict(self.audit_active_rpcs)
        self.event(
            "checkpoint_cut",
            ready=ready,
            pending_rpcs=pending,
            unfinished=self._rollout_manager.recovery_ledger.state_dict(),
            replay_versions=[
                {
                    "group": g["group_id"],
                    "start": g["start_weight"],
                    "end": g["end_weight"],
                    "target": g["target_step"],
                }
                for g in replay.get("groups", [])
            ],
        )
        return await super()._save_data_plane_checkpoint(*args, **kwargs)
