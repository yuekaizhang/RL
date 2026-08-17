# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark draft co-training integration for the DTensor-v2 policy worker.

Bridges the vendored DSpark modules with nemo-rl's automodel training loop:
building and sharding the draft from a pretrained checkpoint, capturing the
policy's hidden states and raw logits during the training forward, computing
the DSpark loss on rollout data, and checkpointing the draft alongside the
policy with a validated metadata record.
"""

import contextlib
import json
import os
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.models.automodel.draft.common import DSparkForwardOutput
from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel
from nemo_rl.models.automodel.draft.loss import compute_dspark_loss

DSPARK_REQUIRED_CONFIG_FIELDS = ("block_size", "target_layer_ids", "mask_token_id")
DRAFT_CHECKPOINT_DIRNAME = "draft"
DSPARK_META_FILENAME = "dspark_meta.json"


class PolicyWithDraft(nn.Module):
    """Composite module pairing the policy and draft for optimizer state I/O.

    The Automodel checkpointer pairs one model with one optimizer for optimizer
    state save/load; the single optimizer here owns param groups from both the
    policy and the draft, so this composite is the module handed to the
    checkpointer's optimizer paths. Training, refit, and inference keep
    referencing the policy module directly.
    """

    def __init__(self, policy: nn.Module, draft: nn.Module):
        super().__init__()
        self.policy = policy
        self.draft = draft

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.policy(*args, **kwargs)


def validate_dspark_draft_config(
    draft_hf_config: Any, dspark_options: dict[str, Any]
) -> None:
    """Validate the draft checkpoint's config.json against the training options."""
    missing = [
        field
        for field in DSPARK_REQUIRED_CONFIG_FIELDS
        if getattr(draft_hf_config, field, None) is None
    ]
    if missing:
        raise ValueError(
            f"DSpark draft checkpoint config.json is missing required fields: {missing}. "
            "Expected a checkpoint produced by DSpark training (e.g. "
            "deepseek-ai/dspark_qwen3_8b_block7)."
        )
    architectures = getattr(draft_hf_config, "architectures", None) or []
    if "Qwen3DSparkModel" not in architectures:
        raise ValueError(
            "DSpark draft checkpoint must declare architectures=['Qwen3DSparkModel'], "
            f"got {architectures}. Only Qwen3-family DSpark drafts are supported."
        )
    confidence_alpha = float(dspark_options["confidence_loss_alpha"])
    if confidence_alpha > 0 and not bool(
        getattr(draft_hf_config, "enable_confidence_head", False)
    ):
        raise ValueError(
            "policy.draft.dspark.confidence_loss_alpha > 0 but the draft checkpoint "
            "has no confidence head (enable_confidence_head is false in config.json). "
            "Set confidence_loss_alpha: 0.0 or use a checkpoint with a confidence head."
        )
    l1_alpha = float(dspark_options["l1_loss_alpha"])
    if (
        l1_alpha < 0
        or confidence_alpha < 0
        or float(dspark_options["ce_loss_alpha"]) < 0
    ):
        raise ValueError("DSpark loss alphas must be non-negative.")


def build_dspark_draft_model(
    model_name: str,
    dspark_options: dict[str, Any],
    torch_dtype: torch.dtype,
    mesh: Any,
) -> Qwen3DSparkModel:
    """Build, validate, and FSDP2-shard the DSpark draft from a pretrained checkpoint."""
    from transformers import AutoConfig

    draft_hf_config = AutoConfig.from_pretrained(model_name)
    validate_dspark_draft_config(draft_hf_config, dspark_options)
    # num_anchors is a training-time knob, not checkpoint architecture; the
    # flex-attention implementation is required by the training block mask.
    draft_hf_config.num_anchors = int(dspark_options["num_anchors"])
    draft_hf_config._attn_implementation = "flex_attention"

    draft_model = Qwen3DSparkModel.from_pretrained(
        model_name,
        config=draft_hf_config,
        torch_dtype=torch_dtype,
    )
    draft_model = draft_model.to("cuda")

    draft_model.requires_grad_(True)
    train_embed_and_head = bool(dspark_options["train_embed_and_head"])
    draft_model.set_embedding_head_trainable(train_embed_and_head)

    from torch.distributed.fsdp import fully_shard

    for layer in draft_model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(draft_model, mesh=mesh)
    return draft_model


def _resolve_layers(
    policy_model: nn.Module,
) -> tuple[nn.Module, nn.ModuleList]:
    """Locate the decoder backbone (embedding owner) and its layer list."""
    base = getattr(policy_model, "model", policy_model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(
            "DSpark hidden capture could not locate `.layers` on the policy "
            f"model (searched {type(base).__name__}). Only HF-style decoder "
            "stacks are supported for DSpark co-training."
        )
    return base, layers


def _hook_output_tensor(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if isinstance(hidden, DTensor):
        # Under TP the layer output can be sharded or partial; materialize the
        # replicated full tensor (a no-op collective when already replicated).
        if any(not p.is_replicate() for p in hidden.placements):
            hidden = hidden.full_tensor()
        else:
            hidden = hidden.to_local()
    return hidden.detach()


class DSparkHiddenCapture:
    """Forward hooks capturing the policy's per-layer hiddens for the draft.

    Captured tensors are detached: the DSpark loss never backprops into the
    policy trunk. The hooks are "armed" by a pre-forward hook on the policy
    root and disarmed by ``collect()``: activation checkpointing replays layer
    forwards during backward (after the loss consumed the capture), and the
    disarmed hooks turn those replays into no-ops instead of re-running
    capture work (and, under TP, its collectives).
    """

    def __init__(self, policy_model: nn.Module, target_layer_ids: list[int]):
        self.target_layer_ids = [int(i) for i in target_layer_ids]
        self._policy_root = policy_model
        base, layers = _resolve_layers(policy_model)
        num_layers = len(layers)
        for layer_id in self.target_layer_ids:
            if layer_id != -1 and not (0 <= layer_id < num_layers):
                raise ValueError(
                    f"target_layer_id {layer_id} out of range for a policy with "
                    f"{num_layers} decoder layers."
                )
        self._modules_by_id: dict[int, nn.Module] = {}
        for layer_id in self.target_layer_ids:
            if layer_id == -1:
                embed = getattr(base, "embed_tokens", None)
                if embed is None:
                    raise ValueError(
                        "target_layer_ids includes -1 (embedding output) but the "
                        "policy model has no `.embed_tokens`."
                    )
                self._modules_by_id[layer_id] = embed
            else:
                self._modules_by_id[layer_id] = layers[layer_id]
        self._handles: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._armed = False

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def activate(self) -> None:
        if self._handles:
            return

        def arm_hook(_module: nn.Module, _args: Any, _kwargs: Any) -> None:
            self._armed = True

        def make_layer_hook(layer_id: int):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                if self._armed:
                    self._captured[layer_id] = _hook_output_tensor(output)

            return hook

        # The root pre-hook re-arms capture at each microbatch forward; the
        # checkpointed backward replay calls only layer forwards, so capture
        # stays disarmed there.
        self._handles.append(
            self._policy_root.register_forward_pre_hook(arm_hook, with_kwargs=True)
        )
        for layer_id, module in self._modules_by_id.items():
            self._handles.append(
                module.register_forward_hook(make_layer_hook(layer_id))
            )

    def deactivate(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()

    def clear(self) -> None:
        self._captured = {}
        self._armed = False

    def collect(self) -> torch.Tensor:
        """Concatenated target hidden states captured by the layer hooks."""
        missing = [i for i in self.target_layer_ids if i not in self._captured]
        if missing:
            raise RuntimeError(
                "DSpark hidden capture did not observe the policy forward "
                f"(missing layer ids {missing}). The capture hooks must be "
                "active during the training forward."
            )
        self._armed = False
        return torch.cat([self._captured[i] for i in self.target_layer_ids], dim=-1)


class DSparkRuntime:
    """Per-worker DSpark co-training state: draft model, capture, and loss."""

    def __init__(
        self,
        draft_model: Qwen3DSparkModel,
        dspark_options: dict[str, Any],
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.draft_model = draft_model
        self.options = dspark_options
        self.loss_weight = float(loss_weight)
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.capture: Optional[DSparkHiddenCapture] = None
        self._teacher_logits: Optional[torch.Tensor] = None
        self._num_microbatch_slots: int = 1

    def begin_global_batch(self, num_microbatch_slots: int) -> None:
        """Record this global batch's microbatch-slot count (identical on all ranks).

        The training loop sums per-microbatch losses across gradient
        accumulation, while each DSpark microbatch loss is a mean over that
        microbatch's (DP-all-reduced) anchor tokens. Dividing every microbatch
        term by the slot count turns the sum into an average of per-slot global
        means, keeping the draft gradient scale independent of gbs/mbs — the
        same effective scale as the policy loss, which normalizes each
        microbatch by the whole-global-batch token denominator instead.
        """
        self._num_microbatch_slots = max(int(num_microbatch_slots), 1)

    def attach_capture(self, policy_model: nn.Module) -> None:
        if self.capture is None:
            self.capture = DSparkHiddenCapture(
                policy_model, list(self.draft_model.config.target_layer_ids)
            )

    def stash_teacher_logits(
        self, logits: torch.Tensor, will_scale_inplace: bool
    ) -> None:
        """Record the policy's raw logits before any in-place temperature scaling.

        With temperature 1.0 no scaling happens and a detached view is exact and
        free; otherwise the in-place div would corrupt the view, so clone first.

        Under TP the logits arrive as a vocab-sharded DTensor; it is stashed
        as-is and gathered to the full vocab lazily in ``compute_loss`` so the
        full-size tensor does not coexist with the policy-loss peak.
        """
        if not isinstance(logits, DTensor) and self.tp_group is not None:
            if dist.get_world_size(self.tp_group) > 1:
                raise RuntimeError(
                    "DSpark co-training with tensor_parallel_size > 1 expects the "
                    "policy logits as a vocab-sharded DTensor, but got a plain "
                    f"{type(logits).__name__}; a bare local shard cannot be "
                    "gathered safely."
                )
        raw = logits.detach()
        self._teacher_logits = raw.clone() if will_scale_inplace else raw

    def compute_loss(self, data_dict: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the draft forward on the captured hiddens and compute the DSpark loss."""
        if self.capture is None or not self.capture.active:
            raise RuntimeError(
                "DSpark loss requested but hidden capture is not active; the worker "
                "must activate capture around the training forward."
            )
        teacher_logits = self._teacher_logits
        if teacher_logits is None:
            raise RuntimeError(
                "DSpark loss requested but no teacher logits were stashed for this "
                "microbatch."
            )
        if isinstance(teacher_logits, DTensor):
            # Vocab-sharded under TP; the draft's teacher gather needs the full
            # vocab dimension.
            teacher_logits = teacher_logits.full_tensor()
        target_hidden_states = self.capture.collect()

        input_ids = data_dict["input_ids"]
        loss_mask = data_dict["token_mask"].float()
        if "sample_mask" in data_dict:
            loss_mask = loss_mask * data_dict["sample_mask"].float().unsqueeze(-1)

        outputs: DSparkForwardOutput = self.draft_model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            teacher_logits=teacher_logits,
        )
        loss, terms = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=float(self.options["loss_decay_gamma"]),
            ce_loss_alpha=float(self.options["ce_loss_alpha"]),
            l1_loss_alpha=float(self.options["l1_loss_alpha"]),
            confidence_head_alpha=float(self.options["confidence_loss_alpha"]),
            process_group=self.dp_group,
            return_terms=True,
        )
        metrics = self._terms_to_metrics(terms)
        # See begin_global_batch: average across microbatch slots so the summed
        # backward matches the policy loss's global-mean gradient scale.
        loss = loss / self._num_microbatch_slots

        # Free per-microbatch stash; the next forward re-stashes.
        self._teacher_logits = None
        self.capture.clear()
        return loss, metrics

    @staticmethod
    def _terms_to_metrics(terms: dict[str, torch.Tensor]) -> dict[str, float]:
        """Flatten the loss terms into shape-stable scalar metrics.

        Ratios are formed per microbatch (num / den); the training-loop metric
        aggregation then averages across microbatches and ranks. This is a
        microbatch-weighted mean rather than an exact token-weighted global
        ratio, which is sufficient for trend monitoring.
        """
        metrics: dict[str, float] = {
            "draft_loss": float(terms["loss"].item()),
            "draft_ce_loss": float(terms["ce_loss"].item()),
            "draft_tv_loss": float(terms["l1_loss"].item()),
            "draft_conf_loss": float(terms["confidence_loss"].item()),
        }
        tau_den = float(terms["tau_den"].item())
        metrics["draft_tau"] = (
            float(terms["tau_num"].item()) / tau_den if tau_den > 0 else 0.0
        )
        # One host transfer per vector instead of one sync per position.
        pos_nums = terms["accept_rate_per_pos_num"].tolist()
        pos_dens = terms["accept_rate_per_pos_den"].tolist()
        for k, (num_k, den_k) in enumerate(zip(pos_nums, pos_dens)):
            metrics[f"draft_accept_rate@{k + 1}"] = num_k / den_k if den_k > 0 else 0.0
        return metrics


@contextlib.contextmanager
def dspark_capture_ctx(runtime: "DSparkRuntime"):
    """Keep the hidden-capture hooks active for the duration of a train call."""
    assert runtime.capture is not None, "attach_capture must run before training."
    runtime.capture.activate()
    try:
        yield
    finally:
        runtime.capture.deactivate()


DSPARK_OPTIMIZER_GROUP_NAMES = ("policy", "draft")


def optimizer_layout_record(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    """Record named param-group layout for resume validation.

    A DSpark co-training optimizer must consist of exactly the named groups
    ["policy", "draft"] in that order; anything else means the optimizer was
    not built by the dspark setup path and its state cannot be paired safely.
    """
    names = [group.get("name") for group in optimizer.param_groups]
    if tuple(names) != DSPARK_OPTIMIZER_GROUP_NAMES:
        raise ValueError(
            "DSpark co-training expects optimizer param groups named "
            f"{list(DSPARK_OPTIMIZER_GROUP_NAMES)} in that order, got {names}. "
            "The optimizer was not constructed by the dspark setup path."
        )
    return [
        {"name": group["name"], "num_params": len(group["params"])}
        for group in optimizer.param_groups
    ]


def dspark_meta_record(
    draft_model: Qwen3DSparkModel,
    model_name: str,
    optimizer: Optional[torch.optim.Optimizer],
) -> dict[str, Any]:
    config = draft_model.config
    return {
        "model_name": model_name,
        "block_size": int(config.block_size),
        "mask_token_id": int(config.mask_token_id),
        "target_layer_ids": [int(i) for i in config.target_layer_ids],
        "optimizer_layout": optimizer_layout_record(optimizer) if optimizer else None,
    }


def draft_checkpoint_dir(weights_path: str) -> str:
    """The draft's DCP directory: a SIBLING of the policy weights directory.

    The draft must live outside the policy weight tree because
    ``detect_checkpoint_format(weights_path)`` walks it recursively and would
    mis-detect the safetensors policy checkpoint as DCP after seeing the
    draft's ``.distcp`` files.
    """
    # abspath normalizes trailing slashes before dirname takes the parent.
    return os.path.join(
        os.path.dirname(os.path.abspath(weights_path)), DRAFT_CHECKPOINT_DIRNAME
    )


def save_draft_checkpoint(
    draft_model: Qwen3DSparkModel,
    weights_path: str,
    meta: dict[str, Any],
) -> None:
    """Save the draft's sharded weights (DCP) plus the dspark metadata record."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    draft_dir = draft_checkpoint_dir(weights_path)
    state_dict = get_model_state_dict(draft_model)
    dcp.save(state_dict, checkpoint_id=draft_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(os.path.join(draft_dir, DSPARK_META_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)
    if dist.is_initialized():
        dist.barrier()


def validate_dspark_checkpoint_meta(
    saved_meta: dict[str, Any], expected_meta: dict[str, Any]
) -> None:
    """Hard-error on inconsistent dspark checkpoint metadata.

    Architecture fields are always compared. The optimizer layout is compared
    only when the current run restores optimizer state (expected layout is
    non-None): weights-only loads (init_optimizer=False, e.g. eval or
    logprob-only workers) legitimately carry no layout while training
    checkpoints do.
    """
    keys = ["block_size", "mask_token_id", "target_layer_ids"]
    if expected_meta.get("optimizer_layout") is not None:
        keys.append("optimizer_layout")
    for key in keys:
        if saved_meta.get(key) != expected_meta.get(key):
            raise ValueError(
                f"DSpark checkpoint metadata mismatch for '{key}': checkpoint has "
                f"{saved_meta.get(key)!r}, current run expects {expected_meta.get(key)!r}. "
                "Refusing to resume with an inconsistent draft configuration."
            )


def load_draft_checkpoint(
    draft_model: Qwen3DSparkModel,
    weights_path: str,
    expected_meta: dict[str, Any],
) -> None:
    """Load the draft's weights from a checkpoint, validating the metadata record."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    draft_dir = draft_checkpoint_dir(weights_path)
    meta_path = os.path.join(draft_dir, DSPARK_META_FILENAME)
    if not os.path.isdir(draft_dir) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"DSpark draft entry not found in checkpoint at {draft_dir}. A dspark "
            "run cannot resume from a checkpoint saved without draft state."
        )
    with open(meta_path) as f:
        saved_meta = json.load(f)
    validate_dspark_checkpoint_meta(saved_meta, expected_meta)
    state_dict = get_model_state_dict(draft_model)
    dcp.load(state_dict, checkpoint_id=draft_dir)
    set_model_state_dict(draft_model, state_dict)


def load_dspark_checkpoint(
    checkpoint_manager: Any,
    model: nn.Module,
    draft_model: Qwen3DSparkModel,
    composite_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    weights_path: str,
    optimizer_path: Optional[str],
    model_name: str,
) -> None:
    """Load a dspark checkpoint with the composite-optimizer pairing rule.

    Policy weights load exactly as without a draft; draft weights load from
    the sibling draft entry with validated metadata; optimizer state pairs
    with the composite module whose param groups span policy + draft. This is
    the single home of that pairing invariant for both resume-at-setup and
    mid-run checkpoint loads.
    """
    checkpoint_manager.load_checkpoint(model=model, weights_path=weights_path)
    load_draft_checkpoint(
        draft_model,
        weights_path,
        expected_meta=dspark_meta_record(draft_model, model_name, optimizer),
    )
    if optimizer_path and optimizer is not None:
        checkpoint_manager.checkpointer.load_optimizer(
            optimizer=optimizer,
            model=composite_model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )
