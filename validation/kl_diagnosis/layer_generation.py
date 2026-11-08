"""Compare pinned vLLM GDN stages with the saved, complete policy component."""

import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

from validation.kl_diagnosis.layer_metrics import compare


@torch.no_grad()
def main(root: Path) -> None:
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    torch.cuda.set_device(0)
    payload = torch.load(
        root / "layer-policy.pt", map_location="cpu", weights_only=False
    )
    policy = payload["stages"]
    weights = {
        name.removeprefix("linear_attn."): value.cuda()
        for name, value in payload["weights"].items()
    }
    hidden = policy["input"].cuda().squeeze(0)
    length = hidden.shape[0]
    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    key_dim = (
        weights["in_proj_qkv.weight"].shape[0] - weights["in_proj_z.weight"].shape[0]
    )
    key_dim //= 2
    value_dim = weights["in_proj_z.weight"].shape[0]
    key_heads, value_heads = key_dim // 128, value_dim // 128
    assert (key_heads, value_heads) == (16, 32)
    q_weight, k_weight, v_weight = weights["in_proj_qkv.weight"].split(
        [key_dim, key_dim, value_dim]
    )
    conv_q, conv_k, conv_v = (
        weights["conv1d.weight"].squeeze(1).split([key_dim, key_dim, value_dim])
    )
    with set_current_vllm_config(VllmConfig()):
        input_norm = GemmaRMSNorm(hidden.shape[-1], eps=1e-6).cuda().bfloat16()
        norm = RMSNormGated(
            128,
            eps=1e-6,
            norm_before_gate=True,
            activation="silu",
            device="cuda",
            dtype=torch.bfloat16,
        )
    input_norm.weight.copy_(weights["input_layernorm.weight"])
    norm.weight.copy_(weights["norm.weight"])
    inference_input = input_norm.forward_cuda(policy["embedding"].cuda())
    report = json.loads((root / "layer-policy.json").read_text())
    report["status"] = "generation_in_progress"
    report["generation_scope"] = (
        "Actual pinned vLLM conv/post-conv/FlashInfer/gated-norm kernels; F.linear reconstructs fused qkvz/ba projections. "
        "TP8-shaped slices execute sequentially on ONE GPU with a local sum, not actual TP8 linear classes or NCCL collectives. "
        "All pipelines share the saved policy input to separate local from propagated differences. No full model or training."
    )
    report["input_norm_vllm_vs_policy"] = compare(
        inference_input.cpu(), policy["input"].squeeze(0)
    )
    report["equal_input_controls"] = {}
    controls = report["equal_input_controls"]

    def prepare(conv_output, a, b, a_log, dt_bias, heads):
        return fused_post_conv_prep(
            conv_output=conv_output.contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            A_log=a_log.contiguous(),
            dt_bias=dt_bias.contiguous(),
            num_k_heads=heads,
            head_k_dim=128,
            head_v_dim=128,
            apply_l2norm=True,
            output_g_exp=False,
        )

    def core(q, k, v, g, beta, *, normalize=False):
        h_v = v.shape[-2]
        out, _ = fi_chunk_gated_delta_rule(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0).float(),
            beta=beta.unsqueeze(0).float(),
            initial_state=torch.zeros(
                1, h_v, 128, 128, device="cuda", dtype=torch.float32
            ),
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=normalize,
        )
        return out.squeeze(0)

    # Cross-inject shared actual policy inputs to isolate prep, core and norm.
    shared_a, shared_b = (policy[x].cuda().squeeze(0) for x in ("a", "b"))
    shared_conv = policy["conv"].cuda().squeeze(0)
    post = {}
    for dtype_name, dtype in (("bf16_dt", torch.bfloat16), ("fp32_dt", torch.float32)):
        prep = prepare(
            shared_conv,
            shared_a,
            shared_b,
            weights["A_log"],
            weights["dt_bias"].to(dtype),
            key_heads,
        )
        post[dtype_name] = prep
        controls[dtype_name] = {}
        for name, actual in zip(("q", "k", "v", "g", "beta"), prep, strict=True):
            expected = policy[name].squeeze(0)
            if name in ("q", "k"):
                # This is a mathematical normalization control, not a capture
                # of FLA's internal normalization kernel.
                expected = expected.reshape(length, key_heads, 2, 128)[:, :, 0]
                expected = (
                    expected.float()
                    * torch.rsqrt(
                        expected.float().square().sum(-1, keepdim=True) + 1e-6
                    )
                ).to(expected.dtype)
            controls[dtype_name][name + "_vs_policy"] = compare(actual.cpu(), expected)
        gate64 = -weights["A_log"].cpu().double().exp() * F.softplus(
            shared_a.cpu().double() + weights["dt_bias"].cpu().double()
        )
        controls[dtype_name]["g_vs_cpu_fp64_full_precision_params"] = compare(
            prep[3].cpu(), gate64
        )
        controls[dtype_name]["beta_vs_cpu_fp64"] = compare(
            prep[4].cpu(), shared_b.cpu().double().sigmoid()
        )
    shared_core = core(
        *(policy[n].cuda().squeeze(0) for n in ("q", "k", "v", "g", "beta")),
        normalize=True,
    )
    controls["flashinfer_shared_policy_inputs_vs_policy_core"] = compare(
        shared_core.cpu(), policy["core"].squeeze(0)
    )
    shared_z = policy["z"].cuda().reshape(-1, 128)
    shared_norm = norm.forward_cuda(policy["core"].cuda().reshape(-1, 128), shared_z)
    controls["vllm_norm_shared_policy_core_z_vs_policy_norm"] = compare(
        shared_norm.cpu(), policy["norm"]
    )

    def pipeline(tp: int, dt_dtype: torch.dtype) -> dict:
        results = {
            name: []
            for name in (
                "qkv",
                "z",
                "b",
                "a",
                "conv",
                "q",
                "k",
                "v",
                "g",
                "beta",
                "core",
                "norm",
                "partial",
            )
        }
        for rank in range(tp):
            qkv_weights = [
                x.chunk(tp, dim=0)[rank] for x in (q_weight, k_weight, v_weight)
            ]
            z_weight = weights["in_proj_z.weight"].chunk(tp, dim=0)[rank]
            qkvz = F.linear(hidden, torch.cat([*qkv_weights, z_weight], dim=0))
            mixed, z = qkvz.split(
                [(2 * key_dim + value_dim) // tp, value_dim // tp], dim=-1
            )
            ba_weights = [
                weights["in_proj_" + name + ".weight"].chunk(tp, dim=0)[rank]
                for name in ("b", "a")
            ]
            b, a = F.linear(hidden, torch.cat(ba_weights)).chunk(2, dim=-1)
            conv_weight = torch.cat(
                [x.chunk(tp, dim=0)[rank] for x in (conv_q, conv_k, conv_v)]
            )
            conv_state = torch.zeros(
                2, conv_weight.shape[0], 3, device="cuda", dtype=torch.bfloat16
            )
            conv_output = causal_conv1d_fn(
                mixed.transpose(0, 1),
                conv_weight,
                None,
                conv_state,
                cu,
                cache_indices=torch.ones(1, device="cuda", dtype=torch.int32),
                has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
                activation="silu",
            ).transpose(0, 1)
            prep = prepare(
                conv_output,
                a,
                b,
                weights["A_log"].chunk(tp)[rank],
                weights["dt_bias"].chunk(tp)[rank].to(dt_dtype),
                key_heads // tp,
            )
            attention = core(*prep)
            normed = norm.forward_cuda(
                attention.reshape(-1, 128), z.reshape(-1, 128)
            ).reshape(length, value_dim // tp)
            partial = F.linear(
                normed, weights["out_proj.weight"].chunk(tp, dim=1)[rank]
            )
            for name, value in {
                "qkv": mixed,
                "z": z,
                "b": b,
                "a": a,
                "conv": conv_output,
                "core": attention,
                "norm": normed,
                "partial": partial,
            }.items():
                results[name].append(value)
            for name, value in zip(("q", "k", "v", "g", "beta"), prep, strict=True):
                results[name].append(value)
        # q/k/v occupy separate contiguous head ranges inside each TP slice.
        joined = {}
        for name, chunks in results.items():
            if name in ("partial", "qkv", "conv"):
                continue
            joined[name] = torch.cat(chunks, dim=1).cpu()
        for name in ("qkv", "conv"):
            separated = [
                x.split([key_dim // tp, key_dim // tp, value_dim // tp], dim=-1)
                for x in results[name]
            ]
            joined[name] = torch.cat(
                [torch.cat([x[i] for x in separated], dim=-1) for i in range(3)], dim=-1
            ).cpu()
        joined["output"] = torch.stack(results["partial"]).sum(0).cpu()
        return joined

    report["pipelines"] = {}
    saved = {
        "equal_input_postconv": {
            key: tuple(x.cpu() for x in values) for key, values in post.items()
        }
    }
    for label, tp, dtype in (
        ("tp1_bf16_dt", 1, torch.bfloat16),
        ("tp8_shaped_bf16_dt", 8, torch.bfloat16),
        ("tp8_shaped_fp32_dt", 8, torch.float32),
    ):
        print("LAYER_PIPELINE", label, flush=True)
        first = pipeline(tp, dtype)
        second = pipeline(tp, dtype)
        comparisons = {}
        for name in (
            "qkv",
            "z",
            "a",
            "b",
            "conv",
            "v",
            "g",
            "beta",
            "core",
            "norm",
            "output",
        ):
            expected = policy[name].reshape(first[name].shape)
            comparisons[name] = compare(first[name], expected)
        report["pipelines"][label] = {
            "versus_policy": comparisons,
            "repeat_output": compare(second["output"], first["output"]),
        }
        saved[label] = first
        (root / "layer-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    report["dt_precision_only_tp8_output_effect"] = compare(
        saved["tp8_shaped_fp32_dt"]["output"], saved["tp8_shaped_bf16_dt"]["output"]
    )
    for source in report["checkpoint_sources"]:
        stat = Path(source["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (source["bytes"], source["mtime_ns"])
    report["status"] = "completed"
    torch.save(saved, root / "layer-generation.pt")
    (root / "layer-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "pipelines": report["pipelines"],
                "controls": controls,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
