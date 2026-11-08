"""Read only checkpoint600's first GDN component and capture its actual stages."""

import hashlib
import json
import os
from pathlib import Path
import sys

from safetensors import safe_open
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    CPAwareGatedDeltaNet,
)
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from validation.kl_diagnosis.layer_metrics import compare


MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/models/Qwen/Qwen3.5-35B-A3B-Base"
CHECKPOINT = Path(
    "validation/results/yuekaiz-qwen35-35ba3b-ep16-fullbatch-la1-s42-pr4963"
    "/checkpoints/step_600/policy/weights/model"
)
PREFIX = "model.language_model.layers.0."


def load_component(
    tokens: torch.Tensor, shapes: dict
) -> tuple[dict, torch.Tensor, list]:
    """Use authoritative DCP saved_offsets; never infer ordering from filenames."""
    tensors = {}
    intervals = {name: [] for name in shapes}
    embeddings = torch.empty(
        tokens.numel(), shapes["embed_tokens.weight"][1], dtype=torch.bfloat16
    )
    embedding_coverage = torch.zeros(tokens.numel(), dtype=torch.int32)
    sources = []
    files = sorted(CHECKPOINT.glob("*.safetensors"))
    assert len(files) == 16, len(files)
    for path in files:
        stat = path.stat()
        sources.append(
            {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
        with safe_open(path, framework="pt", device="cpu") as stream:
            sharding = json.loads(stream.metadata()["DCP_SHARDING_INFO"])
            for name, shape in shapes.items():
                key = (
                    "model.language_model." + name
                    if name == "embed_tokens.weight"
                    else PREFIX + name
                )
                offsets = sharding[key]["saved_offsets"]
                view = stream.get_slice(key)
                local_shape = view.get_shape()
                assert offsets[1:] == [0] * (len(shape) - 1)
                assert list(local_shape[1:]) == list(shape[1:])
                start, end = offsets[0], offsets[0] + local_shape[0]
                assert 0 <= start < end <= shape[0], (key, offsets, local_shape, shape)
                intervals[name].append((start, end))
                if name == "embed_tokens.weight":
                    for index in (
                        ((tokens >= start) & (tokens < end))
                        .nonzero()
                        .flatten()
                        .tolist()
                    ):
                        row = int(tokens[index]) - start
                        embeddings[index].copy_(view[row : row + 1][0])
                        embedding_coverage[index] += 1
                else:
                    value = stream.get_tensor(key)
                    if name not in tensors:
                        tensors[name] = torch.empty(shape, dtype=value.dtype)
                    tensors[name][start:end].copy_(value)
    for name, chunks in intervals.items():
        chunks.sort()
        assert chunks[0][0] == 0 and chunks[-1][1] == shapes[name][0], name
        assert all(
            a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:], strict=True)
        ), name
    assert (embedding_coverage == 1).all()
    return tensors, embeddings, sources


@torch.no_grad()
def main(root: Path) -> None:
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).text_config
    reference = Path("validation/kl_diagnosis/results/21392111/scheduling-repeats.pt")
    sequences = torch.load(reference, map_location="cpu", weights_only=False)["tokens"]
    tokens = torch.tensor(sequences[0][:1024], dtype=torch.long)
    assert tokens.numel() > 32
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    layer = CPAwareGatedDeltaNet(config, layer_idx=0).eval()
    input_norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    torch.set_default_dtype(previous_dtype)
    shapes = {
        "linear_attn." + name.replace("_fp32_params.", ""): tuple(value.shape)
        for name, value in layer.state_dict().items()
    }
    shapes["input_layernorm.weight"] = (config.hidden_size,)
    shapes["embed_tokens.weight"] = (config.vocab_size, config.hidden_size)
    tensors, embedding, sources = load_component(tokens, shapes)
    state = {
        name: tensors["linear_attn." + name.replace("_fp32_params.", "")]
        for name in layer.state_dict()
    }
    layer.load_state_dict(state, strict=True)
    input_norm.load_state_dict(
        {"weight": tensors["input_layernorm.weight"]}, strict=True
    )
    assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
    cpu_input = input_norm(embedding.unsqueeze(0))
    # Establish a CPU FP32/FP64 mathematical reference before GPU kernels.
    sample = cpu_input[:, :16]
    cpu_projection = {}
    for name in ("qkv", "z", "b", "a"):
        weight = tensors["linear_attn.in_proj_" + name + ".weight"]
        out32 = F.linear(sample.float(), weight.float())
        out64 = F.linear(sample.double(), weight.double())
        cpu_projection[name] = compare(out32, out64)
    torch.cuda.set_device(0)
    layer.cuda()
    input_norm.cuda()
    hidden = input_norm(embedding.cuda().unsqueeze(0))
    stages = {"embedding": embedding, "input": hidden.cpu()}

    def save(name: str, value: torch.Tensor) -> None:
        stages[name] = value.detach().cpu().clone()

    def projection_hook(name: str):
        def hook(module, args, output):
            save(name, output)

        return hook

    for name in ("qkv", "z", "b", "a"):
        getattr(layer, "in_proj_" + name).register_forward_hook(projection_hook(name))
    conv = layer.causal_conv1d_fn
    assert conv is not None

    def capture_conv(*args, **kwargs):
        output = conv(*args, **kwargs)
        save("conv", output.transpose(1, 2))
        return output

    primitive = layer.chunk_gated_delta_rule

    def capture_primitive(q, k, v, **kwargs):
        for name, value in {
            "q": q,
            "k": k,
            "v": v,
            "g": kwargs["g"],
            "beta": kwargs["beta"],
        }.items():
            save(name, value)
        output, final = primitive(q, k, v, **kwargs)
        save("core", output)
        return output, final

    layer.causal_conv1d_fn = capture_conv
    layer.chunk_gated_delta_rule = capture_primitive
    layer.norm.register_forward_hook(projection_hook("norm"))
    outputs = [layer(hidden).detach().cpu() for _ in range(3)]
    stages["output"] = outputs[0]
    payload = {"tokens": tokens, "weights": tensors, "stages": stages}
    fingerprints = {
        name: hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, value in tensors.items()
    }
    report = {
        "status": "policy_completed",
        "scope": "First complete GDN component only, original checkpoint600 weights and R14 sequence0 token prefix. No full model, training, TP8 collectives or checkpoint writes.",
        "tokens": tokens.numel(),
        "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "source_checkpoint": str(CHECKPOINT),
        "source_token_file": str(reference),
        "checkpoint_sources": sources,
        "weight_sha256": fingerprints,
        "policy_primitive": primitive.__module__ + "." + primitive.__name__,
        "policy_conv": conv.__module__ + "." + conv.__name__,
        "cpu_projection_fp32_vs_fp64": cpu_projection,
        "cpu_vs_gpu_input_norm": compare(stages["input"], cpu_input),
        "policy_repeat_output": [compare(value, outputs[0]) for value in outputs[1:]],
        "policy_stage_shapes": {
            name: list(value.shape) for name, value in stages.items()
        },
    }
    for item in sources:
        stat = Path(item["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"])
    torch.save(payload, root / "layer-policy.pt")
    (root / "layer-policy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "tokens": tokens.numel(),
                "stages": report["policy_stage_shapes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
