"""Teacher-force identical tokens before/after a reversible SSM precision patch."""

import asyncio
import json
import os
import uuid
from pathlib import Path

import torch


def install():
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    original = VllmAsyncGenerationWorkerImpl.reset_prefix_cache_async
    if original.__name__ == "score_on_reset":
        return

    async def score_on_reset(self):
        root = Path(os.environ["SC_LP_DIAG_DIR"])
        request = root / "score-request"
        if not request.exists():
            return await original(self)
        path = Path(request.read_text())
        request.unlink()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reference = os.environ.get("SC_LP_REFERENCE_DIR")
        if reference:
            assert os.environ.get("SC_LP_CONTROL") == "prefix_cache_disabled"
            payload = torch.load(
                Path(reference) / "selected-600.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert payload["trainer_version"] == payload["train_step"] == 600
            torch.save(payload, root / "scoring-reference.pt")
            (root / "scoring-reference.json").write_text(
                json.dumps(
                    {
                        "source": reference,
                        "checkpoint_step": 600,
                        "control": os.environ["SC_LP_CONTROL"],
                    },
                    indent=2,
                )
                + "\n"
            )
        data = payload["data"]
        sequences = [
            data["input_ids"]
            .unbind()[i][: min(int(data["input_lengths"][i]), 4095)]
            .tolist()
            for i in range(len(payload["ids"]))
        ]
        execution = await self.llm.collective_rpc("diagnostic_execution_status")
        (root / "execution-profile.json").write_text(
            json.dumps(execution, indent=2) + "\n"
        )
        if os.environ.get("SC_LP_CONTROL") == "prefix_cache_disabled":
            assert len(execution) == 8
            assert all(
                not item["batch_invariance"]["env_enabled"]
                and not item["cache_scheduler"]["enable_prefix_caching"]
                and item["cache_scheduler"]["enable_chunked_prefill"]
                and item["cache_scheduler"]["max_num_batched_tokens"] == 2048
                and all(
                    layer["ssm_cache_dtype"] == "torch.float32"
                    and layer["conv_cache_dtype"] == "torch.bfloat16"
                    for layer in item["layers"]
                )
                for item in execution
            ), execution

        async def score_one(tokens):
            from vllm import SamplingParams

            sampling = SamplingParams(
                max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False
            )
            assert sampling.skip_reading_prefix_cache is True
            result = None
            async for output in self.llm.generate(
                prompt={"prompt_token_ids": tokens},
                sampling_params=sampling,
                request_id=f"yuekaiz-diag-{uuid.uuid4().hex}",
            ):
                result = output
            assert result is not None and result.prompt_logprobs is not None
            assert result.prompt_token_ids == tokens
            return torch.tensor(
                [0.0]
                + [
                    result.prompt_logprobs[i][token].logprob
                    for i, token in enumerate(tokens[1:], 1)
                ]
            )

        if os.environ.get("SC_LP_SCHEDULING_AB") == "1":
            from validation.kl_diagnosis.scheduling import score_scheduling

            assert os.environ.get("SC_LP_FORCE_LEGACY_A_LOG") == "0"
            await score_scheduling(
                self,
                sequences,
                original,
                score_one,
                root,
                serial_kernel_ab=os.environ.get("SC_LP_SERIAL_KERNEL_AB") == "1",
            )
            return None

        await original(self)
        before = await asyncio.gather(*(score_one(tokens) for tokens in sequences))
        torch.save(
            {"tokens": sequences, "before": before}, root / "prefill-baseline.pt"
        )
        await original(self)
        before_repeat = await asyncio.gather(
            *(score_one(tokens) for tokens in sequences)
        )
        changes = await self.llm.collective_rpc("diagnostic_ssm_precision")
        try:
            await original(self)
            after = await asyncio.gather(*(score_one(tokens) for tokens in sequences))
            await original(self)
            after_repeat = await asyncio.gather(
                *(score_one(tokens) for tokens in sequences)
            )
        finally:
            restored = await self.llm.collective_rpc(
                "diagnostic_ssm_precision", kwargs={"restore": True}
            )
            await original(self)
        restored_before = await asyncio.gather(
            *(score_one(tokens) for tokens in sequences)
        )
        torch.save(
            {
                "tokens": sequences,
                "before": before,
                "before_repeat": before_repeat,
                "after": after,
                "after_repeat": after_repeat,
                "restored_before": restored_before,
                "changes": changes,
                "restored": restored,
                "legacy_a_log_override": os.environ.get("SC_LP_FORCE_LEGACY_A_LOG")
                == "1",
            },
            root / "fixed-token-ssm-ab.pt",
        )
        # The completed precision A/B is durable before trying another kernel.
        # Switch only prefill; decode and the original BF16-rounded weights stay fixed.
        if os.environ.get("SC_LP_KERNEL_AB") == "1":
            try:
                switched = await self.llm.collective_rpc("diagnostic_prefill_backend")
                await original(self)
                triton = await asyncio.gather(*(score_one(t) for t in sequences))
                await original(self)
                triton_repeat = await asyncio.gather(*(score_one(t) for t in sequences))
            finally:
                kernel_restored = await self.llm.collective_rpc(
                    "diagnostic_prefill_backend", kwargs={"restore": True}
                )
                await original(self)
            flashinfer_restored = await asyncio.gather(
                *(score_one(t) for t in sequences)
            )
            torch.save(
                {
                    "tokens": sequences,
                    "flashinfer": restored_before,
                    "triton": triton,
                    "triton_repeat": triton_repeat,
                    "flashinfer_restored": flashinfer_restored,
                    "switched": switched,
                    "restored": kernel_restored,
                },
                root / "prefill-kernel-ab.pt",
            )
        return None

    VllmAsyncGenerationWorkerImpl.reset_prefix_cache_async = score_on_reset
