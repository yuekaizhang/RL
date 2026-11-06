"""Fixed-token repeatability under controlled request submission concurrency."""

import asyncio
import json
import time

import torch


async def score_scheduling(worker, sequences, reset, score_one, root):
    modes = {"serial1": 1, "microbatch4": 4, "concurrent32": 32}
    started = time.monotonic()
    initial_requests = worker.llm.get_num_unfinished_requests()
    while True:
        if worker.llm.get_num_unfinished_requests() == 0:
            await asyncio.sleep(0.2)
            if worker.llm.get_num_unfinished_requests() == 0:
                break
        assert time.monotonic() - started < 180, "Background rollouts failed to drain"
        await asyncio.sleep(0.1)
    isolation = {
        "initial_unfinished_requests": initial_requests,
        "drain_seconds": time.monotonic() - started,
        "max_foreign_requests_during_measurement": 0,
    }

    async def watch_background():
        while True:
            foreign = sum(
                not request.external_req_id.startswith("yuekaiz-diag-")
                for request in tuple(
                    worker.llm.output_processor.request_states.values()
                )
            )
            isolation["max_foreign_requests_during_measurement"] = max(
                isolation["max_foreign_requests_during_measurement"], foreign
            )
            await asyncio.sleep(0.02)

    watcher = asyncio.create_task(watch_background())
    await worker.llm.collective_rpc(
        "diagnostic_resident_snapshot", kwargs={"label": "before"}
    )

    async def run(mode):
        await reset(worker)
        values = []
        for start in range(0, len(sequences), modes[mode]):
            values.extend(
                await asyncio.gather(
                    *(
                        score_one(tokens)
                        for tokens in sequences[start : start + modes[mode]]
                    )
                )
            )
        return values

    try:
        # Warm the same token shapes before collecting repeatability measurements.
        for mode in modes:
            await run(mode)
        scores = {mode: [] for mode in modes}
        order, timings = [], []
        for round_order in (
            ("concurrent32", "serial1", "microbatch4"),
            ("microbatch4", "serial1", "concurrent32"),
            ("serial1", "concurrent32", "microbatch4"),
        ):
            for mode in round_order:
                started = time.monotonic()
                scores[mode].append(await run(mode))
                order.append(mode)
                timings.append(time.monotonic() - started)
                # Preserve every completed pass if a later diagnostic fails.
                torch.save(
                    {
                        "tokens": sequences,
                        "scores": scores,
                        "order": order,
                        "seconds": timings,
                    },
                    root / "scheduling-repeats.pt",
                )
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        (root / "scheduling-isolation.json").write_text(
            json.dumps(isolation, indent=2) + "\n"
        )
    await worker.llm.collective_rpc(
        "diagnostic_resident_snapshot", kwargs={"label": "after"}
    )
