"""Gate the long jobs on real LA0/LA1 updates and MCore checkpoint recovery."""

import json
import math
from pathlib import Path

from validation.scripts.read_metrics import read_metrics

root = Path("validation/mcore/results")
cases = {
    "la0": root / "smoke-la0-attempt4.log",
    "la1": root / "smoke-la1.log",
    "la1_resume": root / "smoke-la1-resume.log",
}
report = {}
for name, path in cases.items():
    rows = read_metrics(path)
    expected = [3] if name == "la1_resume" else [1, 2, 3]
    assert [row["nemo_rl/step"] for row in rows] == expected
    assert "SC run complete" in path.read_text(errors="replace")
    for row in rows:
        assert row["num_valid_samples"] == 32
        for key in ["gen_kl_error", "policy_kl_error", "reward", "grad_norm", "loss"]:
            assert math.isfinite(row[key])
        assert row["dropped_prompt_groups"] == row["evicted_stale_prompt_groups"] == 0
        assert row["num_masked_seqs_by_logprob_error"] == 0
        assert row["weight_version_lag"] in ([0] if name == "la0" else [0, 1])
    if name == "la1":
        assert max(row["weight_version_lag"] for row in rows) == 1
    if name == "la1_resume":
        assert "Restoring replay buffer metadata" in path.read_text(errors="replace")
    report[name] = {
        "steps": expected,
        "kl": [row["gen_kl_error"] for row in rows],
        "lag": [row["weight_version_lag"] for row in rows],
        "lr": [row["lr"] for row in rows],
    }
output = {"status": "passed", "runs": report}
(root / "smoke-proof.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
