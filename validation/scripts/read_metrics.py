"""Read native SC scalar logs without evaluating arbitrary Python expressions."""

import ast
import json
import math
from pathlib import Path
import re
import sys


class ScalarLiterals(ast.NodeTransformer):
    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in {"float64", "float32", "int64", "int32"}
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.visit(node.args[0])
        return node

    def visit_Name(self, node):
        if node.id in {"nan", "inf"}:
            return ast.Constant(float(node.id))
        return node


def read_metrics(path):
    rows = []
    step_time = None
    for line in Path(path).read_text(errors="replace").splitlines():
        timing = re.search(r"Total step time:\s*([\d.]+)s", line)
        if timing:
            step_time = float(timing.group(1))
        step = re.search(r"train step (\d+)/", line)
        if step and rows:
            rows[-1]["nemo_rl/step"] = int(step.group(1))
        if "step_metrics=" not in line:
            continue
        text = re.sub(r"\x1b\[[0-9;]*m", "", line.split("step_metrics=", 1)[1])
        tree = ScalarLiterals().visit(ast.parse(text, mode="eval"))
        row = ast.literal_eval(tree)
        if step_time is not None:
            row["timing/train/total_step_time"] = step_time
        rows.append(row)
        step_time = None
    return rows


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = read_metrics(path)
        keys = [
            "reward",
            "loss",
            "grad_norm",
            "num_valid_samples",
            "weight_version_lag",
            "evicted_stale_prompt_groups",
            "aborted_stale_inflight_groups",
            "dropped_prompt_groups",
        ]
        result = {
            "path": path,
            "updates": len(rows),
            "metrics": [{k: r.get(k) for k in keys} for r in rows],
            "nonfinite": [
                {
                    k: v
                    for k, v in r.items()
                    if isinstance(v, (int, float)) and not math.isfinite(v)
                }
                for r in rows
            ],
        }
        print(json.dumps(result, indent=2))
