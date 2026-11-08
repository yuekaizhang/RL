"""Notify the existing authorized conversation when the submitted diagnostic ends."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
root = repo / "validation/kl_diagnosis"
job = sys.argv[1]
preflight = len(sys.argv) > 2 and sys.argv[2] == "preflight"
recovery = len(sys.argv) > 2 and sys.argv[2] == "recovery"
kwargs_preflight = len(sys.argv) > 2 and sys.argv[2] == "kwargs"
analysis = len(sys.argv) > 2 and sys.argv[2] == "analysis"
kernel_replay = len(sys.argv) > 2 and sys.argv[2] == "layer"
component_replay = len(sys.argv) > 2 and sys.argv[2] == "component"
conv_replay = len(sys.argv) > 2 and sys.argv[2] == "conv"
source_job = sys.argv[3] if analysis or kernel_replay else job
assert job.isdecimal() and source_job.isdecimal()
while True:
    queue = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T|%R"], capture_output=True, text=True
    )
    state = {
        "job": job,
        "updated_epoch": time.time(),
        "queue": queue.stdout.strip(),
        "query_error": queue.stderr.strip() if queue.returncode else None,
    }
    status_kind = (
        "recovery-preflight"
        if recovery
        else ("preflight" if preflight else "diagnosis")
    )
    if kwargs_preflight:
        status_kind = "worker-kwargs-preflight"
    if analysis:
        status_kind = "scheduling-analysis"
    if kernel_replay:
        status_kind = "gdn-kernel-replay"
    if component_replay:
        status_kind = "gdn-component-replay"
    if conv_replay:
        status_kind = "gdn-conv-precision"
    (root / f"results/{status_kind}-status.json").write_text(
        json.dumps(state, indent=2) + "\n"
    )
    if not queue.returncode and not queue.stdout.strip():
        accounting = subprocess.run(
            [
                "sacct",
                "-j",
                job,
                "-X",
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        (root / "results/final-accounting.txt").write_text(accounting.stdout)
        target = json.loads(
            (repo / "validation/mcore/results/watch-target.json").read_text()
        )["thread_id"]
        message = (
            "[已授权的 gen_kl_error 排查任务完成/失败通知] 请继续原任务，不创建新会话或子agent。项目 "
            + str(repo)
            + "。先读 validation/kl_diagnosis/HANDOFF.md，再查询 sacct -j "
            + job
            + "，读取 validation/kl_diagnosis/slurm/"
            + job
            + "-logs/ray-driver.log 与 results/"
            + job
            + "/analysis.json（若存在）。以HANDOFF最新附录及submitted.tsv为准，核实本轮实际控制变量、相同token与policy参考、并发重复评分、实际cache/scheduler配置、train/eval和单步漂移、refit发送/接收探针。不要假设每轮都包含旧A_log A/B。原1030的GDN_ATTN明确不支持batch-invariant模式，不重复提交该失败模式。若失败主动修复并续验证，先查已有作业防重复。不得动旧900步hold，不得修改原checkpoint；原MCore300 stop标记保持。最终给用户证据、根因范围、必要修复与验证结果。"
        )
        if preflight:
            message = (
                "[已授权的 gen_kl_error 预检完成/失败通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/preflight-"
                + job
                + ".out/.err 与 validation/kl_diagnosis/results/preflight-"
                + job
                + ".json（若存在）。核实原1030镜像两侧实际Ray worker解释器、GPU绑定与探针方法。"
                "再核实 submitted.tsv 最新全EP16诊断作业，避免重复提交；预检失败应主动修复并处理其afterok依赖。"
                "预检通过不等于KL诊断完成，继续验证checkpoint600固定token A_log FP32 A/B、train/eval和单步漂移、refit探针。"
                "不得修改原checkpoint，不得释放900步hold，保留MCore stop标记。"
            )
        if recovery:
            message = (
                "[已授权的 MCore600/固定token排查恢复预检通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 和 session/20261102_233142/handoff_mcore.md 最新附录。查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/recovery-preflight-"
                + job
                + ".out/.err、results/recovery-preflight-"
                + job
                + "/recipe-preflight.json、generation-controls-preflight.json，以及 validation/mcore/results/entry-preflight-"
                + job
                + "-la{0,1}.json。检查真实Hydra/MasterConfig配置和两组MCore精确文件入口导入。失败主动修复预检并更新afterok依赖；成功继续核实最新MCore600及固定token完整作业。以最新submitted清单为准避免重复，不释放900hold、不修改原checkpoint或300stop。"
            )
        if kwargs_preflight:
            message = (
                "[已授权的固定token worker参数预检通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 最新附录，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/worker-preflight-"
                + job
                + ".out/.err、results/worker-preflight-"
                + job
                + "/recipe-preflight.json、prefix-config-tests.xml、worker-kwargs-preflight.json。核实真实typed配置、实际BaseVllmGenerationWorker参数组装的旧错误复现和修正后AsyncEngineArgs配置。失败主动修复小预检并处理依赖，成功继续核实submitted.tsv最新完整诊断；不要重复提交已有完整作业。MCore600独立监控，900hold及原checkpoint/300stop保持。"
            )
        if analysis:
            message = (
                "[已授权的固定token保存数据分析通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 最新附录，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/scheduling-analysis-"
                + job
                + ".out/.err 和 validation/kl_diagnosis/results/"
                + source_job
                + "/analysis.json。这是已有完整捕获数据的分析恢复，不重跑模型或训练。"
                "核实相同token与policy参考、并发重复评分、实际cache/scheduler配置、"
                "train/eval和单步漂移、refit探针；失败只修复必要分析，先查已有作业防重复。"
                "MCore600独立监控，900hold及原checkpoint/300stop保持。"
            )
        if kernel_replay:
            message = (
                "[已授权的单层GDN内核重放通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 最新附录，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/gdn-kernel-"
                + job
                + ".out/.err 和 validation/kl_diagnosis/results/"
                + source_job
                + "/gdn-kernel-analysis.json。只使用已有单层输入，不重跑完整模型或训练。"
                "若HANDOFF记录本轮包含policy primitive对照，还必须核实policy-gdn-analysis.json"
                "及新产物中的policy_primitive_parity，不能把旧单层结果当本轮完成。"
                "核实原FlashInfer/Triton路径重复性、相同输入输出/state差异、共同q/k归一化"
                "及FP64逐token参考的计算假设；单层结果不能直接等同完整模型KL根因。"
                "失败主动修复并续验证，先查已有作业防重复。MCore600独立监控，"
                "900hold及原checkpoint/300stop保持。"
            )
        if component_replay:
            message = (
                "[已授权的完整GDN组件对照完成/失败通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 最新附录，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/gdn-layer-"
                + job
                + ".out/.err 与 validation/kl_diagnosis/results/layer-"
                + job
                + "/layer-policy.json、layer-analysis.json。本轮只读checkpoint600第一层及R14固定token，"
                "不是完整模型或训练；核实DCP saved_offsets覆盖、CPU参考、实际policy路径、"
                "相同输入下投影/卷积/gate/归一化/GDN输出、单变量dt_bias精度与重复性。"
                "TP8-shaped在单GPU顺序模拟，不等于真实TP8引擎或NCCL验证。"
                "若失败先核实已有作业防重复，再修必要探针续验证；不得修改原checkpoint、"
                "释放900hold或重启已完成MCore600，保留两个stop标记。"
            )
        if conv_replay:
            message = (
                "[已授权的GDN卷积精度对照完成/失败通知] 继续原任务，不创建新会话或子agent。项目 "
                + str(repo)
                + "。先读 validation/kl_diagnosis/HANDOFF.md 最新附录，查询 sacct -j "
                + job
                + "，读取 validation/kl_diagnosis/slurm/conv-precision-"
                + job
                + ".out/.err 与 validation/kl_diagnosis/results/conv-"
                + job
                + "/conv-analysis.json、layer-analysis.json。本轮仅使用已保存的21509317组件输入，"
                "不加载完整模型或训练；核实CPU FP64卷积参考、BF16乘积反事实、实际两端卷积、"
                "仅JIT诊断实例增加FP32乘法cast前后和恢复重复性、完整组件输出影响。"
                "局部结果不能等同完整模型KL根因或真实TP8验证。失败先查已有作业再修必要分析；"
                "不得改原checkpoint或释放900hold，保留MCore两个stop标记。"
            )
        env = dict(os.environ)
        prefix = "/lustre/fsw/portfolios/coreai/users/yuekaiz/tools"
        env["PATH"] = (
            prefix + "/node/bin:" + prefix + "/codex/bin:/usr/local/bin:/usr/bin:/bin"
        )
        result = subprocess.run(
            [
                prefix + "/codex/bin/codex",
                "queue",
                "--thread",
                target,
                "--message",
                message,
                "--resume",
            ],
            env=env,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(
            "CONVERSATION_NOTIFICATION",
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
            flush=True,
        )
        break
    print(json.dumps(state), flush=True)
    time.sleep(60)
