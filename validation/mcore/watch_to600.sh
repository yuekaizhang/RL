#!/bin/bash
# Same-conversation notifier used by the authorized 10-minute process monitor.
set -euo pipefail
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
cd "$repo"
reason="${1:-periodic}"
[[ ! -f validation/mcore/results/stop-to600-watch || "$reason" == completed ]] || exit 0
export PATH=/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/node/bin:/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin:/usr/local/bin:/usr/bin:/bin
thread=$(/usr/bin/python3 -c 'import json; print(json.load(open("validation/mcore/results/watch-target.json"))["thread_id"])')
message="[用户授权的 MCore600 每10分钟巡检/事件通知：$reason] 继续原任务，不创建新会话或子agent。项目 $repo。先读 session/20261102_233142/handoff_mcore.md 最新附录、validation/mcore/results/live-status-to600.json 和 slurm/submitted-to600.tsv。实际查询 squeue/sacct、读取新作业错误日志、原生指标、checkpoint、gen_kl_error、lag、LR、有效样本数。两组从完整checkpoint300续到600，同W&B run，chunk512，CPU-staged Adam恢复；新checkpoint/log写在yuekaiz-mcore-la{0,1}-600下，原300不删改。600配置恢复已保存的constant scheduler，不能重置warmup。发现失败、DependencyNeverSatisfied、监控退出、训练停滞主动修复/续跑，先查已有作业避免重复。保持1030镜像、yuekaiz作业名前缀、batch/长度/并行/LR不变；有限KL升高只报警不停止。正常简短汇报步数和KL趋势。也核实validation/kl_diagnosis/HANDOFF.md和submitted.tsv中的AutoModel固定token排查任务（以最新记录为准），继续必要修复验证。旧900作业21315214/16/17必须保持hold。原MCore300 stop-10min-watch保持；600两组完成后核对结果、更新comparison-to600图、取消未使用后续段并写stop-to600-watch。"
date -Is
/usr/bin/timeout 45 /lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin/codex queue \
  --thread "$thread" --message "$message" --resume
