#!/bin/bash
# User-authorized periodic checks delivered to the original Codex conversation.
set -euo pipefail
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
cd "$repo"
[[ ! -e validation/mcore/results/stop-10min-watch ]] || exit 0
export PATH=/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/node/bin:/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin:/usr/local/bin:/usr/bin:/bin
thread=$(/usr/bin/python3 -c 'import json; print(json.load(open("validation/mcore/results/watch-target.json"))["thread_id"])')
message='[用户授权的每10分钟定时巡检] 检查 MCore LA0/LA1 300步实验。项目目录 /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963。先阅读 session/20261102_233142/handoff_mcore.md 最新附录及 validation/mcore/results/live-status.json；实际查询 squeue/sacct，并读取最新作业错误日志、训练步数、checkpoint、gen_kl_error、lag、LR和有效样本数，不要只相信旧monitor日志。原两组在第12步 logprob backward OOM；续跑配置已统一减小 logprob_chunk_size 到512，保持batch/长度/并行规模不变，从checkpoint10恢复；以最新submitted.tsv和handoff记录为准。发现失败、DependencyNeverSatisfied、监控退出或训练长期无进展，主动在本会话报告并处理已授权的可逆修复/续跑；先核实已有作业以防重复提交，不随意改变训练目标。正常时简短汇报步数与KL趋势。所有作业名前缀yuekaiz，使用原1030镜像和本地有权限的W&B账号。旧AutoModel900步作业21315214/16/17必须保持用户hold，不得释放。两条MCore均完成300步后核对结果、更新比较图，移除本定时任务（crontab中YUEKAIZ_MCORE_10MIN段）并写stop-10min-watch标记。继续原有任务，不创建子agent或新会话。'
if [[ "${1:-}" == --self-test ]]; then
  message='[每10分钟监控链路自测] 本条由本机codex queue --resume送入原会话。请确认已收到即可，继续当前OOM修复和续跑，不重复提作业、不释放900步hold。定时巡检授权与目标不变。'
fi
date -Is
/usr/bin/timeout 60 /lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin/codex queue \
  --thread "$thread" --message "$message" --resume
