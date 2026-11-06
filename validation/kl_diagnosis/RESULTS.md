# AutoModel checkpoint600 固定 token KL 排查

2026-11-06。结论：确认 AutoModel refit 会把本应保留 FP32 的 `A_log`
先量化为 BF16，再装入 vLLM 的 FP32 参数。这是一处可修复的精度损失；
本次同 token A/B 不支持它是大部分 `gen_kl_error` 的来源。

## 实际执行与数据

- 完整诊断作业 `21378262`：原1030镜像，EP16 policy + TP8 vLLM，
  读取原 LA1 checkpoint600，完成600→602两次真实 optimizer 更新。
  原 checkpoint 只读，没有改写原训练 run。
- 64份训练 logprob、32份更新后同 token 重算数据、3次 refit 的 sender
  和全部8个 receiver，以及固定 token A/B 均完整保存。
- 完整作业最终 exit1 来自训练结束后的分析断言。恢复时保留了未完成的
  rollout ownership；即使诊断配置改为 LA0，第二批也仍使用 version600。
  实际为第一批 trainer600/rollout600、第二批 trainer601/rollout600。
  分析已按实际 lag 分开，未将第二批计作同版本对照。
- 分析作业 `21378945` COMPLETED，exit0。机器可读结果见
  [summary.json](summary.json)（保存的 analysis.json 副本）和
  [capture-status.json](results/21378262/capture-status.json)。

## 已验证的结果

| 对照 | 有效 response tokens | mean K3 |
|---|---:|---:|
| 第一批，policy 与 generation，同 version600，lag0 | 27,586 | 0.0020589322 |
| 第二批，policy601 与 generation600，lag1 | 23,890 | 0.0023261582 |
| 训练 forward 与更新前 eval 重算，两批合计 | 51,476 | **0，所有 logprob 完全一致** |
| 同 token，第一次 optimizer 更新前后 policy | 27,586 | 0.0033453320 |
| 同 token，第二次 optimizer 更新前后 policy | 23,890 | 0.0026494962 |

第一、二行在不同 batch 上，不能相减来估计 lag1 的因果影响。
两步原生训练 `gen_kl_error` 为0.00205893595和0.00232616201；离线重算
与原生值差约4e-9。两步 LR 均1e-6，有效样本均32，lag分别0和1。
单步漂移说明权重确实发生更新；这些 K3 不能线性相加来预测 gen KL。

3次 refit 中，每次1026个 HF tensor 在8个接收端的名字、shape、dtype、
抽样数值逐位匹配。3D expert stack 中每个 expert 都有抽样。
这支持传输正确，但不是每个 vLLM resident tensor/expert 映射的穷尽证明。
相邻 refit 抽样中，分别有385、371个 tensor 的值发生变化，也排除了三次
都发送相同缓存 payload 的情况。

## A_log 精度 A/B

第一批的32条相同 token、同一个 checkpoint600 policy、同一条 vLLM
prefill scoring 路径，仅将30层 `A_log` 从 BF16 量化值换成源 FP32 值。
8个 TP rank 都先验证 resident `A_log` 精确等于 BF16-rounded source，
再替换，并在评分后恢复；每次评分前重置 prefix cache。
`dt_bias` 没有改变：原1030 vLLM 的 `A_log` 明确为 FP32，`dt_bias`
使用模型默认 dtype，二者的存储约定不同。

| 同27,586个 response token 对照 | mean K3 | reverse mean K3 |
|---|---:|---:|
| 原 BF16 A_log，vLLM prefill 与 policy | 0.0024520390 | 0.0024987263 |
| 源 FP32 A_log，vLLM prefill 与相同 policy | 0.0022403660 | 0.0025493990 |
| 原 vLLM prefill 与原 vLLM decode，同权重 | 0.0020584344 | 0.0022291641 |

第一个方向平均 K3 降低约8.63%，反向反而增加约2.03%，最大绝对 logprob
差也未改善。本次只有一批、顺序执行的 before/after，没有重复运行负对照。
不能把它描述成整体训练 KL 改善9%，或认为改完 `A_log` 就解决了问题。
实际 rollout decode 与修复后的 prefill 也不能混作同一条计算路径的 A/B。

同权重 vLLM 自身的 prefill/decode 差异已达到约0.00206。结合 train/eval
完全一致、抽样传输匹配，剩余排查范围重点是生成与重算路径的数值、
kernel、缓存状态精度和 batching。原镜像代码采用 GDN chunked prefill
与 recurrent decode，运行日志显示 decode kernel 为 CUDA。当前证据尚不能
指认某一个 kernel 为 bug，也不能断言剩余误差全部正常。

## 最小代码修复与验证

`_refit_tensor_dtype` 对源 FP32、叶子名为 `A_log` 的 tensor 保留 FP32。
已有 `prepare_refit_info` 与 payload generator 共用此 helper，所以 metadata
和传输 dtype 同时修复。BF16/FP64 输入仍按原默认 refit dtype 处理。

新增 FP32/BF16/FP64 三种 dtype 的 metadata/payload 回归，检查量化敏感值
的逐位保留，并覆盖 `dt_bias` 的独立 dtype 约定。作业 `21379230` 使用
原1030镜像，运行原 generator 测试、router metadata 测试、新增3例测试，
再用捕获的真实30层 A_log 验证 metadata、payload 和 GPU copy。
**15项测试全部通过，无失败、无跳过；30层 A_log 全部逐位匹配，误差为0。**
证据见 `results/a-log-fix-tests-21379230.xml` 和
`results/a-log-fix-21379230.json`。早先 `21379127` 在测试启动前因隔离环境
缺少 pytest 退出；重试仅安装小型测试依赖到容器 /tmp，没有同步模型依赖。

本次没有用修复后的源码重跑完整收敛实验。它是已确认精度损失的最小修复，
不是剩余 KL 的完整根因结论。后续宜先做同 token 重复评分负对照，再隔离
prefill/decode GDN 路径和缓存 dtype，保持原训练目标与 checkpoint 不变。
重复评分作业 `21379496` 已提交：相同 token 按原始/原始/FP32/FP32/恢复原始
的顺序评分，每次清空 prefix cache。旧 BF16 refit 仅通过隔离诊断 hook
显式重现，普通训练使用修复后的 FP32 规则。该轮结果尚未产生。

MCore600续跑独立进行，原300结果和 stop 标记保留。旧 AutoModel900
作业21315214/16/17继续保持用户 hold。
