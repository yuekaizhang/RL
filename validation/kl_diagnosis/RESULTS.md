# AutoModel checkpoint600 固定 token KL 排查

2026-11-06 03:07 更新：**同权重、同 token、同一种 prefill 实现，重复评分
本身就出现约0.0016–0.0019的 K3，已与本轮 lag0 的 gen KL 约0.00194同量级。**
因此，下面第一轮单次 A/B 的约9%下降，不能视为稳定、可归因的修复效果。
`A_log` 的 FP32→BF16 精度损失仍是独立验证过的事实，保留FP32的修复有效；
它对整体训练 KL 的影响尚未建立可靠结论。

完整对照及两次更新在 `21379496` 中完成，主要分析结果已保存。
该作业最后因单层重放的辅助 kernel 调用缺少状态索引而 exit1；已仅用
1GPU作业 `21384317` 补完重放，COMPLETED/exit0，未重新训练模型。
机器结果见 [repeat-summary.json](repeat-summary.json) 和
[gdn-state-summary.json](gdn-state-summary.json)。

| 本轮相同24,316个有效 response token 的对照 | mean K3 |
|---|---:|
| 原始生成与policy，同version600，lag0 | 0.00193933 |
| 原始 FlashInfer prefill 连续重复两次 | 0.00169084 |
| FP32 A_log 的 FlashInfer prefill 连续重复两次 | 0.00154652 |
| Triton/FLA prefill 连续重复两次 | 0.00161255 |
| 恢复原始权重后与之前的原始评分比较 | 0.00193177 |

这些重复是在同一权重、同一请求 token 下完成，但实际 batching、请求调度
及 prefix cache 使用可能变化；目前还不能把差异定性为某个 kernel 的固有
不确定性。切换 Triton 后重复差异仍存在，也未单独指认 FlashInfer 为根因。
所有8个rank、30层都实际执行了替代方法，每层36次，之后恢复原方法。

运行时探针确认：**所有 SSM cache 都是 FP32**，conv cache 为 BF16，
`A_log` 为 FP32、`dt_bias` 为 BF16。因此“BF16 SSM cache 每步回写导致积累”
不适用于当前主运行。真实第一层544个 token 的同输入重放中，使用同一个
Triton recurrent kernel，FP32逐token与FP32连续计算的输出和最终状态均
完全一致；人为改为BF16状态后输出有约0.55%相对RMS差异。这个单层测试
不等于完整 CUDA decode 正确性验证，BF16结果也不是主运行的根因证据。

下一轮 `21384522` 已提交，固定修复后FP32 A_log权重与token，预热后分别
用串行、4并发、32并发评分，各重复3次并交错顺序。评分前等待后台请求
排空，期间记录外部请求是否混入；前后抽样核实生成端参数逐位不变。
该对照同时改变请求调度、batch形状和可能的prefix复用，不能直接归因于
GEMM。当前仍待GPU结果，MCore600续跑与旧900步hold保持原安排。

以下保留第一轮诊断的原始结果，便于对照。

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

继续排查时，已在仍排队的同一作业中增加：相同权重、相同 token，仅将
GDN prefill 从官方 FlashInfer 方法切换为官方 Triton/FLA 方法，重复评分并
恢复原方法；同时记录真实 sampling 参数、各层 cache dtype 和实际调用次数。
另捕获一层真实 Q/K/V，在同一 allocation 内重放最多1024个 token，比较
连续FP32计算、逐token FP32状态和逐token BF16状态。后者仅检验单层状态
存储精度的机制，不能直接作为整模型 KL 或 CUDA decode bug 的结论。
这些新增对照目前只通过静态检查，尚待 GPU 实验结果。

MCore600续跑独立进行，原300结果和 stop 标记保留。旧 AutoModel900
作业21315214/16/17继续保持用户 hold。
