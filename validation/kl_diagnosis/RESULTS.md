# AutoModel checkpoint600 固定 token KL 排查

2026-11-06 05:50 更新：**相同权重、相同 token 下，串行评分三次逐位一致，
4并发与32并发时出现明显重复差异。** 因此，生成端差异至少有一部分依赖
请求调度、batch形状或缓存管理条件，不能全部解释为 LA1 策略滞后。
不过，稳定的串行评分与同version600的policy仍有约0.00216的K3差异，
剩余根因尚未完全定位，也不能将并发差异直接定性为某个kernel的bug。

作业 `21384522` COMPLETED/exit0，原1030镜像，EP16 policy + TP8 vLLM，
checkpoint600只读，完成600→602两次真实更新。固定32条序列、26,114个
有效response token，每种并发量预热后重复3次，按交错顺序执行。
机器结果见 [scheduling-summary.json](scheduling-summary.json)。

| 同一组权重和token的对照 | mean K3 |
|---|---:|
| 串行1，三组重复评分两两比较 | **全部0，所有logprob逐位一致** |
| 并发4，三组重复评分两两比较 | 0.00070238 / 0.00127714 / 0.00116656 |
| 并发32，三组重复评分两两比较 | 0.00130916 / 0.00130916 / 0 |
| 串行评分与同version600 policy，三次 | **稳定0.00216220** |
| 串行与并发32，按轮次比较 | 0.00190226 / 0.00181736 / 0.00181736 |

评分前确认后台请求排空；每20ms观察，整个评分期间没有外部rollout混入。
所有8个rank的906个resident tensor抽样前后逐位不变，实际model.eval、
没有活动Dropout，所有30层SSM cache均FP32。三次refit每次1026个HF tensor
在全部8个接收端抽样精确匹配。生产修复已生效：30层`A_log`的wire dtype
均FP32，量化误差为0。这些是抽样检查，并非所有resident映射的穷尽证明。

两步原生训练指标为：step601 genKL0.0020506231/lag0，
step602 genKL0.0020259171/lag1；两步LR均1e-6、有效样本均32。
这两个batch不同，不能相减当作策略滞后的因果效应。
64份train和32份单步漂移数据的补充分析已完成，复用了保存数据：

| 同token对照 | 第一次更新 | 第二次更新 |
|---|---:|---:|
| train forward与更新前eval重算 | **逐位一致，K3=0** | **逐位一致，K3=0** |
| policy单次optimizer更新前后 | 0.00220782 | 0.00282455 |

单步漂移与后端评分差异不能线性相加来预测gen KL。

`21385207` 的 batch 不变性对照**未执行成功**：作业运行5分32秒，
在生成引擎初始化时退出，原1030 vLLM selector明确拒绝该模式与`GDN_ATTN`
组合。没有新的评分、更新或refit结果，不能据此判断该模式改善或恶化KL。
已保留失败日志，并在提交脚本中提前拒绝该不兼容配置。

检查原1030的`SamplingParams`进一步确认：`prompt_logprobs=0`默认设置
`skip_reading_prefix_cache=True`。因此上一轮差异不能解释为直接读取了缓存
中的共享prefix；但开启prefix cache仍会影响GDN状态的管理和分块边界。

关闭prefix cache的 `21385590` 在模型加载前因新增Hydra配置键的写法错误
退出，耗时3分16秒。GDN selector及cache/scheduler预检已通过，但没有评分、
训练更新或refit结果。已将四个新键写进独立`ep16_noprefix.yaml`，并增加
真实Hydra继承、CLI覆盖、MasterConfig和SC配置校验。

恢复预检 `21388461` 使用原1030镜像、1GPU，**COMPLETED/exit0，耗时3分18秒**。
实际通过上述诊断配置校验、GDN selector、prompt-logprob默认行为及cache
配置检查；同时验证MCore两组续跑入口、checkpoint300及optimizer状态发现。
MCore此前两个首段因Megatron同名`examples`包抢占导入而退出，现改为按
确切源文件路径加载，预检已验证两组入口。以上均未加载完整模型或执行训练。
机器证据见 [recovery-preflight-summary.json](recovery-preflight-summary.json)。

新的固定token对照 **21388465** 已接续：关闭prefix cache，复用21384522的
完全相同32条token、policy logprob和checkpoint600；chunk预算2048、SSM/conv
cache实际精度FP32/BF16保持一致。继续三种并发量各三次评分、两次真实更新
及三次refit检查，八个worker运行时配置也必须核对。预检通过后已解除依赖，
当前Priority排队，尚无新KL结果。MCore600两组新首段为21388471／21388475，
同样通过预检门控后排队；仍需在实际恢复后验证第301步及保存的scheduler。

现有`vllm_cfg.fp32_lm_head`实现只支持Nemotron-H，不能直接用于当前Qwen；
AutoModel当前路径实际生成完整logits，并非已启用fused linear CE。

之前 `21379496` 的相同权重、相同token、同一种prefill实现重复评分，
本身就出现约0.0016–0.0019的K3，与该轮lag0 genKL0.00194同量级。
因此第一轮单次A_log A/B约9%下降，**不能视为稳定、可归因的修复效果**。
`A_log`的FP32→BF16精度损失是独立验证过的事实；其对整体训练KL的影响
尚未建立可靠结论。历史机器结果见 [repeat-summary.json](repeat-summary.json)。

实际所有SSM cache都是FP32，conv cache为BF16、`dt_bias`为BF16。
“BF16 SSM cache每步回写导致积累”不适用于当前主运行。
真实第一层544个token的同输入重放中，同一个Triton recurrent kernel的
FP32逐token与FP32连续计算输出及最终状态完全一致；人为使用BF16状态
后输出有约0.55%相对RMS差异。这是单层机制对照，不等于完整CUDA decode
正确性验证，BF16结果也不是主运行根因证据。
完整对照和两次更新在21379496中完成；其最后的辅助重放因缺状态索引exit1，
已仅用1GPU作业 `21384317` 补完，COMPLETED/exit0。
见 [gdn-state-summary.json](gdn-state-summary.json)。

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

本次没有用修复后的源码重跑完整收敛实验。`A_log`修复已通过15项GPU回归，
并在21384522完整两步诊断中验证实际FP32传输；它不是剩余KL的完整根因结论。
最新的串行/并发负对照和下一步固定token计划见本文开头。

MCore600续跑独立进行，两组仍Priority排队、无新300步之后的数据；原300
结果和stop标记保留。旧AutoModel900作业21315214/16/17继续保持用户hold。
