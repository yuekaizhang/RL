# AutoModel checkpoint600 固定 token KL 排查

2026-11-06 09:18 更新：**在同一个引擎、相同权重和token、固定串行条件下，
仅切换GDN prefill后端，完整模型logprob会稳定改变；但换成Triton没有改善
与policy之间的差距。** 这把一个可重复的数值影响定位到了prefill路径选择，
仍不能据此判断哪个实现错误，或将全部gen KL归因于该kernel。

完整作业 **21392111 COMPLETED/exit0，13分13秒**，完成两次真实更新和三次refit，
末尾分析与退出均成功。继续用R9的32序列、26,114个response token及同一
checkpoint600 policy参考；初始1026个HF tensor抽样精确一致。每阶段先预热，
然后按FlashInfer三次→Triton三次→恢复FlashInfer三次执行，始终并发1。
机器证据见 [serial-kernel-summary.json](serial-kernel-summary.json)。

| 同引擎、同权重/token的对照 | mean K3 |
|---|---:|
| FlashInfer三次重复两两比较 | 全部0，逐位一致 |
| Triton三次重复两两比较 | 全部0，逐位一致 |
| 恢复FlashInfer三次重复两两比较 | 全部0，逐位一致 |
| 恢复FlashInfer与切换前FlashInfer | 全部0，逐位一致 |
| FlashInfer与Triton，三次 | 稳定0.00220373 |
| FlashInfer与相同policy | 稳定0.00216808 |
| Triton与相同policy | 稳定0.00224619 |

全部30层×8rank实际切换并恢复，每层Triton均调用144次。decode仍是CUDA，
cache始终关闭、chunk budget2048，SSM FP32/conv BF16/A_log FP32；没有外部
请求混入，906个resident tensor/rank抽样前后逐位不变，三次refit每次1026个
HF tensor在8rank抽样精确匹配。两步train/eval重算均逐位一致，policy单步
漂移K3约0.002314/0.002264。原生step601/602 genKL约0.00214324/0.00186997，
lag0/1、LR1e-6、有效样本32，离线重算与原生值相差约4e-9。训练batch不同，
不能用这些原生值与上一轮直接相减来估计内核效果。

单层重放 **21398471 COMPLETED/exit0，3分20秒**，只用1GPU，没有加载模型或
训练。实际第一层677个token，q/k/v BF16、g/beta FP32、初始state为0；捕获的
`use_qk_l2norm_in_kernel=false`，因此保持已预处理输入，没有额外再归一化。
原FlashInfer/Triton各自输出与最终state重复三次均逐位一致，二者输出彼此的
相对RMS差异为0.231%，最终state为0.286%。见
[gdn-kernel-summary.json](gdn-kernel-summary.json)。

| 同输入与FP64逐token递推的相对RMS误差 | 输出 | 最终state |
|---|---:|---:|
| FlashInfer chunk prefill | 0.1843% | 0.1774% |
| Triton chunk prefill | 0.1964% | 0.2302% |
| FP32 recurrent | 0.1589% | 0.000223% |
| FP64参考仅将输出舍入BF16 | 0.1589% | 不适用 |

FP32 recurrent的输出误差接近BF16输出舍入下限，支持该数学参考与当前输入
约定相容；两种chunk prefill包含额外的计算精度差异。这里没有完整模型输出
或loss，也不能把单层相对RMS与完整模型K3线性相加，尚不能定性为kernel bug。

实际policy primitive对照 **21398928 COMPLETED/exit0，3分39秒**，仍只1GPU。
先在CPU建立FP32/FP64逐token参考，再在原policy环境构造一个未训练的
`CPAwareGatedDeltaNet`仅检查绑定，确认是`fla.ops.gated_delta_rule.chunk`的
`chunk_gated_delta_rule`。不使用组件权重，不加载checkpoint或完整模型。

相同的预处理后q/k/v/g/beta输入逐值核对，按AutoModel源码把q/k扩展为4头。
生成侧FlashInfer、Triton扩头前后输出均逐位不变。policy primitive两种beta
精度下各3次重复的输出和state也均逐位一致。机器结果见
[policy-gdn-summary.json](policy-gdn-summary.json)。

| 相同单层输入、同shape的输出对照 | 相对RMS差异 |
|---|---:|
| vLLM FlashInfer vs policy FLA（共享FP32 beta） | 0.2433% |
| vLLM Triton vs policy FLA（共享FP32 beta） | 0.1478% |
| vLLM FlashInfer vs policy FLA（BF16 beta反事实） | 0.2475% |
| vLLM Triton vs policy FLA（BF16 beta反事实） | 0.2444% |

CPU FP32递推与CPU FP64的输出相对RMS差约1.55e-7，CPU/GPU FP64参考差约
3e-16，支持数学参考的实现及布局相容。policy FLA的最终state契约是K,V，
vLLM参考是V,K；按API转置后比较，没有把直接比较不同布局产生的大差异
误当成bug。policy FLA对FP64参考的输出误差约0.1992%（共享FP32 beta）。

这直接验证了训练/生成环境的GDN primitive即使输入相同仍有数值差异，但它
不是完整GDN层的投影、卷积、门控或归一化对照。BF16 beta是源码动机的独立
反事实，没有捕获policy的原始投影值，不能把该差异全部归因于实际beta精度。
当前没有根据将某一条路径定性为bug，也不采用切内核或关cache作为生产修复。

此前R13：**关闭prefix cache并没有解决重复评分差异，也没有缩小
稳定的串行评分与policy之间的差距。** 因此不把关闭cache当作训练修复。
本轮继续支持“至少一部分误差来自后端计算条件，而非LA1策略滞后”，但尚不能
指认某个GDN kernel、AutoModel算子或vLLM实现为完整根因。

R13 `21390581` 完成600→602两次真实更新和三次refit后，在解释器清理阶段
崩溃（Slurm exit11，12分23秒）；64份train、32份drift及全部refit/评分捕获齐全。
只用1GPU分析任务 `21391605` 恢复末尾分析，**COMPLETED/exit0，2分28秒**，
没有重跑模型、训练或改动原checkpoint。机器结果见
[prefix-cache-summary.json](prefix-cache-summary.json)。

固定使用R9的32条序列、26,114个response token及保存的同version600 policy
logprob。相同token逐个核对，初始refit的1026个HF tensor抽样也精确一致。
实际引擎日志确认旧轮cache开启、新轮关闭，chunk budget均2048；新轮全部
8rank探针确认cache=false、Mamba cache mode=none、SSM FP32/conv BF16，
GDN prefill仍为FlashInfer，A_log FP32，没有开启不支持的batch-invariant模式。

| 相同权重/token的mean K3对照 | 原cache开启（R9） | cache关闭（R13） |
|---|---:|---:|
| 串行1，三组重复两两比较 | 全部0 | 全部0 |
| 并发4，三组重复两两比较 | 0.000702 / 0.001277 / 0.001167 | 0.001177 / 0.000428 / 0.000886 |
| 并发32，三组重复两两比较 | 0.001309 / 0.001309 / 0 | 0 / 0.001775 / 0.001775 |
| 串行评分与相同policy，三次稳定值 | 0.00216220 | 0.00216808 |

cache开关不同的两轮串行评分彼此仍有K3约0.001127差异；由于来自不同引擎
运行及节点，这不能单独归因于cache，更不能据此宣称cache存在错误。并发重复
波动没有消失，串行与policy的整体差距也几乎不变。

R13中评分期间外部请求数始终0，8rank各906个resident tensor抽样前后逐位不变。
三次refit每次1026个HF tensor在8rank抽样精确匹配，30层A_log wire FP32、
误差0。两步train/eval重算均逐位一致，单步policy漂移K3约0.002415/0.002634。
原生step601/602 genKL约0.00220440/0.00189095，离线同batch重算差约4e-9；
lag分别0/1、LR均1e-6、有效样本均32、丢弃及logprob mask均0。两步训练batch
与R9并不相同，不能直接比较原生KL来估计cache或lag的因果效应。

后续 `21392111` 的同引擎串行FlashInfer/Triton/恢复FlashInfer对照已成功完成，
数值结果、恢复检查及后续单层计划见本文开头。
诊断runner把分析移到解释器退出前；仅在训练、显式资源关闭和分析均成功后
结束进程，避开本轮已复现的Ray析构崩溃。未修改生产训练的退出行为。

此前R9（2026-11-06 06:50）：**相同权重、相同 token 下，串行评分三次逐位一致，
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

`21388465` 仍在启动时退出，耗时6分02秒：NeMo-RL的worker已显式传入
`enable_prefix_caching`，将同名键放入`vllm_kwargs`造成Python重复传参。
本轮没有refit、评分或训练更新；之前的配置预检没有覆盖实际worker参数组装，
不能作为完整启动成功的证据。

现将开关改放在源码读取的`policy.generation.vllm_cfg.enable_prefix_caching`，
同时补充`VllmSpecificArgs`的可选bool/None声明，并添加三例类型校验回归。
默认值继续由已有CUDA capability规则决定。新的预检会直接调用生产worker的
参数组装路径：旧kwargs位置必须复现错误，修正后必须到达engine构造入口，
并用真实`AsyncEngineArgs`验证最终配置；在该入口截断，不创建完整模型。

真实worker参数组装在 `21390570` 中通过，并在最终预检 `21390696` 中再次通过：旧位置准确复现重复传参
错误，修正后达到engine构造入口，并通过真实`AsyncEngineArgs`配置校验。
最终配置为prefix cache关闭、TP8、context4096、chunk2048、SSM FP32／conv
BF16、dummy加载，Mamba cache mode为`none`，没有创建engine或加载模型。
见 [worker-kwargs-summary.json](worker-kwargs-summary.json)。

最终预检 **21390696 COMPLETED/exit0，耗时3分27秒**。三例新增类型校验
和两例已有开关测试全部通过：**5 passed、0失败、0跳过**；真实worker参数
组装的错误复现、修正后入口以及实际`AsyncEngineArgs`配置校验也全部通过。
生产配置声明与测试已推送到同一分支，提交 `f93efd64d`。

前两次小预检的测试框架问题已单独处理：仓库全局覆盖改写pytest版本、
输出捕获冲突，以及`--vllm-only`排除了未标记的纯单元测试。它们不算测试
通过；最终预检固定pytest8.4.2、关闭捕获并显式执行五个测试节点。

完整同配置对照 **21390581** 的捕获与恢复分析均已完成，关闭prefix cache后的
数值结果及分析恢复范围见本文开头。MCore600两组首段21388471／21388475仍排队，
原300步checkpoint保留。

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
