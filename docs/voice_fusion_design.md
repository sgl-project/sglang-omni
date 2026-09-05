# Higgs TTS Voice Timbre Fusion — 设计说明

## 目标
给 sglang-omni 的 Higgs TTS 加"音色融合":一次合成可同时条件化 N 个参考音色,按权重在
**解码输出分布层**加权融合(不是 prompt 拼接),得到一个稳定的"中间音色"。

## 机制(横向扩展,非侵入调度内核)
一个融合请求 = N 个 sibling batch 行,每行独立 prefill 出一个参考音色的 KV 上下文。
解码每一步:`modality_head.generate() -> logits_BNV [B,8,1026]` 之后、`batched_step` 之前,
对同组 N 行做**加权概率归约**(同组拿到同一融合分布、同 seed 抽同一帧),N 条上下文锁步演化,
仅 leader 行输出音频。组内共享 `generation_done` 做"同生同灭"屏障。

## 核心组件
| 机制 | 落点 |
|---|---|
| 归约算法 | `fusion.py::fuse_group_logits` / `fuse_group_generation_done`(纯 torch,无 sglang 依赖,可独立单测) |
| 归约钩子 | `model.py` `decode_codebooks_batch` / `decode_codebooks_batch_cg`,在 `batched_step` 前替换 logits |
| 融合注册表 | `fusion.py::FusionRegistry`(线程安全,build 线程写、GPU worker 线程读);`model.py` 的 `set_fusion_group`/`has_any_fusion`/`is_fusion_follower`/`mark_fusion_group_poisoned` 等是对它的薄委托 |
| 请求拆分 | `request_builders.py::build_fusion_sibling_requests`:1 融合 payload → N 条 `HiggsSGLangRequestData`,共享 `fusion_group_id` + 一个具体 seed,leader=第一个 sibling |
| prefill 侧原子准入(尽力,非硬保证) | `omni_scheduler.py::OmniScheduler.get_next_batch_to_run`(override):在调用上游选批之前,先对 `self.waiting_queue`(一个纯 Python list,尚未流入 `ScheduleBatch.prepare_for_extend`)做门控——缺员的组整组暂扣,凑齐且预算内的组整组挪到队首、成员相邻,排在普通请求前面;调用完再把暂扣的请求放回队首。只动 list,不碰张量(见下文"Co-batching") |
| prefill 侧隔离 + 中毒标记 | `model.py::_batch_local_fusion`:prefill 完成后触发首个解码 token 那一步若发现组缺员,隔离在场行(不参与融合)并把该组标记为 `FusionRegistry._poisoned`——因为这里没有 `Req` 句柄,没法自己 abort,而组之后可能"自愈"(所有成员后来都凑齐了),中毒标记是让下面的完整性检查即便凑齐也照样 abort 的唯一办法 |
| 组完整性兜底 | `model_runner.py::_populate_fusion_buffers`(decode CG 路径,真正会遇到"组被拆散"的地方,无论是 KV retract、sibling 还没轮到,还是曾经中毒但现在看起来凑齐的组) |
| 组级联清理 | `omni_scheduler.py::OmniScheduler._cascade_abort_split_fusion_group`(在 `stream_output` 里触发):一个融合成员以 `FINISH_ABORT` 结束时,把组里还注册着的其它成员一起 abort(并向每个被级联的成员发一条 client 可见的 error,leader 也不例外),防止缺席的 sibling 变成永久卡在 `waiting_queue` 里的僵尸请求 |
| 输出去重 | `model_runner.py::_finish_fusion_follower`:follower 的解码帧与 leader 重复,不 append/发音频,但仍要在同一步被标记 finished,否则组会"拆分" |
| 异步 lookahead 规避 | `model_runner.py::HiggsTTSModelRunner.lookahead_eligible`:只要有任何融合请求注册,强制走同步解码路径,避免这个仓库自建的 one-step-lookahead 把 launch 阶段设置的 FINISH_ABORT 误判成"上一步的过期行"而丢弃 |

## 核心算法

`fuse_group_logits`(见 `fusion.py` 完整 docstring)对同组 N 行做**加权 log-linear 归约**
(product-of-experts / 加权几何平均,不是概率空间的算术平均)喂给标准 sampler;单行组(非
融合请求)原样返回**未经任何处理的原始 logits**——不预先除以温度。返回值额外带一个
`is_grouped_B` 掩码,调用方必须据此决定每行喂给 sampler 的温度——只有真正被分组的行才在
归约时把温度折叠进去、随后以 `temperature=1` 采样;单行组必须保持自己的真实温度并且只被
除一次,否则会有两个后果:(a) 破坏 sampler 的 greedy 短路(`temperature<=阈值` 时不经过
`multinomial`,直接 `argmax`);(b) 即便没有触发 greedy 短路,普通(非融合)请求也会被
采样两次温度(`T²`)而不是一次。这一契约由 `test_voice_fusion.py` 的两组回归测试共同守护:
greedy 场景验证 RNG 消耗量而非采样概率巧合
(`test_singleton_greedy_sampling_matches_baseline_...`),非 greedy 场景直接比较采样分布
是否与不融合时的 baseline 一致
(`test_singleton_nongreedy_sampling_is_not_scaled_by_temperature_twice`)。

**为什么是 log-linear 归约,不是概率空间算术平均(真实 bug,已修复)**:v1 实现在概率空间
做加权算术平均——`blended = Σ w_i · softmax(logits_i/T)`,再取 log 喂给 sampler。这个版本
上线后被实测证实有一个严重问题:两个明显不同的参考音色、权重打平(0.5/0.5)时,连续 8 次
独立生成(不同随机种子)的结果按音高中位数聚成两簇——4 次贴近参考 A 单独克隆的音高、4 次
贴近参考 B,没有一次落在两者中间;而权重悬殊(0.9/0.1)时 8 次全部稳定贴近权重更大的那一方,
完全没有跳变。这排除了"权重没被正确应用"这个假设(否则悬殊权重也该乱跳),精确定位到:
**只有势均力敌时才会两极分化**。机制上的原因是:两个差异较大的音色各自的 softmax 分布,
算术平均出来是双峰的——几乎没有落在"两者都说得通"的中间 token 上的概率质量,所以每一步
从这个混合分布采样,采到的要么是偏 A 的 token,要么是偏 B 的 token,从来不是中间的。而
自回归解码里,一旦前几步(建立音高/音区的关键步)偶然采到了偏向某一方,后续步里那一方自己
的条件概率会因为"上文更像自己会说的话"而更自信、在混合里占比越滚越大,雪崩式锁死成单一
音色——这不是低概率的采样噪声,是双峰分布结构性地不给"中间地带"分配质量。

修法:把归约挪到 **log 空间**(logits 层面加权求和:`blended_logits = Σ w_i · (logits_i/T)`),
而不是概率空间。`softmax(Σ w_i · logits_i/T)` 在数学上*精确*正比于
`Π softmax(logits_i/T)^{w_i}`(加权几何平均),两者只差一个不随 vocab 变化的可加常数,
`softmax`/`argmax`/top-k/top-p 都不受这个常数影响——所以直接对温度缩放后的 logits 做加权
求和,和"先分别 softmax、加权几何平均、再取 log"完全等价,而且更简单(不需要 `softmax`
/`log` 往返,也不需要给融合概率加 `_LOG_FLOOR`,原来的 `_LOG_FLOOR` 只用来保护
`group_weight_sum` 的除零)。几何平均(product-of-experts / AND)不像算术平均(mixture-
of-experts / OR)那样是双峰的——它把概率质量集中在"两个参考音色都觉得说得通"的 token 上,
这才是"融合音色的一帧"应该有的样子。这个修法已经用真实推理复现验证(见"仍需真实引擎验证的
项"),单测已更新(`test_two_member_equal_weight_is_geometric_mean` 替代旧的
`test_two_member_equal_weight_is_prob_average`,并新增
`test_three_member_group_is_weighted_geometric_mean` 覆盖 N>2 的一般情形)。

这个修法和"CFG 外推留作 follow-up"(见下文已知限制)是同一个方向但不是同一件事:CFG 本身是
"把已有分布往条件方向外推、锐化"(`z_out = z_uncond + s·(z_cond - z_uncond)`,权重可以在单纯形
之外),解决的是"生成结果不够贴近参考音色"这个问题——直接套在旧的概率空间算术平均上,只会让
双峰分布的两个峰都更尖锐,加速锁死,并不能解决势均力敌时的两极分化。但 log-linear 归约把 CFG
真正需要的底层运算(log 空间线性组合)已经就位了——后续如果要做 CFG 外推,是在现在这个
log-linear 骨架上加一个无条件(无参考音色)的额外 sibling 行、给它负权重,而不是另起一套机制。

**log-linear 归约是按精度加权,不是按权重加权(第二个真实 bug,已修复)**:上面这个修法上线后
用真实推理复现测试,结果是把"随机抛硬币锁死一个"变成了"绝大多数时候锁死同一个"——8 次权重
0.5/0.5 的独立生成里 7 次贴近同一个参考音色,只有 1 次贴近另一个,不再是 50/50 的巧合分布,
但也远不是稳定居中的融合。排查发现:换成响度对齐后的参考音频(把两条参考音频的 RMS 拉到同一
水平)重跑,偏向不但没消失反而更彻底(8/8),排除了"测试用的两条参考音频本身响度不对等"这个
最简单的解释。真正原因是 log-linear 归约(几何平均 / product-of-experts)的一个数学性质:
它是**按精度(分布的尖锐程度,等价于熵的倒数)加权,不是按名义权重加权**——两个方差不同的
高斯分布做加权几何平均,结果的均值是"精度加权平均"(`(w/σ_A²·μ_A + (1-w)/σ_B²·μ_B) /
(w/σ_A² + (1-w)/σ_B²)`),不是简单的 `w·μ_A + (1-w)·μ_B`;而且"权重"和"分布尖锐程度"在这个
归约里数学上是同一个杠杆——`softmax(0.5·z_A + 0.5·(1.5·z_B))` 和
`softmax(0.5·z_A + 0.75·z_B)` 是完全相同的分布(把 B 的 logits 整体放大 1.5 倍,等效于把 B
的名义权重从 0.5 提到 0.75)。如果两个参考音色本身在这个模型里的逐步预测天然有置信度差异
(不管是因为音色本身特点、录音质量,还是别的原因),名义权重打平并不代表感知上的话语权也打平。

修法:在 `fuse_group_logits` 做加权求和**之前**,先把组内每个成员的 logits 按
`_solve_entropy_matching_gamma` 求出的每行(按 codebook 单独求)缩放系数 gamma 校正到同一个
"置信度基准"(组内按权重加权平均的熵),再做加权求和——这样权重才重新变回真正起决定作用的
因素。gamma 的求解用 5 步定长牛顿迭代(`dH/dgamma = -gamma·Var_p(logits)`,标准指数族分布对
数熵的解析导数,推导:`H(gamma) = -gamma·E_p[z] + logZ(gamma)`,`dlogZ/dgamma = E_p[z]`,
两次求导消去 `E_p[z]` 项即得),定长迭代次数、定长张量形状,没有 host 端收敛判断分支,CUDA
graph 安全;gamma 钳制在 `[1/3, 3]` 防止极端不对称的输入把缩放系数推到数值不稳定的范围。
迭代次数从 3 步调到 5 步:实测中等熵区间(2.5-6.5 nat)3 步就已经是浮点精度下的精确解,但真实
解码时一个音色的分布常年处在高置信度(低熵,0.2-1.8 nat)区间,3 步在那个区间还留有
0.1-0.35 nat 的残差,5 步能把残差收到 1e-4~1e-5 nat。单行组(非融合请求)完全不受影响——仍然
走最后那个 `torch.where` 直接返回原始 logits,校正后的值只在真正被分组的行上才会被使用。
单测见 `test_voice_fusion.py` 里 `test_entropy_matching_*` 一组:已匹配熵的两行是近似恒等
变换、明显更平坦的一方会被上调、组间互不影响、极端不对称时钳制生效。

**熵匹配之后仍然残留的问题:AR 滞后(第三个真实 bug,已实现修法,真实引擎验证进行中)**:
熵匹配上线后重新实测(同样的 A/B 对照协议),0.5/0.5 权重的 8 次独立生成从"绝大多数锁死同一个"
变成了更分散的分布——出现了真正居中的样本,也仍有靠近两端的样本,不是稳定居中。原因:熵匹配
只纠正**每一步的边际分布**,纠正不了自回归解码里的滞后放大——两个 sibling 共享同一段已经
采样出来的上下文,一旦前几步(建立音高/音区的关键步)偶然采到了偏向某一方,那一方自己的条件
分布会因为"上文更像自己会说的话"而更自信、在混合里占比越滚越大,这个反馈发生在跨步骤的轨迹
层面,不是熵匹配能触及的单步边际分布层面。

修法:`FusionRegistry` 给每个融合请求维护一个跨解码步持续存在的"轨迹反馈"微调量 `delta`
(log 尺度:`有效权重 = 名义权重 * exp(delta)`,不是名义权重本身被改写,只是这一步实际喂给
归约的值被调整)。每个解码步采样出这一步的共享帧之后,用
`mean_log_likelihood_of_sampled_frame` 算出每个成员自己(未经融合的)分布对这一帧的
似然,和组内按**名义**权重(不是已经被 delta 调整过的有效权重——目标是"贴近调用方要的比例",
不是"贴近这一步已经倒向的比例")加权平均的似然比较,按 `delta ← clip(delta - λ·(ℓ_i - 组内
加权平均 ℓ), ±ln2)` 更新——哪个成员这一步"赢得"更多(自己的似然比组内平均高很多),它的
delta 就被压低,下一步有效权重跟着降低;反之则被抬高。这是一个积分反馈控制器,直接消掉滚雪球
效应,而不是像熵匹配那样只治标(削弱雪球的启动速度)。

实现上分两条路径:eager 解码路径(`HiggsTTSModel._update_fusion_deltas`)在同一次
`decode_codebooks_batch` 调用内直接同步算完更新(这条路径本来就有一次 D2H,不受 CUDA graph
约束);CG 解码路径(生产环境走的路径)拆成两半——`decode_codebooks_batch_cg` 内部无条件写入
一个新增的 shadow buffer `_cg_fusion_ell`(每行自己对这一步采样帧的似然,定长张量操作,无
host 分支,对非融合请求是零成本占位),`HiggsTTSModelRunner` 只在真的有融合流量时才把这个
buffer D2H 回来(`_collect_step_outputs_cg`),再在纯 Python 的 `_decode_collect_host` 收尾
循环里做实际的 delta 更新(`HiggsTTSModelRunner._update_fusion_deltas`)——这一步会跳过本步
已经被 `_populate_fusion_buffers` 因为拆散/中毒而隔离并标记 `FINISH_ABORT` 的行(它们这一步
是按独立单例解码的,不是真的融合意见,不该污染正在被拆掉的组的 delta)。`delta` 本身存在
`FusionRegistry`(host 端 Python dict),不是 CG buffer——因为它必须跨步持续存在,不能像
`_cg_fusion_group`/`_cg_fusion_weight` 那样每步都被重置成默认值。`mean_log_likelihood_of_
sampled_frame` 对已经 `generation_done`(会解出 `STOP_CODE=-1`)的行做了 `clamp_min(0)`——
不 clamp 会在 `gather` 时越界,CUDA 下是致命的 device-side assert;clamp 之后的值本身是
"未使用的占位",因为组内任一成员 done 就代表整组 done,这行的观测值不会再被任何存活的组消费。

积分增益 `λ` 通过环境变量 `HIGGS_FUSION_DELTA_LAMBDA`(默认 0.1)配置,专门为了能在真实引擎上
不用每次都重新发布镜像就扫一遍候选值(0.05/0.1/0.2)。单测见 `test_voice_fusion.py` 的
`test_registry_delta_*`/`test_mean_log_likelihood_*` 和
`test_voice_fusion_pipeline.py` 的 `test_update_fusion_deltas_*`(后者依赖 sglang,本机
Windows 环境跑不了,只能 `py_compile` 语法检查,真实正确性靠单测里手工验证过的算例 + 独立
review)。这是目前为止针对这个 bug 尝试的第三层修法,如果这一层还不能收敛,已知的止损点是:
接受现状(不再是随机锁死单一音色,只是权重-结果映射还不够精确),把残留局限性写清楚,不再往
输出分布层面继续加码——下一步升级会是条件层面的融合(prompt/embedding 插值,或者真正的 CFG
外推),那是另一个项目,不是这个 bug 的延伸。

**第四个真实 bug:观测量本身在原始 logits 上有系统性偏差(已定位、已修复、真实引擎复现验证
已完成,结论见下方"结案"段落)**:轨迹反馈控制器第一次真实引擎验证时,结果趋势是错的——λ 从 0 加到 0.1 再到 0.2,
0.5/0.5 的 8 次独立生成不是越来越居中,而是越来越一致地偏向同一个音色(B):基线(仅熵匹配)
2/8 A + 1/8 居中 + 5/8 B,λ=0.1 时 2/8 A + 0/8 居中 + 6/8 B,λ=0.2 时 0/8 A + 0/8 居中 +
8/8 B——增益越大、越偏,和一个正确的负反馈积分控制器应有的"增益越大越快收敛到居中点"完全相反,
这个方向性异常本身就说明控制器有系统性偏差,不是简单调参能解决的。

根因(定位方式:先复核 leader/anchor 索引、CG/eager 两条路径的行序是否有错位、是否有单步重复
更新——逐一排除,均未发现;再对更新公式本身做离线闭环仿真复现异常,确认是公式层面的系统性偏差,
不是接线 bug):`ell_i - 组内加权平均 ℓ` 这个观测量,如果直接喂**未经熵匹配的原始** per-member
logits,在期望意义下等于 `-H(q) - KL(q‖p_i)`(`q` 是这一步实际采样用的融合分布)——两个
`-H(q)` 项在做差时抵消,于是这个控制器实际伺服的是"每个成员相对 `q` 的原始空间 KL 散度相等",
而不是"名义权重比例相等"。两个真实的、不同的参考音色,天然就有 sharpness(置信度/logits 幅度)
差异——这正是熵匹配专门为**融合本身**去修正的那个因素,但这里的观测量绕开了熵匹配,直接测的是
原始 logits。sharper 的一方,KL 随偏移量增长得更快(方差小),换句话说"KL 相等"这个不动点会
明显偏向 sharper 的那一方,不是 50/50 的名义混合点——控制器几步之内(λ 越大越快)就把 delta
积分到这个偏离的不动点,叠加自回归滚雪球效应,就固化成"总是同一个音色赢,且增益越大赢得越彻底"。

用一个理想化的双峰-注意力仿真(`fuse_group_logits` + `mean_log_likelihood_of_sampled_frame`
的真实实现,人工构造 sharp/flat 两个 sibling,反复采样-更新几十步)可以稳定复现这个异常:
在原始 logits 上观测,`B_sharp` 配置(B 更 sharp)λ=0.05/0.1/0.2 分别收敛到 75%/77%/68% 的
`near B`,居中样本几乎为零;`A_sharp` 配置对称地倒向 A。换成熵匹配后的 `matched_logits`
(即 `fuse_group_logits` 融合求和**之前**那一步、每个成员自己的、按组内加权平均熵重新缩放过的
分布——注意不是组内融合后的共识分布,那样会变成"测的是这个成员多认同共识",而不是它自己独立
的意见)做观测,同样的 sharp/flat 不对称配置下,λ=0.05/0.1/0.2 全部保持 100% 居中,和对称配置
(`sym`)的行为一致——sharpness 不对称对居中结果的影响被完全消除。仿真脚本同时验证了一个
"看似更直接"的替代修法——用每个成员自己的熵去归一化观测量(`ℓ_i + H(p_i)`)——反而更糟:同样
的 sharp/flat 配置下这个方案让 sharp 成员几乎 100% 锁死获胜,比不修复更极端,原因是它只补偿了
"sharpness 带来的置信度加成"这一项,却放大了 KL 尾部随偏移量增长更快这个二次项的影响。

修法:`mean_log_likelihood_of_sampled_frame` 的调用方(`HiggsTTSModel` 的 eager/CG 两条
`_update_fusion_deltas` 路径)统一改喂 `fuse_group_logits` 新增的第三个返回值
`matched_logits`(温度已经折算、且已经熵匹配缩放,所以温度参数传全 1 的 tensor,不能再除一次),
不再喂真正的原始 logits——`fuse_group_logits` 的签名从 `(logits_out, is_grouped_B)` 变成
`(logits_out, is_grouped_B, matched_logits_BNV)`。CG 路径因为 `fuse_group_logits` 本来就在
`decode_codebooks_batch_cg` 内被调用一次,这个改动实际上更简单了(不需要再单独在调用前多存一份
"原始 logits" 变量,直接复用 `fuse_group_logits` 已经算好的那份)。单测见
`test_delta_observation_on_raw_logits_is_biased_toward_the_sharper_member`(复现异常本身)和
`test_delta_observation_on_matched_logits_is_not_biased_by_sharpness`(验证修法消除了这个
偏差)。

**结案:真实引擎复现验证结果 + 止损决定(第三层修法到此为止,不再继续加码)**。用同样的
A/B 对照协议在真实 GPU 上重新跑了 λ=0.1(默认值)和 λ=0.2 两组、各 8 次独立生成:

- λ=0.1:206.5, 229.1, 215.6, 235.8, 121.7, 109.4, 190.4, 109.4 Hz
- λ=0.2:202.9, 225.8, 106.9, 246.9, 218.7, 231.1, 217.5, 237.8 Hz

(参考锚点:单独克隆 A ≈83Hz,单独克隆 B ≈221Hz。)第四个 bug 本身确认修复:两组都不再出现
"锁死同一个音色、且增益越大越彻底"的病态模式——λ=0.1 里有 3/8 落在偏向 A 的一侧,λ=0.2 里也
有 1/8 明显靠近 A,不再是 λ=0.2 时 8/8 全部倒向 B 的那种全锁死。但控制器**没有**达到"稳定
居中"这个原本预注册的成功标准(≥5/8 落在中间地带)——两组结果仍然是两极分化的形状(样本要么
靠近 A 那一侧、要么靠近 B 那一侧),只是不再被一个 bug 系统性地推向固定的一侧。

这不是"λ 还不够大"能解决的,是这个机制本身的结构性天花板(经与 Fable 确认):(1)`delta`
钳制在 `±ln2`,两个成员都打到钳制值时,有效权重比也只能到 `exp(ln2)/exp(-ln2) = 4:1`(名义
0.5/0.5 最多变成有效 0.8/0.2)——而这个仓库自己最早的实测(见"核心算法"一节)已经确认:
明显偏斜的权重(0.9/0.1)是稳定地让偏向的那个音色赢,不会凭空生成"中间"帧,换句话说控制器
这个执行器的整个可用范围,本来就只活在"换一个赢家"的区间里,不在"生成真正中间音色"的区间里;
(2)在真实解码常年所处的高置信度(低熵)区间,一次积分更新的量级(λ·ell 差距)往往在几步之内
就把 delta 推到钳制值——λ 再翻倍也只是把"打到钳制"提前几步,不会改变最终打到钳制这件事本身;
(3)一旦上文已经锁定某个音区,两个 sibling 观测到的都是"对同一个已经确立的上文的延续"的
似然,这个观测量本身的差距会随着锁定而收窄,恰好在控制器最需要发力的时候,它能感知到的误差
信号反而变弱了。这三条合起来指向同一个结论:这一层修法能做的、且已经做到的,是消除一个真实的
方向性 bug(不再随机联合几率地系统性偏向固定一侧),而不是消除 AR 自回归解码本身"前几步一旦
定音区就很难回头"这个更深层的现象——后者不是这一层(输出分布层面的事后加权修正)能够触及的,
触及需要条件层面的融合(prompt/embedding 插值,或者真正的 CFG 外推),这是另一个项目。

按照这个 bug 从一开始就写好的止损标准(见上一段"这是目前为止针对这个 bug 尝试的第三层修法"处
的原话),现在正式停在这里:保留这个修复后的控制器(默认 `HIGGS_FUSION_DELTA_LAMBDA=0.1`,
因为它已经是正确定向的、在 n=8 的实测里没有表现出害处,对偏向 A 一侧的样本有一个微弱但方向
正确的牵引——基线里偏 A 的样本落在 0.20-0.27,修复后 λ=0.1 偏 A 的样本落在 0.28-0.39,更靠近
居中一点),不再在这个输出分布层面的机制上继续加码。已知局限:0.5/0.5 的短句融合,不同随机
种子之间仍然是两极分化的(样本要么明显偏向某一个参考音色,要么另一个),不是稳定居中在两者
之间——这是当前(第三层)修法状态下的真实上限,不是尚待调参解决的问题。

`fuse_group_generation_done` 做"组内任一成员 done ⇒ 全部 done"的屏障,让共享 seed 的
sibling 行在同一步终止,不会有的先跑完、有的还在解码的错位。

## Co-batching:prefill 侧原子准入门控 + decode 侧隔离/级联 abort 兜底

"同批锁步"这件事,理想情况下希望调度器保证 N 个 sibling 总是一起进 prefill、一起进
decode。这里**曾经**试图在 `OmniScheduler.get_next_batch_to_run` 里、在上游已经选完
batch **之后**强制这一点:若某融合组只有部分成员在这批里,就把这些成员从 `batch.reqs`
里摘出退回 `waiting_queue`。**这个机制已经被移除,因为它是错的**:上游
`get_new_batch_prefill` 返回 batch 之前,已经调用过 `ScheduleBatch.prepare_for_extend()`,
把整批请求的 `input_ids`/`seq_lens`/`out_cache_loc` 等张量按*原始*(未摘除前的)`reqs`
顺序拍平好了。事后再摘 `batch.reqs` 会让这些张量与摘除后的 `reqs` 列表长度对不上,搞坏的
不只是被摘除的 sibling,是**这一整批**请求(含无关的普通请求)。上游自己的 `filter_batch`
是只用于 decode batch 的工具,从不 touch extend 张量——没有支持的方式能在
`prepare_for_extend` 之后收缩一个已经组装好的 prefill batch。

现在的原子准入改为在上游选批**之前**做门控,而不是事后修剪结果——`self.waiting_queue`
在这一步还只是一个纯 Python list,尚未流入 `ScheduleBatch`/`prepare_for_extend`,对它做
过滤和重排不涉及任何张量,不会有上面那种腐化风险:

**Prefill 侧(门控,尽力而为,非硬证明)**:`get_next_batch_to_run` 在调用
`_Upstream.get_next_batch_to_run` 之前,先跑
`_reorder_queue_for_atomic_fusion_admission`(整个方法持有 `self._request_admission_lock`——
和 `abort()` 自己改 `self.waiting_queue` 用的是同一把锁,因为 abort 可能从另一个线程——Stage
自己的事件循环,不是这个调度器的 tick 线程——并发跑进来;这把锁是 `threading.RLock()`,可重入,
所以下面第 5 步里 give-up 路径调用 `self.abort()`——它自己也会 `with self._request_admission_lock`
——不会自锁死):

1. 扫描 `self.waiting_queue`,按 `_fusion_group_members` 把请求分成"某融合组的成员"和
   "普通请求"。
2. 一个融合组若**不是**全部成员当前都在 `waiting_queue` 里(有的还没 build 完、有的在别处
   跑着、有的刚被 retract 还没归队),这一组当前在场的成员**整组暂扣**——不放进这次要交给
   upstream 的队列;这正是"部分成员被送进 prefill"这件事本身,不该发生。这种"缺员"不计入下面
   第 5 步的放弃计数——它总会自己收敛(build 迟早完成,或者 decode 侧兜底迟早把它级联 abort
   掉),不需要一个放弃机制。
3. 若这一刻有一个 chunked prefill 请求正在处理中(`self.chunked_req is not None`),这个 tick
   **所有**融合组一律暂扣——chunked 请求已经吃掉了这个 tick 一部分 chunk/input token 预算,
   而这个数字从外面读不到,与其按一个已经被吃掉一部分、读不出真实剩余值的预算去估算,不如整体
   保守跳过这个 tick。
4. 一个全员在场、且没有 chunked 请求在跑的组,先按请求数(而不是 token 数)上限检查——
   upstream 的准入循环不仅在 token 预算耗尽时停止,一旦
   `len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs)`(通常由
   `max_running_requests` 决定)也会整体停止,这和 token 预算是两个独立的维度,只查 token
   预算不够;超过请求数上限的组直接暂扣。这里的 `running_bs` 和下面 token 预算用的是**同一份**
   `_prefill_in_flight_reqs()`(见下)算出来的在途请求数,而不是单独去读
   `len(self.running_batch.reqs)`——原因和 token 预算完全一样:upstream 把上一个 tick 的
   batch 折进 `running_batch`这件事,发生在这个门控运行之后、真正的准入循环开始之前,门控这一刻
   看到的 `running_batch` 可能还没算上刚结束的那一批,如果只用它算请求数上限,会在 prefill 爆发
   后的那一个 tick 里把这个上限算大,放行一个实际会被请求数上限拦腰截断的组。
   再用 `_estimate_available_prefill_tokens`(镜像 `PrefillAdder.rem_total_tokens`/
   `budget_state` 的主项:可用 KV + 可驱逐 tree cache,减去 `_prefill_in_flight_reqs()`——
   `self.running_batch.reqs` 并上尚未被 upstream 折入的 `self.cur_batch`/`self.last_batch`
   (按 rid 去重)——预留的 `max_new_tokens` 上界,再用 `chunked_prefill_size`/
   `max_prefill_tokens` 分别封顶)和
   `_fusion_group_prefill_cost`(镜像 `add_one_req` 的主项:
   `len(origin_input_ids) + max_new_tokens + page_size`,`origin_input_ids` 是刻意偏保守的
   `extend_input_len` 近似值——如果 sglang 自己的 radix KV cache 真的给某个 sibling 命中了
   前缀、让它的真实 `extend_input_len` 比这个估计小,这只会让这一项的估算偏大,方向仍然安全,
   只会让门控更容易判断"装不下"而不是更容易误判"装得下")判断这一 tick 估计的空闲预算是否装
   得下整组;装得下就把这一组整体挪到 `waiting_queue` 最前面、成员紧邻排列、排在所有普通请求
   之前(多组都装得下时,按扫描到的顺序依次从同一份预算/名额里扣,像多个背包物品顺序装箱一样,
   不会让后一组的估计撞车);装不下就和"缺员"的组一样被整组暂扣,等下一个 tick(可能因为已运行
   请求推进了 decode 释放出更多 KV,或者 chunked 请求跑完了)再试一次——这种"预算/名额不够"的
   暂扣**会**计入第 5 步的放弃计数。
5. 一个组因为预算/名额不够被连续暂扣满 `_MAX_FUSION_WITHHOLD_TICKS`(200)个 tick 仍然没有
   被放行,就不再无限期暂扣下去——`_advance_withhold_ticks_and_give_up` 直接放弃它:对组里每个
   rid 都发一条 client 可见的 error(和 `_cascade_abort_split_fusion_group` 一样,对 follower
   的合成 rid 发也无妨,反正没人订阅那个 routing key),然后 `abort()` 掉组里任意一个成员(会
   级联清掉整组)。给出的 rid 集合会从这次暂扣列表里剔除,不会在下面第 6 步被放回队列。没有这个
   放弃路径,一个门控自己估算"永远装不下"的组会被无限期暂扣,客户端连一个最终的 abort/error 都
   等不到——这本身就是这个门控自己新引入的一种活性倒退,必须堵上。
6. 暂扣的(未被放弃的)请求在 upstream 调用返回后(`finally` 块里、同样持锁)立即放回
   `waiting_queue` 最前面。**放回之前会先剔除掉这段时间内被 `_aborted_request_ids` 标记过的
   rid**:一个正处于暂扣状态的请求本来就不在 `self.waiting_queue` 里,如果这时候另一个线程对
   它(或组里任意成员)调用了 `abort()`——不管是客户端主动 cancel,还是第 5 步自己的放弃路径——
   `abort()` 自己那份"从 `waiting_queue` 里摘除"的清理逻辑根本找不到它(它已经不在队列里了),
   但它确实已经被标记为 aborted、组注册表也被清掉了;如果放回时不做这个检查,这个已经被判了死刑
   的请求会被原样复活成一个不再属于任何融合组的普通请求,之后被正常 admit、解码,把已经废弃的
   请求当成正常任务跑到底。

**为什么"整组挪到队首"是必要的,不只是"全部在场"就够**:upstream 的准入循环
(`Scheduler._get_new_batch_prefill_raw` 内的 `for req in self.waiting_queue: ...
if res != CONTINUE: break`)是按队列顺序逐个尝试、一旦遇到装不下的请求就整体停止——不是
"跳过装不下的、继续找后面能装的"。如果一组的成员虽然都在场,但中间被普通请求隔开,某个插在
中间的无关请求恰好把预算耗尽,组内排在它后面的成员这一 tick 就轮不到,组照样被拆散,即使
"预算总量看起来够整组用"这个判断本身没错。把每个装得下的组整体挪到队首、成员相邻,能保证没有
无关请求能插进同一组的两个成员之间抢预算,这是让"总预算够"这个估计真正转化为"这组一定一起被
尝试"的关键一步,不能省略。这个前提依赖部署用的是默认的 FCFS 调度策略(`calc_priority` 对 FCFS
是纯 no-op,不会在门控之后再打乱队列顺序)——如果哪天切到 `lpm`/`lof`/`random` 之类会重排
`waiting_queue` 的策略,这里的"整组相邻"假设会被 upstream 自己的重排破坏,目前代码和文档都没有
去强制固定/校验 `schedule_policy`,这是一个隐式依赖,没有做防御性检查。

**这个门控是保守估计,不是精确复刻,也不是数学证明**:`_estimate_available_prefill_tokens`
故意只镜像 `PrefillAdder` 预算核算里"可用 KV/tree cache 减去在途请求预留、再按
chunked-prefill/整段 input token 上限封顶"这几项主项,不管 SWA/dllm/优先级抢占这些 Higgs TTS
用不到的分支,也不去精确复刻 `new_token_ratio` 折算——chunked prefill 本身**确实在用**
(`engine_builder.py` 设了 `chunked_prefill_size=8192`),不是"用不到",已经作为一个独立维度
折进了预算封顶里,而不是被忽略。本机没有真实引擎能拿来验证一份"精确复刻"是否真的处处一致,与其
做一个自信但可能在某个分支上算错、从而**放行本不该放行的组**的精确版本,不如做一个"宁可低估、
多等一个 tick"的保守版本——低估的唯一后果是这一 tick 白等,从不会让一个真的装不下的组被误判成
装得下。即便如此,这仍然是概率性的缓解,不是形式化证明"绝不会拆分":一个真实引擎里我们没建模到
的预算维度(比如 KV 压力下 retract 造成的组重新入队后的实际扣费、`page_size > 1` 时的对齐开销)
理论上仍可能让这里估计"装得下"的组被 upstream 实际只收了一部分——这也是下面 decode 侧兜底继续
保留、而不是被这个新门控取代的原因。

**Decode 侧(仍然保留,作为兜底,不是主防线,但仍是唯一的"正确性下限")**:
1. `HiggsTTSModelRunner._populate_fusion_buffers`(decode CG 路径,`OmniScheduler.
   _retract_running_requests` 只作用于 running batch,所以只有这条路径会真的遇到"组被
   retract 拆散",或"sibling 还没轮到进 decode"两种情况)——若某组在本 step 缺员,只隔离该
   组的在场行:降级为独立单例(不参与融合,避免用不完整分布产出错误音频),把它们的
   `req.finished_reason` 设为 `FINISH_ABORT()`。同批其它未受损的行(含其它融合组、普通请求)
   不受影响。
2. `OmniScheduler.stream_output` 里的 `_cascade_abort_split_fusion_group`:上面第 1 步
   abort 的是"在场"的行;组里"缺席"的那个 sibling(比如被 KV 压力 retract、正躺在
   `waiting_queue` 里等下一轮)如果没人管,会在未来某个 tick 自己重新 prefill、自己进
   decode——这时组里只剩它一个,`expected_fusion_group_size` 要么已经因为同伴清场而缩到 1
   (于是它会被当成误打误撞的单例、悄悄产出未融合的音频),要么组注册表还没清干净、它又会
   被判定"缺员"再 abort 一轮——都不是我们想要的。修法:`stream_output` 处理一个 fusion 成员
   的 finish 时,如果它的 `finished_reason` 恰好是 `FINISH_ABORT`(区别于正常同步完成用的
   `FINISH_MATCHED_TOKEN`——组正常结束时全体成员总是在同一 decode step 一起触发屏障、一起
   出现在同一次 `stream_output` 调用里,不会有"其它成员还没完成"的情况),就把组里其它仍
   注册着的成员一起 `abort()` 掉。`abort()` 本身既能从 `waiting_queue` 摘除、也能处理正在
   跑的行,是这个仓库里唯一验证过、安全处理"请求可能在队列也可能在运行"两种状态的通用清理路径。

prefill 路径(`model.py::_batch_local_fusion`,prefill 完成后触发首个解码 token 的那一步)
**不再**保留整体 `RuntimeError`。移除组原子准入之前,这里的 raise 被认为"理论上不可达";
但原子准入已经不存在,sibling 的 prefill 落在不同批次是正常情况,再对整批硬 raise 会把这批
里所有无关请求一起炸掉——正是 BLOCKING-3 已经在 decode 侧修掉的那种"殃及无关请求"模式,不该
在 prefill 侧重新引入。现在的做法是隔离而非 raise:把这批里在场的成员降级为独立单例(不
参与融合)、打日志、并把这个 `group_id` 标记为 **poisoned**(`FusionRegistry.mark_poisoned`)。

**为什么需要"中毒"标记,而不只是隔离**:一个组在 prefill 侧被拆散后,可能会"自愈"——比如
sibling A 这一步单独 prefill(被隔离,采了一帧未融合的 codes),sibling B 下一步也单独
prefill(同样被隔离),再下一步两个都进了同一个 decode batch,这时候单看"在场人数是否等于
预期人数"会发现 2/2、完全正常,`_populate_fusion_buffers` 会误以为可以放心恢复融合——但
A 和 B 的 KV 上下文在各自被隔离的那一步已经各自采样了不同的、未融合的第一帧,从那一刻起就
永久错位了,绝不是"看起来凑齐了就可以继续"。中毒标记解决了这个问题:`_populate_fusion_
buffers` 现在即使发现人数凑齐,只要这个组曾经中毒,也会照样隔离+abort,不会被"看起来正常"
骗过去。

**仍然存在、这次没有补上的缺口**:`_batch_local_fusion` 只有 `HiggsGenParams`、没有
`Req` 句柄,没法自己把在场行标记为 abort——它能做的只是隔离+中毒,把"真正 abort"这件事
留给下一个碰到这个组的 decode step。也就是说,从"组被拆散、隔离生效"到"下一个 decode step
检测到中毒并真正 abort"之间,**这一步(以及自愈过程中每一次单独被隔离的那一步)已经采样出
的、未融合的一帧 codes 是真实输出,可能已经产出甚至被流式发出**。中毒标记保证了"组最终一定
会被 abort、不会被自愈骗过去悄悄产出成功结果",但不能追溯撤回已经发生的那一帧输出。这不是
"已解决",是一个已知的、尚未补上的正确性缺口。

**加了 prefill 侧门控之后,decode 侧兜底还会被触发到的场景(诚实说明,不是"已解决")**:
1. **KV 压力下的中途 retract**:门控只在 prefill 准入这一刻起作用——一个组已经全员准入、
   开始 co-batched decode 之后,如果调度器因为 KV 压力对 running batch 做 retract(选中了
   组里的某个成员退回 `waiting_queue`),这件事发生在 prefill 门控完全看不到的地方(运行中
   的请求,不在 `waiting_queue` 里),门控没有能力阻止,也不该去阻止(那是 upstream 自己的
   retract 淘汰逻辑,不是这个仓库拥有的代码路径)。这仍然要靠 decode 侧隔离+级联 abort 兜底。
2. **保守估计仍然可能偏乐观的边界情况**:`_estimate_available_prefill_tokens` /
   `_fusion_group_prefill_cost` 是主项(含 token 预算、chunked-prefill/input-token 上限、
   `max_running_requests` 请求数上限)的保守复刻,不是精确复刻(见上一段),真实引擎里没建模
   到的更细维度(比如 `prefill_max_requests`、context-parallel/LoRA 相关的额外分支、
   `page_size > 1` 时的对齐开销、KV 压力下 retract 拆散后重新入队的组的真实扣费)理论上仍可能
   让门控放行一个实际装不下的组。

以上任何一种情况发生时,融合请求会被 decode 侧检测到"缺员"从而直接 abort,而不是透明地多等
一两个 tick 再重试——客户端仍然需要能处理"融合请求返回一个 abort/error 而不是音频"的情况
(并可自行重试)。门控把这类情况的发生频率从"调度压力下的常态"压到"边界情况",但不是把它
归零,也不应该被包装成"两层原子保证都齐了"。

## 为什么 ramp(delay/EOC 状态机)不会因参考长度不同而错位——以及一个尚未补上的口子
一个容易误判的点:N 个 sibling 的参考音色时长可能不同,直觉上会担心各自的
delay/EOC 状态机(见 `sampler.py::HiggsBatchedSamplerState`)因此错位。多数情况下不会,
原因是两点组合,而非某种"对齐参考长度"的机制:

1. `stages.py` 里 `_MAX_REF_AUDIO_SEC = 100` 把单条参考**音频**硬顶在 7500 帧
   (100s×75Hz),留了大约 692 个 token 的余量给文本+特殊 token,不超过 `chunked_prefill_size`
   (8192,见 `engine_builder.py`)——"chunked prefill of the multi-codebook prompt is unsafe
   (sampler state machine has no rollback)"。因此*通常*每个 sibling 的 prefill 都能一步内
   完成,不会被拆成多个 scheduler tick。
2. `HiggsBatchedSamplerState.reset_row`(`sampler.py:91`)只在 sibling 首次拿到
   sampler-pool 行时(即 prefill 准入那一刻)把 `delay_count`/`step_count` 清零,且之后只由
   `batched_step` 的调用次数(解码步数)驱动——与该 sibling 自己参考音频多长完全无关。

**尚未补上的口子**:上面第 1 点只顶住了*音频*本身的长度,没有顶住"音频 + 目标文本 + 特殊
token"的**总**长度。`_MAX_REF_AUDIO_SEC` 留的 692-token 余量对绝大多数正常长度的目标文本
够用,但没有任何代码显式校验"这条 sibling 的完整 prompt 长度 ≤ chunked_prefill_size"——
一段异常长的目标文本 + 一条接近 100s 上限的参考音频组合起来,理论上仍可能压过
`chunked_prefill_size`,触发多 tick 的 chunked prefill,这时"prefill 一步完成"的前提就不
成立了。修这个口子需要在 `build_fusion_sibling_requests`(`request_builders.py`)校验总
prompt 长度,但那里目前拿不到活的 `chunked_prefill_size`(一个 server_args 值,当前这条构
建路径完全不感知调度器配置)——贸然写一个脱离实际配置的硬编码阈值,又会制造新的一份"文档/
常量说一套、真实配置是另一套"的漂移,所以暂时按已知限制记录,而不是塞一个假安全的检查。

（历史上这里曾设想过"prefill 时按组内最长参考做 BOC 左填充对齐"的方案,但从未实现,
也不需要——上面两点已经是完整的正确性论证。）

## 已知限制
- prefill 侧原子准入门控(见上文)是保守估计 + 尽力而为,不是形式化证明;已运行请求中途被
  KV 压力 retract 拆散一个正在 co-batch decode 的组,门控看不到也管不了,仍然只能靠 decode
  侧隔离+级联 abort 兜底;客户端仍需要能处理"融合请求返回 abort 而不是音频"的情况并自行重试。
- 门控本身可能因为估算持续偏保守而反复暂扣同一个组;`_advance_withhold_ticks_and_give_up`
  在连续暂扣满 `_MAX_FUSION_WITHHOLD_TICKS`(200)个 tick 后放弃并对客户端报错,避免无限期
  暂扣、客户端连最终结果都等不到——但这个阈值本身是拍的一个数字,没有基于真实引擎下"正常情况
  最多需要暂扣几个 tick"的实测数据来标定。
- 单个 sibling 的 prompt 总长度(参考音频 + 目标文本 + 特殊 token)没有显式校验是否超出
  `chunked_prefill_size`,只靠 `_MAX_REF_AUDIO_SEC` 留的固定余量隐式兜底(见上一节)。
- sampler pool 容量 = `max_running_requests + 1`;一个融合请求占 N 行,部署需按
  "1 融合请求 = N 行" 给 KV/并发计费,否则 KV 压力下更容易触发 decode 侧的缺员 abort。
- 仅 logit 融合(非 prompt 拼接);CFG 外推留作 follow-up。
- 融合请求完全绕开现有的参考缓存栈(speaker artifact cache / `reference_code_cache_key` /
  已上传语音),每次都会重新过一遍 codec 编码,已上传的具名语音也无法参与融合。作为 v1 范围
  可以接受,但这是一个尚未文档化过的能力缺口。

## 已修复的两个"隔离/级联"自身的 bug(第二轮对抗审查抓到,均已修正)
第一版的 decode 侧隔离 + 级联 abort(上一节)在实现细节上曾有两处自己的问题,均已修正:

1. **`_populate_fusion_buffers` 的 dirty-flag 越界**:早期实现只把"清空"写到 `[:bs]`(当前
   这一步的 batch size),但 `_cg_fusion_group`/`_cg_fusion_weight` 是固定 `pool_size` 的
   buffer,不随每步 batch size 变化。若一步 bs=4(含一个融合组占 2,3 号槽位)之后,组结束、
   batch 缩到 bs=2,只清了 `[:2]`,槽位 2,3 仍留着旧组的 `group=[2,2]`/`weight=[0.6,0.4]`;
   若 dirty flag 此时被错误地设为 False,后续 batch 再长回 bs=4、两个全新无关请求恰好落在
   槽位 2,3,`has_any_fusion()` 早退检查会让这两个不相关的请求被静默按旧组"融合"在一起。
   修法:dirty→clean 的那次转换改成清空整个 buffer(`[:pool_size]`),而不是只清 `[:bs]`,
   一次性、简单、正确,不需要额外的高水位状态。回归测试:
   `test_populate_buffers_clean_reset_scrubs_stale_slots_beyond_a_shrunk_batch`
   (`test_voice_fusion_pipeline.py`)。
2. **级联 abort 对"同批已完成但还没被下一次 filter_batch 清理"的成员会误伤**:一个组的多个
   在场成员如果*同时*被 `_populate_fusion_buffers` 隔离+标记 FINISH_ABORT,它们会在同一次
   `stream_output` 调用里各自轮到自己的处理——但第一个被处理的成员触发的级联,如果不加区分地
   对"组里其它还注册着的成员"调用 `abort()`,会把同批里*也*刚完成、还没被下一轮 `filter_batch`
   清理出 `running_batch.reqs` 的成员也 `abort()` 掉——而 `abort()` 对"已完成但还在批里"的
   请求走的是立即从 `running_batch.reqs` 摘除这条路径,和 BLOCKING-2 里禁止的"prepare_for_
   extend 之后摘 reqs"是同一类张量/reqs 错位。修法:`stream_output` 先收集本轮所有"本次调用
   即将完成"的 rid 集合,级联只对*不在*这个集合里的成员(即真正缺席、下一轮才可能自己冒出来
   的那个 sibling)调用 `abort()`——同批一起完成的成员各自走自己的正常处理,不需要被级联。

## 仍需真实引擎验证的项(本机 Windows 无 sglang,无法跑真实引擎)
1. **decode 侧隔离 + 级联 abort 的端到端行为**:`_populate_fusion_buffers` 的隔离逻辑和
   `_cascade_abort_split_fusion_group` 的级联逻辑都各自过了单测(`test_voice_fusion_pipeline.py`
   里 mock 了 `HiggsTTSModelRunner`/`FusionRegistry` 直接调用真实方法),但两者串起来的真实
   端到端行为(KV 压力下真的触发 retract → 隔离 → 级联 → 客户端收到什么)只验证到"逻辑上
   应该正确",没有在真实引擎上跑过。
2. **prefill→decode 过渡的实际发生率,以及新增门控的实际效果**:加了
   `_reorder_queue_for_atomic_fusion_admission` 门控之后,sibling 们在真实负载下实际上有
   多大比例一起进 decode(而不是触发 decode 侧的 abort 路径),门控生效前后这个比例的对比,
   `_estimate_available_prefill_tokens` 的保守估计在真实 KV 池/tree cache 状态下是否经常
   过度保守(不必要地多等 tick,浪费吞吐)或——更需要关注——是否存在被低估掉的预算维度导致
   估计过于乐观、仍然放行了实际装不下的组。这些只有在真实引擎 + 真实负载下才能量化,直接决定
   这个功能在生产环境下的可用性,需要用不同长度的参考音频、不同并发度实测。
3. **CUDA graph 兼容性**:`_cg_fusion_group`/`_cg_fusion_weight` 每步重填是否与图重放兼容。
4. **非融合热路径零开销**:`FusionRegistry.has_any()` 让零融合流量的服务器跳过每步的
   follower 检查/buffer 填充,已有针对计数器本身的纯 Python 单测
   (`test_voice_fusion.py` 的 `test_registry_*`),但吞吐层面的"确实零开销"还没有真实
   engine 下的压测数据支撑;此外 `fuse_group_logits` 在 CG 解码路径上是无条件调用的(不像
   eager 路径有 `is_fused` 门槛),对全零融合流量的服务器而言这是一次额外的、可忽略但确实
   存在的 fp32 softmax 开销,不是真正的零成本。
5. **`FINISH_ABORT` 直接赋值 `finished_reason` 而非走 `to_finish` 的时机是否安全**:上游
   `Req.to_finish` 字段的注释明确写着"如果想在事件循环中途 abort 一个请求,应该设置
   `to_finish` 而不是直接设置 `finished_reason`,否则请求会被过滤掉、再也不会响应"。
   **更正**(第二轮审查抓到第一版这里的错误理解):upstream 的 `process_batch_result_decode`
   确实每个 decode step 都会调用 `req.check_finished()`,不是"Higgs 完全不走这条路"。Higgs
   TTS 的整条完成信号链路(包括早已验证工作的正常完成路径 `_mark_sampler_finished`)在这个
   相同的代码层级都是直接设置 `finished_reason`,从未使用 `to_finish`——`_populate_fusion_
   buffers`(BLOCKING-3)、`_cascade_abort_split_fusion_group` 都是照抄这个既有约定,而不是
   新造一种写法。但"Higgs 这套直接赋值的写法为什么能绕开上游注释里的警告、和 `check_finished()`
   的实际交互到底是不是安全的",仍然是一个没有在真实引擎上专门针对 abort 路径验证过的开放问题,
   不是已经证明安全。
6. **异步 lookahead 与 fusion 的交互(已按最保守方式规避,未实测)**:这个仓库自建的
   one-step-lookahead 解码(`enable_async_decode`,`OmniScheduler._resolve_and_process`)
   会在 launch 阶段(`_populate_fusion_buffers` 设置 FINISH_ABORT 发生的地方)之后、resolve
   阶段之前有一个时间窗口;`_resolve_and_process` 用"resolve 前先快照 `req.finished()`"的
   方式区分"上一步就已结束的过期行"和"这一步 resolve 过程中才结束的行",但 launch 阶段设置的
   FINISH_ABORT 发生在这个快照*之前*,会被误判成"上一步的过期行"而被整行摘出批次、永远不会
   走到 `stream_output`,级联 abort 也就永远不会触发——这正是级联机制想避免的"僵尸 sibling"。
   修法:`HiggsTTSModelRunner.lookahead_eligible` 现在在有任何融合流量时返回 `False`
   (`model.has_any_fusion()`),让融合相关的 batch 强制走同步路径,整个 launch/resolve 分离
   带来的时间窗口根本不存在。这个修法本身只在本机做了静态代码走查(确认
   `lookahead_eligible` 返回值确实能让 `_event_loop_async_decode` 完整跳过 launch/resolve
   分离,直接同步跑),没有在真实开启 `enable_async_decode` 的引擎上实测过。
7. **prefill 门控与并发 `abort()` 的真实交互,以及放弃阈值的标定**:
   `_reorder_queue_for_atomic_fusion_admission`/`_restore_queue_after_atomic_fusion_admission`
   持 `self._request_admission_lock` 防止一个正在暂扣窗口期内的请求被另一个线程的 `abort()`
   复活(见"Co-batching"一节),这个交互只在单测里用 mock/手写的 `abort` 验证过锁本身可重入、
   以及 restore 会剔除 `_aborted_request_ids` 里的 rid,没有在真实并发(多个请求同时到达/
   取消、真实 GIL 调度)下跑过。`_MAX_FUSION_WITHHOLD_TICKS=200` 这个放弃阈值也是拍的,没有
   基于真实引擎下"正常场景最多暂扣几个 tick 就该放行"的数据标定过,过小可能在正常波动下就误杀
   本该等等就成的组,过大则放大"客户端要等很久才会等到最终失败"的体感延迟。
8. **log-linear 归约 + 熵匹配修法的真实引擎复现验证**:`fuse_group_logits` 改概率空间算术
   平均为 log 空间加权求和、再加熵匹配缩放(见"核心算法"一节)这两层修法本身有
   `test_voice_fusion.py` 的纯 torch 单测守护数学正确性,但"两极分化/精度加权偏斜是否真的
   消失"只能靠真实引擎实测——已经用同样的 A/B 对照协议(不同随机种子重复生成、按参考音色单独
   克隆的音高中位数做锚点)在真实 GPU 上验证过一轮:纯 log-linear 归约(未加熵匹配)把"50/50
   随机锁死一个"变成了"8 次里 7 次锁死同一个、响度对齐后偏斜反而更彻底(8/8)",排除了测试
   素材响度不对等的解释,坐实了精度加权是真实机制;熵匹配这一层加上去之后,还需要用同样的
   权重矩阵 {0/1, 0.25/0.75, 0.5/0.5, 0.75/0.25, 1/0} 重新跑一遍,确认 0.5/0.5 时音高不再
   偏向任何一方、稳定集中在两个参考音色的(对数尺度)中间值附近,且方差和单一音色克隆时相当。
   另外需要留意"两个参考音色的分布在某一步完全不重叠"这种边界情况下几何平均可能采到质量很薄的
   尾部 token、产生音质瑕疵(粗糙/破音)——真实引擎上应该用差异较大的音色对专门听一遍这种情况;
   熵匹配的 3 步定长牛顿迭代对"温度趋近于 0(greedy)导致方差趋近于 0"这类退化输入的数值稳定性
   也只在合成数据的单测里验证过,没有在真实引擎的极端 sampling 参数下跑过。
9. **(已完成,结论见"AR 滞后"一节"结案"段落)** 轨迹反馈控制器改用 `matched_logits` 观测量
   之后的真实引擎复现验证:已经在真实 GPU 上用同样的 A/B 对照协议跑完 λ=0.1/0.2 两组、各 8 次
   独立生成。结论:方向性 bug(锁死固定一侧、增益越大越彻底)确认修复,但"稳定居中"这个预注册
   成功标准未达到——两极分化的形状仍在,判定为这一层机制的结构性天花板而非调参可解,已按项目
   自己写好的止损标准停止在这一层,不再继续在输出分布层面加码。留在这里的记录仅供追溯,不再是
   待办项。

## 第五阶段:参考侧融合(reference-space fusion)——零训练重设计

止损之后对"音色融合算法"整体重新立论。硬约束:零训练(不允许任何权重更新/微调/LoRA/
新学习模块;允许 frozen 组件的任意推理时计算与 DSP)。

### 为什么彻底离开输出分布层

前四个阶段修掉了四个真实 bug(调度拆批、算术平均双峰坍缩、精度加权偏斜、控制器观测偏差),
每一层修正都被真实引擎复现验证,但 0.5/0.5 的输出仍然是种子双峰。四轮证据合起来指向的是
层级性结论,不是实现瑕疵:

1. 两个 expert 条件分布的高概率交集是 {A-like 帧, B-like 帧};"中间音色帧"在两个分布里
   都是低概率尾部,任何 pooling(算术/几何/熵匹配)都只能在交集里挑,无法把尾部变成众数。
2. AR hysteresis:前几帧确立 register 之后,说话人一致性先验作用于已生成历史,两个 expert
   都预测"延续当前 register",融合从此退化为恒等。
3. 每步重加权这个 actuator 无法触及 KV 中已积累的承诺。

推论:融合必须发生在模型形成说话人后验**之前**——让模型看到的参考本身已经是中间音色。
这样"中间音色"就是参考的众数而不是两个分布的尾部,AR hysteresis 从敌人变成盟友(它锁定的
register 恰好就是我们要的中间 register),而单参考克隆根本没有产生双峰的机制。

### 新算法:校准合成 + WORLD 参数域 morph 出混合参考

给定 N 段参考音色与归一化权重,离线构造一个"混合音色参考",此后所有请求都是普通的
单参考克隆(构造一次,按 (音色组指纹, 权重, 算法版本) 缓存):

1. **校准合成**:选一段固定的校准文本 S(音素覆盖较丰富的自然语体),用引擎让每个参考
   音色各自克隆朗读 S(固定种子表 + F0 质量闸:输出中位 F0 相对该音色单克隆锚点偏差
   |ln 比| < ln 1.35,不过闸则换种子重试)。得到内容完全相同、音色各异的 N 段音频。
2. **WORLD 对齐 morph**:对每段做 WORLD 分解(harvest F0 + stonemask 精修、cheaptrick
   谱包络、d4c 非周期性);以权重最大者的时间轴为基准,用谱包络特征(log-SP 32 带池化、
   逐维去均值)自实现 DTW 逐帧对齐;然后按权重插值——log-F0 加权平均(单方 voiced 的帧用
   全局音高中位数比修正后保留 voiced)、谱包络 log 域加权平均、非周期性线性加权;WORLD
   重合成得混合波形。v1 不做时长 morph(语速由目标文本与模型主导,参考语速影响弱)。
3. **混合参考进入标准管线**:混合波形作为普通 raw-audio 参考(转录 = S)走既有
   waveform 前传路径,由 audio_encoder 阶段 codec encode;之后的每个请求就是一条
   完全标准的单参考克隆请求——CUDA graph、radix cache、流式输出全部白得。

权重语义:插值系数即权重,w∈[0,1] 连续;N>2 直接推广为多元加权平均(全部对齐到
argmax-weight 的时间轴)。w=0/1 或单参考时短路直接用原始参考(不经过校准合成,保真)。

关键机理押注(也是与被证伪路线的本质区别):**克隆是一次流形投影**。WORLD morph 的输出
不必完美——它带有轻微 vocoder 伪影、谱包络插值只是 formant 中间化的近似——但它只是参考,
不是成品;zero-shot 克隆模型会把参考投影回真实语音流形,提取的是音色统计而非逐帧细节。
morph 误差表现为"中间点位置的小偏差",而不是"输出质量的劣化"。这与输出侧 morph(生成 A、B
两版再插值,伪影直接进成品、每请求 2 倍成本、流式不可行)形成对比,后者仅作为保底降级路径。

### E1 最小端到端验证(真实引擎,已完成)

协议:与前四阶段同一对测试音色(A 男低、B 女声)、同一验证文本;校准文本约 35 字;
morph α∈{0.25, 0.5, 0.75}(α = B 的份额);每个混合参考做独立采样克隆(α=0.5 8 次,
其余各 4 次);端点对照用 cal_A/cal_B 自身再克隆(recycle,与混合参考同为"二手合成物",
对照才公平);另设 concat(cal_A+静音+cal_B)拼接单参考对照(候选"双参考 in-context"的
最简形式)。预注册判据:α=0.5 的 8 次输出在 recycle 端点 log 轴上位置均值 ∈ [0.35, 0.65]
且带内比例过半、std < 0.15。

结果(本地 librosa.pyin 中位 F0,与历史锚方法一致;pod 侧 pyworld.harvest 交叉复核一致):

| 组 | n | 输出中位 F0 (Hz) | 几何均值 | log 位置(recycle 端点) |
|---|---|---|---|---|
| cal_A recycle(端点) | 2 | 106.6, 117.9 | 112.1 | ≡ 0 |
| mix α=0.25 | 4 | 122.8, 122.8, 149.0, 148.5 | 135.1 | 0.29(目标 0.25) |
| mix α=0.5 | 8 | 156.5, 141.4, 154.7, 146.0, 154.2, 155.1, 160.1, 144.3 | 151.4 | **0.47,std 0.068,8/8 ∈ [0.35, 0.65]** |
| mix α=0.75 | 4 | 179.7, 162.0, 192.6, 199.4 | 182.9 | 0.76(目标 0.75) |
| cal_B recycle(端点) | 2 | 217.5, 208.9 | 213.1 | ≡ 1 |
| concat 拼接对照 | 4 | 130.1, 114.5, 121.2, 119.3 | 121.1 | 0.12(锁 A) |

判读:

- **双峰消失**:mix 三组共 16 次克隆,0 次锁端;α=0.5 组 log-std 0.042,与单音色克隆的
  自然方差同级(recycle 组 0.020/0.051)。旧机制同协议 0.5/0.5 下是 ~15/16 锁端
  (λ=0.1 修复后 1/8 带内、λ=0.2 0/8 带内)。预注册判据达标。
- **权重连续单调**:输出组几何均值 112.1 → 135.1 → 151.4 → 182.9 → 213.1,随 α 严格单调,
  组间零重叠——w 第一次成为真实的连续 actuator(旧机制下 0.9/0.1 只会"翻转赢家")。
- **morph 管线自身精确**:三个混合参考自身的中位 F0(120.7/145.1/170.6)与 log 插值理论值
  (118.8/143.1/172.4)偏差 < 2%。
- **克隆坍缩被证伪**:模型没有把混合参考拉回任何真实端点,而是稳定跟随参考的中间音色。
- **拼接对照失败(反证)**:concat 参考 4 次全部锁 A(0.12),证明"把两段参考都给模型"
  不产生平均化——必须在参考信号本身完成融合,这正是本方案的差异化所在。
- 已知的系统性小偏移:克隆输出相对参考自身 F0 普遍略有上移/向先验漂移(cal_A recycle
  +13.7%、α=0.5 +4.3%),这是 zero-shot 克隆的基线保真特性,不是融合机制引入的。

### 工程集成(已实施,全部收敛在 higgs 推理管线内部)

约束:不改上层调用方(API 仍是 `references: [{audio|codes, text?, weight}]`),不做
serve 层编排;构造管线完整地活在 higgs 自己的 stage 代码 + engine 进程里。实际架构:

- **preprocessing**(`stages.py`):融合请求初始化 `state.fusion_build`(校准文本 +
  最终请求的 prompt 拼接零件 `build_prompt_parts`,见 `text_tokenizer.py`——最终 prompt
  的 `-100` 占位数取决于混合参考帧数,在引擎侧才可知,所以只能传"前后件");pre-encoded
  参考在此补校准 prompt。
- **audio_encoder**(`stages.py`):raw-audio 参考照旧 codec encode,并为每个参考补
  校准 prompt(其音色朗读校准句)。
- **tts_engine**(核心,新模块 `fusion_reference.py`):
  - `request_builder`(`request_builders.build_reference_fusion_requests`)先查
    engine 进程内的混合参考缓存(key = 各参考 codes 指纹 + 归一化权重 + 校准文本 +
    算法版本):命中 → 直接构造普通单参考请求,零额外成本;未命中 → 借用
    `fusion_siblings` 入队通道发出 N 条**引擎内部校准行**(固定种子/采样参数,
    `internal_done_callback` 标记),并向 `FusionReferenceOrchestrator` 注册构造组。
  - `OmniScheduler.stream_output` 新增一个通用的 internal-request 短路(共享文件唯一
    改动,约 20 行):带 `internal_done_callback` 的行完成时填好 finish_reason、释放
    引擎槽位、回调编排器,**永不外发**——客户端只订阅 `request_id`,它在真请求完成时
    才收到结果。
  - 编排器(engine 进程单例,挂在 model 上,`post_scheduler_setup` 里绑定 scheduler):
    scheduler 线程回调只收集 codes 与推进状态机;单工作线程执行重活——undelay →
    CPU codec(fp32,懒加载)decode 校准音频与各参考原声 → F0 质量闸(不过则用下一个
    固定种子补发重试行,上限 3 个种子)→ WORLD morph(N>2 按权重两两哈夫曼式归约,
    每步都是 E1 验证过的二元 morph)→ CPU codec encode → 混合参考 codes → 写缓存 →
    `prefix + [-100]×T + suffix` 拼出最终 prompt → 把真请求(用户原始采样参数 + 流式
    元数据)入队。真请求此后与任何普通单参考请求完全同构,vocoder/流式零感知。
  - abort/超时:`_fusion_group_members` 里注册 `request_id → {校准行 rids}`(成员集合
    刻意**不含** `request_id` 本身——原子准入门控按"组员是否都在队列"判定,而
    `request_id` 在构造期没有队列行,含入会被无限扣留);client abort 经现有级联通道
    杀掉在途校准行;编排器另有 300s deadline 扫除兜底(覆盖"行在等待队列里被摘除、
    永远到不了 stream_output"的路径)。
- **模式开关**:`HIGGS_FUSION_MODE=reference`(默认)| `logits`(旧 sibling 逐步
  logit 融合,保留作研究对照;其原子准入/共享种子/级联 abort 基础设施全部保留)。
  校准文本可用 `HIGGS_FUSION_CAL_TEXT` 覆盖(进缓存 key)。
- 新增依赖:pyworld(BSD;WORLD 分解/合成),DTW 为自实现 numpy,无其它新依赖。
- 单测:`test_reference_fusion.py`(19 项:缓存 key、DTW、prompt 零件重组、编排器
  状态机 happy/abort/空输出/质量闸重试/abort 竞态/超时扫除/缓存淘汰、WORLD morph 的
  log 加权落点与权重单调性),不依赖 sglang,纯 CPU 可跑;旧 `test_voice_fusion.py`
  44 项回归通过(logits 对照模式未受影响)。

### 长参考自动切段融合(split_fuse,复用本机制)

单参考 raw 音频超过 `HIGGS_REF_TRIM_SECONDS`(默认 30s)时的默认处理
(`HIGGS_REF_LONG_MODE=split_fuse`):preprocessing 把整段音频等分为 ≤4 段
(丢弃语音活跃占比 < 0.15 的段),当作同一说话人的 N 路等权重参考走上述
reference-space 融合——整段录音都对音色有贡献,而不是只保留一个窗口。要点:

- **等长切段**是刻意设计:audio_encoder 对等长段用 `codec.encode_batch`
  一次 GPU 前向批量编码;段码另有进程级内容寻址缓存(同一参考重复请求
  零编码),引擎侧混合参考缓存则免去重复校准构建。
- **确定性**:同一段音频永远切出同样的段(内容哈希/缓存 key 稳定)。
- **降级兜底**:auto-split 构建携带一对无参考转写的 prompt 零件
  (`fallback_prompt_prefix/suffix`);若校准 F0 闸在全部 3 个种子上耗尽,
  编排器不再对客户端报错,而是取最高权重的原始段做普通单参考克隆
  (`FusionReferenceOrchestrator._serve_fallback`)——保证"以前能成的请求
  现在也能成"。
- **reference_text 语义**:切段后各段与全文转写不再对应,用户传入的
  reference_text 会被忽略(打 warning 日志);混合参考的转写是校准句。
- 仅一段有语音时退化为"用该段做普通单参考"(等效基于活跃度的裁剪)。
- `HIGGS_REF_LONG_MODE=trim` 切回纯裁剪(取最优 30s 窗口,零额外成本);
  `HIGGS_FUSION_MODE=logits` 或 `HIGGS_REF_TRIM_SECONDS<=0` 时自动回退 trim。
- 尚未做的听感验证:同声源各段融合的混合参考质量(E1 只测过异声源对)。

### 遗留风险(E1 未覆盖,集成前需补验证)

1. **听感**:E1 只量化了 F0 维度;"morph 伪影被克隆净化"的假设需要人耳确认(重点听
   α=0.5 输出有无 WORLD buzzy 感、formant 是否自然)。
2. **formant 细节**:v1 谱包络是 log 域直插(近似"平均谱"),未做频率轴 warp 的
   formant 轨迹插值;对 formant 结构差异极大的音色对,中间态可能偏"模糊"而非"中间"。
3. **音色对泛化**:E1 只测了一对差异极大的音色(男低/女声);同性别相近音色、带背景噪声
   参考、跨语言参考的表现未测。
4. **α=0.25 组内小分裂**(122.8×2 vs 148.5×2):低权重端的克隆方差略大,需要更多样本
   标定每档权重的实际落点方差。
5. 校准文本的语种/风格与目标文本不一致时的影响未测(E1 校准与验证同为中文自然语体)。
