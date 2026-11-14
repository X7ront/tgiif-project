# TGIIF 机制探针报告（WN18RR，冻结 checkpoint）

日期：2026-09-12  
协议：validation，strict filtered ranking，`feature_mode=triple`，`hash_hops=1`，`fusion_mode=invariant-margin`，`f=0.0001`，`m=0.75`。

## 1. 目的与架构保护

本组实验不重新训练，不执行 optimizer step，不保存或覆盖模型参数。全部干预发生在已训练 checkpoint 的前向打分路径上；新增代码仅为独立探针脚本。核心模型、训练器、官方评测器和测试均未修改。完整 52 项单元测试通过。

固定模型的计算路径写为：

\[
s_0(q,e)=\sigma\!\left(w_0^\top h_0(x_{q,e})+b_0\right),
\]

\[
z_s=E_s(x),\qquad z_I=E_I(z_s),\qquad
z_B=E_B([z_s,p]),
\]

\[
\delta_I=T(w_I^\top z_I+b_I),\qquad
\delta_B=T(w_B^\top z_B+b_B),
\]

\[
g=\sigma\!\left((w_g^\top[z_s,p]+b_g)/\tau_g\right).
\]

当前 `invariant-margin` 主线中，bias 分支只承担 treatment 辅助监督，不直接进入最终排序：

\[
r(q,e)=\alpha\tanh(g(q,e)\delta_I(q,e)),
\]

\[
c_q=f\left(\max_e s_0(q,e)-\min_e s_0(q,e)\right),
\]

\[
s(q,e)=s_0(q,e)+\operatorname{clip}(m\,r(q,e),-c_q,c_q).
\]

其中本实验 `f=10^{-4}`、`m=0.75`、`alpha=0.1`。训练目标由 official rank、residual rank/preserve、CF、treatment 和 orthogonality 项组成；梯度探针验证 official rank 与 residual 参数路径隔离。

## 2. 冻结 checkpoint 的打分路径干预

每个 checkpoint 均与它自己的 `m=0` 比较；每行共 6,068 个 head/tail query。

| checkpoint | 干预 | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|
| Full s23 | m=0 | 0.883678 | 0.882828 | 0.884311 | 0.884641 |
| Full s23 | Native | 0.923057 | 0.922215 | 0.923698 | 0.924028 |
| Full s23 | Gate=1 | 0.923882 | 0.923039 | 0.924522 | 0.924852 |
| Full s23 | Gate shuffled | 0.905428 | 0.904581 | 0.906065 | 0.906394 |
| Full s23 | Invariant shuffled | 0.923057 | 0.922215 | 0.923698 | 0.924028 |
| Full s23 | Invariant sign flip | 0.905696 | 0.904252 | 0.905735 | 0.905900 |
| CF-off s23 | m=0 | 0.883678 | 0.882828 | 0.884311 | 0.884641 |
| CF-off s23 | Native | 0.923882 | 0.923039 | 0.924522 | 0.924852 |
| CF-off s23 | 任一 gate/invariant 干预 | 0.923882 | 0.923039 | 0.924522 | 0.924852 |
| Treatment-off s23 | m=0 | 0.883678 | 0.882828 | 0.884311 | 0.884641 |
| Treatment-off s23 | Native | 0.725452 | 0.720007 | 0.724127 | 0.738299 |
| Treatment-off s23 | Gate=1 | 0.923882 | 0.923039 | 0.924522 | 0.924852 |
| Full s41 | m=0 | 0.711184 | 0.708635 | 0.712920 | 0.715722 |
| Full s41 | Native | 0.726854 | 0.724291 | 0.728576 | 0.731378 |
| Full s41 | Gate=1 / shuffled | 0.726854 | 0.724291 | 0.728576 | 0.731378 |
| Full s41 | Invariant shuffled | 0.706656 | 0.705010 | 0.707152 | 0.710778 |
| Full s41 | Invariant sign flip | 0.494698 | 0.490112 | 0.494232 | 0.496869 |
| Treatment-off s41 | m=0 | 0.711184 | 0.708635 | 0.712920 | 0.715722 |
| Treatment-off s41 | Native | 0.529034 | 0.523237 | 0.530323 | 0.540376 |
| Treatment-off s41 | Gate=1 | 0.729639 | 0.727093 | 0.731378 | 0.734179 |

配对 triple-cluster bootstrap（2,000 次）给出：

- Full s23，Native − m=0：`+0.039379`，95% CI `[0.033448, 0.044986]`。
- Full s23，Native − Gate shuffled：`+0.017629`，95% CI `[0.014169, 0.021089]`。
- Full s41，Native − m=0：`+0.015669`，95% CI `[0.011398, 0.020442]`。
- Full s41，Native − Invariant shuffled：`+0.020198`，95% CI `[0.016371, 0.023953]`。
- Treatment-off s23，Native − Gate=1：`-0.198430`，95% CI `[-0.211323, -0.186289]`。
- Treatment-off s41，Native − Gate=1：`-0.200605`，95% CI `[-0.212701, -0.188629]`。

因此，跨 seed 最稳定的发现不是“learned gate 总能直接提高排序”，而是：关闭 treatment supervision 后，结构分支本身仍能形成有用校正，但 learned gate 会错误压制或错配该校正；强制 `Gate=1` 在两个 seed 都恢复到高分。Full 中 gate 的候选级排序贡献具有 seed 异质性：s23 明显，s41 因 gate 与 cap 饱和而消失。

## 3. 截断、饱和与退化机制

| checkpoint | candidate gate mean | gate std | 正 cap 候选饱和率 | Native 与 Gate=1 精确相同比例 | 同号 invariant query 比例 |
|---|---:|---:|---:|---:|---:|
| Full s23 | 0.367752 | 0.418881 | 0.908666 | 0.974066 | 0.917436 |
| CF-off s23 | 0.979988 | 0.020216 | 1.000000 | 1.000000 | 0.956493 |
| Treatment-off s23 | 0.315567 | 0.434296 | 0.545981 | 0.868823 | 1.000000 |
| Full s41 | 0.999983 | 0.001680 | 1.000000 | 1.000000 | 0.423533 |
| Treatment-off s41 | 5.91e-16 | 9.82e-13 | 0.640351 | 0.294079 | 1.000000 |

解释：`f=0.0001` 使有效查询中的残差大量落在 `±c_q`。因此多数情况下真正进入排序的不是连续幅度，而是受限校正的符号或极少量未饱和候选。CF-off s23 与 Full s41 都完全饱和，故 Gate=1、gate shuffled 与 Native 同分。Treatment-off s41 的 gate 几乎为零，残差平均绝对值仅 `2.54e-30`；这些值虽数值极小但并非全为精确零，仍可改变 exact-tie 下的顺序，所以报告同时保留精确零与 `1e-12` 近零口径。

## 4. 表征探针：treatment 信息被路由到哪里

在 train-only CF cache 上抽取 20,000 个样本，使用 exact intersection-feature grouped 5-fold CV；同一精确特征行不会跨训练/测试折。

| 表征 | Full s23 AUROC | CF-off s23 AUROC | Treatment-off s23 AUROC |
|---|---:|---:|---:|
| 原始 intersection feature | 0.872407 | 0.872407 | 0.872407 |
| shared | 0.823820 | 0.733274 | 0.874529 |
| invariant | **0.553295** | 0.718510 | 0.790985 |
| bias | **0.999879** | 0.999517 | 0.989298 |

结果支持一个明确的表示分工：bias 分支持续保留 treatment 信息，而 Full 的 invariant 表征最接近不可线性解码。单独关闭 CF 或 treatment supervision 都会让 treatment 信息重新进入 invariant 表征，说明二者在当前实现中表现为联合约束，而非任一单项即可完成分离。

这只证明 treatment 可解码性下降，不等同于完整因果识别；并且 CF-off 的冻结 checkpoint 排序不差于 Full，因此不能声称“去 treatment 程度越高，MRR 必然越高”。

## 5. 梯度路径探针

Full s23 checkpoint 上读取 32 个训练 batch，执行反向但不执行 optimizer step。加权梯度范数均值：

| 损失 | Shared | Invariant | Bias | Gate |
|---|---:|---:|---:|---:|
| Residual rank | 0.076134 | 0.001505 | 0 | 0.000709 |
| Residual preserve | 0.214345 | 0.003114 | 0 | 0.001039 |
| CF | 0.000557 | 0.000060 | 0 | 0 |
| Treatment | 0.116569 | 0 | 0.006433 | 0 |
| Orthogonality | 4.34e-7 | 0 | 3.59e-8 | 0 |

Shared 参数上的平均梯度余弦：

- residual rank vs CF：`-0.540`；
- residual rank vs treatment：`-0.356`；
- CF vs treatment：`+0.673`；
- treatment vs orthogonality：`-0.718`（仅 9 个非零 batch）。

这说明 CF 与 treatment 方向总体协同，但二者和 residual ranking 在 shared encoder 上存在明显冲突。它解释了为何表征分离可以增强，而最终 MRR 不一定同步提高。当前 orthogonality 项梯度近乎失活，不应在论文中把它描述为主要有效机制。

## 6. 查询分层结果

Full s23 的增益集中在 `_hypernym`（relation 0）：MRR `0.699386 -> 0.801155`，增量 `+0.101769`；Full s41 为 `0.794150 -> 0.840175`，增量 `+0.046026`。其他关系基本不变，s41 的 `_derivationally_related_form` 还有 `-0.006023` 的轻微退化。

按自身 baseline rank 分组：

- s23 的 baseline `>100` 组由 `0.000124` 提高到 `0.349128`，但 baseline rank 1 组由 `1.000000` 降到 `0.999067`；
- s41 的 baseline `>100` 组由 `0.000172` 提高到 `0.064736`，但 rank 1 组由 `1.000000` 降到 `0.996527`。

因此该 checkpoint 的总体提升主要来自少数困难 `_hypernym` 查询被大幅救回，而不是对所有关系均匀改进。Popularity quartile 上的增益方向跨 seed 一致但强度异质：s23 最大增益位于 Q4（`+0.076100`），s41 最大位于 Q1/Q2（`+0.027932/+0.026671`）。

## 7. 可支持与不可支持的论文结论

可支持：

1. CF 与 treatment supervision 联合减少 invariant 表征中的 treatment 可解码信息，并将 treatment 信息保留在 bias 分支。
2. treatment supervision 通过 shared representation 间接稳定 gate/残差通路；它不直接给 gate 参数梯度。
3. bounded invariant correction 在两个 seed 都相对自身 `m=0` 获得正的 validation MRR 增益。
4. 当前增益主要是困难 `_hypernym` 查询的结构性修正；gate 的直接排序作用受 seed 与截断饱和状态影响。
5. CF/treatment 表征目标与 residual ranking 存在梯度冲突，是后续优化的明确对象。

不可过度声称：

1. 不可将 train-only popularity treatment 与匹配直接解释为真实世界因果干预。
2. 不可声称 bias 分支直接参与最终排序；在 invariant-margin 主线中它是辅助分支。
3. 不可声称 learned gate 在所有 seed 都优于常数 gate；现有证据恰好显示 Gate=1 常持平或略优。
4. 不可声称更强的表征去偏必然带来更高 MRR。

## 8. 证据文件

- `score_pathways/*/score_pathway_probe.json`：总指标与配对 bootstrap。
- `score_pathways/*/query_ranks.csv`：逐 query、逐干预排名。
- `score_pathways/*/group_metrics.csv`：relation、popularity quartile、baseline-rank 分层。
- `representations/*.json`：grouped-CV treatment probes 与 CF 距离。
- `gradients/full_s23/loss_gradient_probe.json`：逐损失梯度范数与余弦。
- `cap_saturation/*.json`：gate、cap、饱和与精确零统计。
- `score_pathways/invalid_order_alignment_full_s23/`：已明确标记为无效的早期顺序对齐试跑，不得进入论文。

