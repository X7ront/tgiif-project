# TGIIF：按 CompGCN 风格组织的公式与证据对齐稿

## 0. 文档边界

`1911.03082v2.pdf` 是 Vashishth et al. 的 **Composition-based Graph Convolutional Network (CompGCN)**（ICLR 2020）论文。本文档借用其“问题定义 → 更新方程 → 可表达性/退化关系 → 实验问题”的论证组织方式，不把 CompGCN 的公式或实验结果当作 TGIIF 的实测结果。

下面的 TGIIF 公式均对应当前纯净 WN18RR 项目中的实现；其中带“实现定义”的式子是代码事实，带“命题/解释”的段落是需要由实验支持、但不声称已被数学定理完全证明的研究表述。

## 1. 问题定义与交集特征

给定多关系知识图谱

\[
\mathcal G=(\mathcal V,\mathcal R,\mathcal E),\qquad e_i=(h_i,r_i,t_i)\in\mathcal E.
\tag{1}
\]

对固定 `hash_hops=1`，每个三元组由 OpenKE 的 intersection-feature 管线映射为

\[
x_i=\Phi_{\mathrm{int}}(h_i,r_i,t_i)\in\mathbb R^{d_x},
\tag{2}
\]

其中当前主线使用 `feature_mode=triple`。这一步是输入表示定义，不等同于 CompGCN 的消息传递层。

## 2. 训练集 treatment 与反事实配对

所有 treatment 信息只由训练三元组构造。令训练集端点度为

\[
d_{\mathrm{tr}}(v)=\sum_{(h,r,t)\in\mathcal E_{\mathrm{tr}}}
\big[\,h=v\,\big]+\big[\,t=v\,\big].
\tag{3}
\]

三元组的训练集 popularity 特征为

\[
p_i=\log(1+d_{\mathrm{tr}}(h_i))+\log(1+d_{\mathrm{tr}}(t_i)).
\tag{4}
\]

在同一 relation 的训练样本内，以中位数划分 treatment：

\[
T_i=\mathbf 1\left[p_i>\operatorname{median}\{p_j:r_j=r_i\}\right].
\tag{5}
\]

令 \(\tilde x_i\) 表示按训练集统计量标准化后的交集特征，则反事实匹配为

\[
c(i)=\arg\min_{j}\|\tilde x_i-\tilde x_j\|_2,
\quad r_j=r_i,\quad T_j\ne T_i.
\tag{6}
\]

该约束保持 relation 不变，只改变由 popularity regime 定义的 treatment。缓存审计结果为：训练三元组 86,835；完整匹配率 1.0；有效 CF 匹配率 0.950124；同关系率 1.0；treatment 相反率 1.0；fingerprint 和索引一致。

## 3. Official baseline 与双分支表示

官方打分器给出 baseline logit 与 score：

\[
(u_i,\ell_i^0)=F_{\mathrm{base}}(x_i),
\qquad s_i^0=\sigma(\ell_i^0).
\tag{7}
\]

共享残差表示和两条分支为

\[
z_i^{I}=E_I(u_i),
\qquad z_i^{B}=E_B([u_i,p_i]),
\tag{8}
\]

其中 \(I\) 是 invariant branch，\(B\) 是 treatment/popularity-aware bias branch。两条分支均参与训练；在当前 `invariant-margin` 推理模式中，B 分支不直接改写最终排序。

分支输出和门控为

\[
\delta_i^I=D_I(z_i^I),\qquad
\delta_i^B=D_B(z_i^B),
\tag{9}
\]

\[
g_i=\sigma\left(\frac{G([u_i,p_i])}{\tau_g}\right),
\quad
\delta_i^{\mathrm{mix}}=\alpha_I g_i\delta_i^I+
\alpha_B(1-g_i)\delta_i^B.
\tag{10}
\]

式 (10) 是完整 additive fusion 的混合式；当前论文主结果使用 bounded invariant-margin fusion，其排序路径取

\[
\delta_i^{\mathrm{rank}}=\alpha_I g_i\delta_i^I.
\tag{11}
\]

## 4. 有界 invariant fusion

对一个 filtered-ranking query \(q\)，令 baseline 候选得分范围为

\[
R_q=\max_{j\in\mathcal C(q)}s_{qj}^0-
\min_{j\in\mathcal C(q)}s_{qj}^0.
\tag{12}
\]

实现先对残差作 query-level preparation、缩放和双曲正切压缩，再施加 residual multiplier \(m\)：

\[
\widehat\Delta_{qj}=r_s\tanh\!\left(P_q(\delta_{qj}^{\mathrm{rank}})\right),
\qquad
\Delta_{qj}=\operatorname{clip}(m\widehat\Delta_{qj},-fR_q,fR_q),
\tag{13}
\]

\[
s_{qj}=s_{qj}^0+\Delta_{qj}.
\tag{14}
\]

其中 \(f\) 为 `invariant_margin_fraction`，本轮恒等性小测使用 \(f=0.25\)，正式 1000ep 证据统一按验证协议重新评测为 \(f=0.0001,m=0.75\)。若启用 top-gap guard，则只有满足实现中的 gap ratio 条件的 query 才允许残差；本文不把 guard 误写成普遍理论保证。

## 5. 训练目标与可检验退化关系

官方 baseline 的 rank loss 保留在总目标中，且 baseline 参数不冻结：

\[
\mathcal L=
\mathcal L_{\mathrm{rank}}(s^0)
 +\lambda_{rr}\mathcal L_{\mathrm{res-rank}}
 +\lambda_{rp}\mathcal L_{\mathrm{preserve}}
 +\lambda_{cf}\mathcal L_{CF}
 +\lambda_t\mathcal L_{treat}
 +\lambda_o\mathcal L_{orth}.
\tag{15}
\]

当前实现中的三项可直接对应为

\[
\mathcal L_{CF}=\operatorname{SmoothL1}(z_i^I,z_{c(i)}^I),
\tag{16}
\]

\[
\mathcal L_{treat}=\operatorname{BCEWithLogits}(Q(z_i^B),T_i),
\qquad
\mathcal L_{orth}=\mathbb E_i\left[\cos(z_i^I,z_i^B)^2\right].
\tag{17}
\]

当前实现的可检验退化关系是

\[
m=0\ \Longrightarrow\ s_{qj}=s_{qj}^0,
\tag{18}
\]

以及在 invariant-margin 模式下

\[
\text{disable-}I\ \Longrightarrow\ s=s^0,
\qquad
\text{disable-}B\ \Longrightarrow\ s\text{ 与 Full 相同}.
\tag{19}
\]

式 (18) 是实现恒等式；式 (19) 是当前 fusion 语义的结构性验收，不是对所有 fusion 模式的普遍结论。

## 6. CompGCN 风格的可表达性叙事如何迁移

CompGCN 论文先给出 relation-aware update，再用 Proposition 4.1 说明通过限制参数可退化为 GCN、R-GCN、Directed-GCN 和 Weighted-GCN。TGIIF 可以采用同样的叙事，但必须把“可退化关系”限定为当前 scorer 的实现恒等式：

1. `m=0` 是 official baseline identity（式 18）。
2. `lambda_cf=0` 是 CF-off 训练控制，不等于“模型没有 invariant representation”。
3. `lambda_treat=0` 是 treatment-off 训练控制，不等于“删除 popularity 输入”；若要主张不使用 treatment，必须同时检查数据路径和分支输入。
4. `disable-invariant-branch` 与 `disable-bias-branch` 是 checkpoint 推理消融，不等同于重新训练后的结构消融。

因此，论文应称这些为 **implementation-level reductions and controls**，而不要声称已经证明 TGIIF 在数学意义上包含某一整类模型。

## 7. 与 CompGCN Q1–Q4 对齐的实验问题

| 论文式问题 | TGIIF 对应问题 | 当前证据 |
|---|---|---|
| Q1：整体 link prediction | Full TGIIF 相对自身 `m=0` baseline 是否改变验证排序 | 3 seeds 的 Full 均高于自身 baseline；增益 +0.017016、+0.039380、+0.015669 |
| Q2：组件/算子影响 | CF-off、Treatment-off、Treatment-shuffled，以及 I/B 推理消融 | 已有三 seed 1000ep 表和本次 20ep identity check |
| Q3：规模/参数效率 | treatment/cache 覆盖、匹配约束、分支诊断开销 | WN18RR CF audit 已完成；尚未做系统参数规模曲线 |
| Q4：任务泛化 | FB15k-237 的同协议验证与最终 test | 云端数据管线已准备；本文件不把未在本证据包内的数值写成已完成结果 |

## 8. 当前已测数据

### 8.1 1000ep WN18RR 主消融（validation）

统一报告协议：`hash_hops=1`、`feature_mode=triple`、推理 `f=0.0001,m=0.75`；每个 checkpoint 与自己的 `m=0` 控制比较。

| Seed | Full MRR | CF-off | Treatment-off | Treatment-shuffled |
|---:|---:|---:|---:|---:|
| 7 | 0.507368 | 0.509306 | 0.509243 | 0.509107 |
| 23 | 0.923058 | 0.923882 | 0.725452 | 0.923882 |
| 41 | 0.726847 | 0.729632 | 0.529035 | 0.714158 |

Full 相对自身 `m=0` baseline 的 MRR 增益分别为 `+0.017016`、`+0.039380`、`+0.015669`。CF-off 略高于 Full，因此论文应把 CF 解释为 invariance constraint，而不是已被证明的直接 MRR 增益来源。Treatment-off 在 seed 23、41 显著下降，但 seed 7 不下降；这应写成 seed-sensitive evidence。

### 8.2 20ep 公式对齐小测

该小测用于检查训练链路、CF/treatment/orth 诊断和 bounded fusion 是否真实启动，不用于最终性能比较：Full `m=0.75` 的 validation 为 `MRR=0.057308`、`MR=11708.169922`、`H@10=0.063448`、`H@3=0.055702`、`H@1=0.051912`。

### 8.3 20ep checkpoint 恒等性验证

| 推理设置 | MRR | H@10 | H@3 | H@1 |
|---|---:|---:|---:|---:|
| Full, `m=0.75` | 0.057308 | 0.063448 | 0.055702 | 0.051912 |
| Official baseline, `m=0` | 0.015750 | 0.017139 | 0.010218 | 0.009394 |
| Disable invariant, `m=0.75` | 0.015750 | 0.017139 | 0.010218 | 0.009394 |
| Disable bias, `m=0.75` | 0.057308 | 0.063448 | 0.055702 | 0.051912 |

这组结果验证了式 (18)–(19) 的实现语义：当前 invariant-margin 模式中，invariant branch 是唯一改变排序的残差路径，bias branch 仍通过 treatment supervision、CF/orth 相关训练和诊断参与模型学习，但不直接进入最终排序加法。

## 9. 证据边界与下一步小测试

已完成且可直接写入初稿：训练集-only treatment/CF 构造审计、三 seed 1000ep 主消融、固定 checkpoint 的分支恒等性、20ep 公式对齐 smoke test。

在不重新长训的前提下，推荐只补三类小规模证据：

1. 从 20ep JSONL 汇总 `cf_loss`、`treat_loss`、`orth_loss`、gate mean/std、两分支梯度和 residual saturation 的时间曲线。
2. 对已有 1000ep checkpoint 统一导出 branch contribution、treatment accuracy 和 CF valid ratio 的末段统计。
3. 对 FB15k-237 仅做同协议的 baseline/full/CF-off 最小验证，作为跨数据集管线证据；不要把尚未完成的结果写成结论。

不要把旧草稿中的 `MRR=0.913` 直接当作当前主结果；本证据包的统一主表是 `0.507368/0.923058/0.726847` 三 seed 结果。CompGCN 的 Q4 分类实验也不应被移植成 TGIIF 已完成的任务。

## 10. 可重复命令

公式恒等性检查脚本：

```bash
cd /path/to/CTS-Net-TGIIF-WN18RR-clean-20260906
source /home/xtront/miniconda3/etc/profile.d/conda.sh
conda activate intersection_feature_cu128
bash examples/run_paper_formula_identity_checks.sh
```

输出：`paper_evidence/smoke/identity_checks/*.log` 和机器可读汇总 `paper_evidence/smoke/identity_checks_summary.json`。
