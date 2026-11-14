# FB15k-237 TGIIF 云端实验日志整合报告

生成日期：2026-09-13  
数据集：FB15k-237  
正式实验种子：23  
归档：`CTS-Net-TGIIF-FB15k237-clean-20260907-cloud-results-20260913-100902.tar.gz`  
归档 SHA256：`F31638D647B16FC98EAA91954818E03C6E9B43EA47528F48915CD4933E5162AF`

## 1. 数据来源与审计范围

- 原归档保留不动，项目解压到独立目录 `CTS-Net-TGIIF-FB15k237-cloud-results-extracted-20260913/CTS-Net-TGIIF-FB15k237-clean-20260907`。
- 解压内容共 163 个文件、186,479,678 bytes（约 0.174 GiB）。
- 审计了所有 config JSON、epoch JSONL、history JSON、checkpoint、train log、run manifest 和 filtered-ranking 评测日志。
- 完整 JSON/JSONL 均可解析，未发现 NaN、Inf、JSON 损坏、重复 epoch 或中间缺失 epoch。
- 指标 CSV：`FB15K237_CLOUD_METRICS.csv`。

## 2. 实际实验口径

所有正式训练共有设置为：FB15k-237、seed=23、Adam、lr=0.005、margin=5、hash_hops=1、triple feature、hidden_dim=128、branch_dim=64、100 batches、25 entity negatives、训练期间不冻结 baseline。

实际主架构不是早期 gate-soft 双分支融合，而是：

- `fusion_mode=invariant-margin`
- 训练 config 中 `invariant_margin_fraction=0.25`
- Full：`lambda_cf=0.1`、`lambda_treat=0.1`、`lambda_orth=0.01`
- CF-off：仅将 `lambda_cf` 改为 0
- Treatment-off：仅将 `lambda_treat` 改为 0
- Treatment-shuffled：`shuffle_treatment_labels=true`

正式评测日志文件名记录为 `f0.0001`，即评测时使用 `invariant_margin_fraction=0.0001`；但评测程序没有把完整参数回显到日志正文，因此这一取值的证据来自文件名而不是日志内 manifest。所有正式结果均为 unconstrained filtered test ranking，评测三元组数为 20,466。

训练 config 中虽写有 `eval_split=valid`，但训练使用 `skip_test=true`，正式分数来自之后单独运行的 test evaluator。因此下表的 1000ep 以上结果均应标为 **test**，不是 validation。

## 3. 全部可用评测结果

### 3.1 50ep smoke（仅用于验证管线）

| 来源 | m | Split | MRR | MR | Hits@10 | Hits@3 | Hits@1 | H@10-H@1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Local smoke | 0 | valid | 0.140230 | 2526.020508 | 0.153151 | 0.134274 | 0.132991 | 0.020160 |
| Local smoke | 0.75 | valid | 0.158389 | 2965.175293 | 0.178671 | 0.151611 | 0.146621 | 0.032050 |
| Cloud smoke | 0 | valid | 0.178689 | 1732.332397 | 0.194554 | 0.176504 | 0.167265 | 0.027289 |
| Cloud smoke | 0.75 | valid | 0.155042 | 3135.257080 | 0.173767 | 0.151297 | 0.144140 | 0.029627 |

Local 与 Cloud smoke 的方向不一致，因此两组只能说明训练、保存和评测链路可运行，不能作为正式性能或可复现性证据。

### 3.2 正式 test 结果

| 实验 | Epoch | m | MRR | MR | Hits@10 | Hits@3 | Hits@1 | H@10-H@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 1000 | 0 | 0.242065 | 1461.767090 | 0.262020 | 0.236759 | 0.230455 | 0.031565 |
| Full | 1000 | 0.75 | 0.387460 | 908.586975 | 0.407578 | 0.384100 | 0.375305 | 0.032273 |
| CF-off | 1000 | 0 | 0.242065 | 1461.767090 | 0.262020 | 0.236759 | 0.230455 | 0.031565 |
| CF-off | 1000 | 0.75 | 0.407229 | 893.702698 | 0.425095 | 0.404280 | 0.395583 | 0.029512 |
| CF-off | 3000 | 0 | 0.374315 | 1154.296631 | 0.389353 | 0.368929 | 0.366681 | 0.022672 |
| CF-off | 3000 | 0.75 | 0.445628 | 685.445923 | 0.473370 | 0.442001 | 0.430910 | 0.042460 |
| Treatment-off | 3000 | 0 | 0.374315 | 1154.296631 | 0.389353 | 0.368929 | 0.366681 | 0.022672 |
| Treatment-off | 3000 | 0.75 | 0.413979 | 1152.319092 | 0.418572 | 0.411048 | 0.409802 | 0.008770 |
| Full | 3500 | 0 | 0.402564 | 1092.656738 | 0.417693 | 0.397366 | 0.395363 | 0.022330 |
| Full | 3500 | 0.75 | 0.460341 | 628.224365 | 0.488005 | 0.453508 | 0.449086 | 0.038919 |
| Full | 4000 | 0 | 0.429205 | 1013.199585 | 0.446619 | 0.425120 | 0.421064 | 0.025555 |
| **Full** | **4000** | **0.75** | **0.464146** | **667.105469** | **0.493233** | **0.457271** | **0.452946** | **0.040287** |
| Full | 4500 | 0 | 0.424894 | 1033.040771 | 0.446570 | 0.422725 | 0.414053 | 0.032517 |
| Full | 4500 | 0.75 | 0.459802 | 657.407166 | 0.489959 | 0.454363 | 0.447376 | 0.042583 |

> 表中最佳值仅表示归档内最高的已观测 test 分数。由于 3500/4000/4500 都直接评测了 test，不能在论文中把 4000ep 当成经 validation 选择后得到的无偏最终 test 结果。

## 4. 每个 checkpoint 的 m=0 配对增益

| 实验 | Epoch | Delta MRR | Delta MR | Delta H@10 | Delta H@3 | Delta H@1 | m=0.75 的 H@10-H@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 1000 | +0.145395 | -553.180115 | +0.145558 | +0.147341 | +0.144850 | 0.032273 |
| CF-off | 1000 | +0.165164 | -568.064392 | +0.163075 | +0.167521 | +0.165128 | 0.029512 |
| CF-off | 3000 | +0.071313 | -468.850708 | +0.084017 | +0.073072 | +0.064229 | 0.042460 |
| Treatment-off | 3000 | +0.039664 | -1.977539 | +0.029219 | +0.042119 | +0.043121 | 0.008770 |
| Full | 3500 | +0.057777 | -464.432373 | +0.070312 | +0.056142 | +0.053723 | 0.038919 |
| Full | 4000 | +0.034941 | -346.094116 | +0.046614 | +0.032151 | +0.031882 | 0.040287 |
| Full | 4500 | +0.034908 | -375.633605 | +0.043389 | +0.031638 | +0.033323 | 0.042583 |

所有完整正式 checkpoint 在 `m=0.75` 下均高于自身 `m=0`，且均满足 Hits@10 > Hits@3 > Hits@1。这里的 `m=0` 是同一个 TGIIF checkpoint 关闭 residual 后的 baseline 分支，不是另行训练的独立 baseline；同 epoch 的 Full/消融 m=0 完全一致，符合 baseline rank 隔离和确定性训练轨迹。

## 5. 架构诊断结论

`invariant-margin` 的实现明确令 `mixed_delta = invariant_contribution`，bias/treatment 分支只承担辅助监督，不直接加入最终 residual。日志与代码行为一致：

| 实验 | Epoch | 最终 gate mean | 最终 invariant contribution share | 最终 residual saturation | 最终 factual treatment accuracy | 最终 CF treatment accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full | 1000 | 0.999998 | 1.000000 | 0.995594 | 0.971172 | 0.900600 |
| CF-off | 1000 | 0.999998 | 1.000000 | 0.996789 | 0.960393 | 0.874097 |
| CF-off | 3000 | 1.000000 | 1.000000 | 0.998436 | 0.980132 | 0.926764 |
| Treatment-off | 3000 | 0.569768 | 0.999975 | 0.056347 | 0.550298 | 0.475560 |
| Full | 3500 | 1.000000 | 1.000000 | 0.990236 | 0.972315 | 0.908769 |
| Full | 4000 | 1.000000 | 1.000000 | 0.992242 | 0.981411 | 0.932656 |
| Full | 4500 | 1.000000 | 1.000000 | 0.993414 | 0.982984 | 0.935577 |

据此可以严谨地表述：当前 FB15k-237 主线是“treatment/CF 辅助训练 + invariant-only 最终打分”，而不是两个分支共同决定最终排名。Full 中 treatment classifier 正常学习，但 gate 饱和到 1，bias contribution 接近 0；Treatment-off 的分类准确率回到接近随机水平，证明 treatment loss 的开关确实生效。

## 6. 消融可得与不可得结论

### 可以直接支持的结论

1. **Residual/invariant correction 有稳定配对增益。** 七个完整正式 checkpoint 的 `m=0.75` MRR 均高于其自身 `m=0`。
2. **1000ep 的 CF 没有带来正向最终 test 增益。** 同 epoch 严格对照中，Full MRR=0.387460，CF-off MRR=0.407229，Full 低 0.019769；Hits@10/3/1 也分别低 0.017517/0.020180/0.020278。
3. **treatment supervision 的机制开关有效。** Treatment-off 3000ep 的 treatment accuracy 接近随机，treatment-on 配置则达到约 0.98/0.93；但归档缺 Full 3000ep，不能从现有结果单独量化 treatment loss 对同 epoch 最终 MRR 的净贡献。
4. **Full 的现有最高观测点为 4000ep。** 3500 -> 4000 的 MRR 从 0.460341 升到 0.464146，4000 -> 4500 降到 0.459802。配对 residual 增益则从 +0.057777 降到约 +0.0349。

### 当前不能严谨支持的结论

- 不能声称 CF 提升 FB15k-237 性能；唯一严格同 epoch 对照反而是 CF-off 更高。
- 不能声称 bias 分支直接改善排名；当前 fusion 明确排除了 bias contribution。
- 不能用 CF-off 3000 与 Treatment-off 3000 的差异直接分解两个机制贡献，因为两者分别关闭不同模块，且缺少 Full 3000 锚点。
- 不能把 4000ep test 最高值当作无偏模型选择结果；应先在 validation 选择 epoch/超参数，再只评测一次 test。

## 7. 完整性与缺失实验

| 运行目录 | 目标 epoch | JSONL | History | Checkpoint | 成对评测 | 判定 |
|---|---:|---:|---|---|---|---|
| smoke_ep50_s23 | 50 | 50 | 有 | 有 | valid m=0/0.75 | 完整 smoke |
| smoke_ep50_s23_cloud | 50 | 50 | 有 | 有 | valid m=0/0.75 | 完整 smoke |
| full_ep1000_s23 | 1000 | 1000 | 有 | 有 | test m=0/0.75 | 完整 |
| cf_off_ep1000_s23 | 1000 | 1000 | 有 | 有 | test m=0/0.75 | 完整 |
| treatment_off_ep1000_s23 | 1000 | 71 | 无 | 无 | 无 | **中断** |
| cf_off_ep3000_s23 | 3000 | 3000 | 有 | 有 | test m=0/0.75 | 完整 |
| treatment_off_ep3000_s23 | 3000 | 3000 | 有 | 有 | test m=0/0.75 | 完整 |
| treatment_shuffled_ep3000_s23 | 3000 | 1 | 无 | 无 | 无 | **中断** |
| full_ep3500_s23 | 3500 | 3500 | 有 | 有 | test m=0/0.75 | 完整 |
| full_ep4000_s23 | 4000 | 4000 | 有 | 有 | test m=0/0.75 | 完整 |
| full_ep4500_s23 | 4500 | 4500 | 有 | 有 | test m=0/0.75 | 完整 |

两个中断日志都没有 Python traceback、CUDA OOM、NaN 或显式错误信息。Treatment-off 1000 在 epoch 70 后停止，Treatment-shuffled 3000 在 epoch 0 后停止，更像进程被外部终止或会话中断；仅凭归档无法确定具体终止源。

归档还缺少：

- Full 3000ep（因此 3000ep 消融矩阵没有共同锚点）；
- 完整 Treatment-shuffled；
- 完整 Treatment-off 1000ep（若论文只统一采用 3000/4000ep，可不补这一项）；
- 正式长训 validation 日志；
- 其他 seed。

## 8. 轨迹一致性检查

- Full 1000 与 Full 3500 的前 1000 个 epoch JSONL 逐行完全一致。
- Full 3500、4000、4500 的重叠 epoch 逐行完全一致。
- CF-off 1000 与 CF-off 3000 的前 1000 个 epoch 逐行完全一致。

因此 Full 1000/3500/4000/4500 可以作为同一确定性训练轨迹的不同终点分析。归档内 `.ipynb_checkpoints` 文件属于编辑器生成的冗余副本，不计为额外独立实验。

## 9. 建议用于论文的数据边界

- 可作为当前 FB15k-237 单种子探索结果：Full 1000/3500/4000/4500、CF-off 1000/3000、Treatment-off 3000，以及每个 checkpoint 自身的 m=0 配对结果。
- 50ep local/cloud 仅放复现附录或工程记录，不进入主性能表。
- 当前最值得补的是同配置 **Full 3000 validation/test** 和完整 **Treatment-shuffled 3000**；它们能闭合现有 3000ep 消融矩阵。
- 若准备正式论文表，应先规定 validation 选模规则，再补冻结规则下的单次 test；否则 3500/4000/4500 的连续 test 比较会构成 test-set selection。
