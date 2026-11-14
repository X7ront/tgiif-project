# WN18RR 核心 test 重评结果与采用边界

生成日期：2026-11-10

## 本轮最终采用范围

- 只采用实验 1-7 的统一 test 评价：同一 evaluator、filtered ranking、
  `fraction=0.0001`，并对每个 checkpoint 分别报告 `m=0` 与 `m=0.75`。
- 实验 10-19（全部原理/机制探针）明确排除，未运行。
- 实验 8（fraction 门禁）和实验 9（validation fraction sensitivity）虽已完成，
  但不属于论文当前必要实验，不纳入主结论。
- 实验 20（逐 query rank、分组指标、paired bootstrap）已停止，不纳入论文主证据。

## Test 结果

| Checkpoint | m=0 MRR | m=0.75 MRR | Delta MRR | m=0.75 Hits@1 | Hits@3 | Hits@10 | 论文采用状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| Full seed 23 | 0.881871 | 0.925364 | +0.043493 | 0.924378 | 0.926292 | 0.926611 | 主实验 |
| Full seed 7 | 0.505971 | 0.525274 | +0.019303 | 0.520421 | 0.520421 | 0.521059 | 多 seed |
| Full seed 41 | 0.711190 | 0.732405 | +0.021215 | 0.729260 | 0.734684 | 0.738194 | 多 seed |
| CF-off seed 23 | 0.881871 | 0.925737 | +0.043866 | 0.924697 | 0.926771 | 0.927090 | 排除严格消融 |
| Treatment-off seed 23 | 0.881871 | 0.713592 | -0.168279 | 0.707881 | 0.712508 | 0.727824 | 排除严格消融 |
| Treatment-shuffled seed 23 | 0.881871 | 0.925737 | +0.043866 | 0.924697 | 0.926771 | 0.927090 | 排除严格消融 |
| Treatment-off seed 41 | 0.711190 | 0.518702 | -0.192488 | 0.512444 | 0.520421 | 0.530313 | 可作 matched 消融 |

## 超参数对齐判定

1. Full seed 23/7/41 的训练参数除 seed 外一致，训练 fraction 均为 `0.25`。
   seed 41 config 中原生 `eval_residual_multiplier=0.03`，本轮 test 明确覆盖为统一的
   `m=0.75`；这是评价协议覆盖，不改变 checkpoint。
2. Treatment-off seed 41 与 Full seed 41 的训练参数一致，唯一目标差异为
   `lambda_treat: 0.1 -> 0.0`，可作为当前严格 matched 消融。
3. seed 23 的 CF-off、Treatment-off、Treatment-shuffled 训练 fraction 均为 `0.005`，
   而 Full seed 23 为 `0.25`。虽然本轮评价协议完全一致，但训练协议不一致，不能作为
   “除对照项外其他超参数全部对齐”的严格单因素消融。
4. 评测无法修复 checkpoint 已固化的训练超参数。若论文需要 seed 23 的完整严格消融，
   必须以 Full seed 23 的训练配置（含 `fraction=0.25`）重新训练对应消融 checkpoint。

## 结果位置

核心 test 的每次独立运行均位于：

`results/WN18RR/checkpoint_only_reevaluation_20261110/core/`

每个运行目录包含 `metrics.json`、`resolved_eval_config.json`、`run_manifest.json`、
`command.txt` 和 `stdout.log`。manifest 保存 checkpoint、config、evaluator、数据和 cache
的绝对路径与 SHA256。
