# run_qaoa

## 概览

- **模块**：`quantum_hw.algorithms.qaoa`
- **作用**：QAOA 优化接口，支持 MaxCut 与 custom Z/ZZ 代价项。

## 问题模式

- `problem="maxcut"`：需提供 `edges`，可选 `weights`
- `problem="custom"`：需提供 `terms`（及可选 `constant`）

## 返回

- `QAOAResult`
  - `best_cost`
  - `best_params`
  - `cost_history`

## 待补

- 参数 `p` 与可训练参数维度关系
- `callback` 与迭代日志格式
