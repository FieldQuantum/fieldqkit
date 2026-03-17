# mpo process simulator

## 模块

- quantum_hw.sim.mpo

## 概览

该模块提供“过程（算符）级”模拟：

- 输入：量子线路
- 输出：线路整体酉变换的 MPO 表示

不负责：

- observable 期望计算
- counts 采样

它的定位是过程张量演化/结构研究，而不是测量统计接口。

## 张量形状约定

- 单站点 MPO 张量：T[Dl, pout, Dr, pin]
- 维度含义：
  - Dl/Dr: 左右虚拟键
  - pin: 输入物理腿
  - pout: 输出物理腿

## 关键函数

### simulate_mpo_process(qc, *, param_values=None, max_bond_dim=None, device=None) -> List[torch.Tensor]

- 初始过程是单位算符 I^{\otimes n} 的 MPO。
- 对每个门执行“左乘”更新（U <- G U）。
- 支持：
  - 固定门：one_qubit/two_qubit/three_qubit
  - 参数门：one_qubit_param/two_qubit_param
  - 功能门：barrier/measure/delay 跳过
- reset 不支持：
  - reset 是非酉过程，本函数会抛错。

### max_bond_dim 行为

- None: 不做显式截断
- int: 对脏区间按需执行压缩 sweep，并将键维限制在给定上界

## 算法要点

- 单比特门：直接作用于每个站点的输出腿
- 多比特门：
  - 先把门分解为局部 gate-MPO
  - 对非相邻比特插入 identity bridge
  - 再与当前过程 MPO 在区间内逐站点合并
- 压缩流程复用 MPS 模块的 SVD 与区间管理工具

## 返回结果使用建议

- 返回值是站点张量列表，不是稠密矩阵。
- 若需要与稠密参考比对，可先把 MPO 全收缩为 (2^n, 2^n) 矩阵。

## 测试

- tests/test_sim_mpo.py 覆盖：
  - 与稠密参考酉矩阵一致性
  - max_bond_dim 上界约束

## 相关页面

- mps simulator（mps.md）
- simulator interface（interface.md）
