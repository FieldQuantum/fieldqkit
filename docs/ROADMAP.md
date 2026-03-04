# 快速开始与导航

## 你可以怎么写这套文档

建议按下面顺序补全：

1. 先补核心入口：`QuantumHardwareClient.run_auto`
2. 再补算法页：`run_shadow` / `run_vqe` / `run_qaoa`
3. 再补校准页：readout / RB / tomography
4. 最后补基础工具页：circuits / observables / readout / zne / result types

## 每页最小完成标准（MVP）

- 功能一句话
- 函数/类签名
- 参数说明（含默认值）
- 返回结构
- 一个最小示例
- 常见错误与注意事项（2-4 条）

## 与旧文档关系

- `DOCUMENTATION.md` 继续保留，作为“全景总览”。
- `reference/*` 作为“可索引的细粒度 API 页面”。
- 后续可在 `DOCUMENTATION.md` 每节末尾补“深入阅读”跳转链接到对应页面。
