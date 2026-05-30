# API

## 概览

- 模块路径：`fieldqkit.api`
- 模块定位：统一硬件执行 API（入口客户端 + 后端发现 + 任务抽象 + provider 适配）

## 页面导航

- [QuantumHardwareClient](./QuantumHardwareClient.md) —— 高层 API 入口，自动硬件选择与任务编排
- [run_with_backend](./run_with_backend.md) —— 低层执行接口，供算法层复用
- [hardware_discovery](./hardware_discovery.md) —— 硬件发现与后端选择

**深入学习：**
- [Backend](./Backend.md) —— 后端抽象、拓扑操作、硬件 profile
- [Task](./Task.md) —— 任务协议与适配器接口
- [provider_runtime](./provider_runtime.md) —— Provider Runtime 工厂
- [providers](./providers.md) —— 六个 Provider 实现（Quafu、TianYan、GuoDun、Tencent、Origin、FieldQuantum 云）
- [cqlib](./cqlib.md) —— 共享远程平台客户端
- [platform_credentials](./platform_credentials.md) —— 凭证管理

## 推荐阅读顺序

**入门用户（关注功能使用）：**
1. [QuantumHardwareClient](./QuantumHardwareClient.md) —— 了解如何调用 `run_auto()` 和 `build_circuit()`
2. [hardware_discovery](./hardware_discovery.md) —— 学习硬件选择与后端解析
3. [run_with_backend](./run_with_backend.md) —— 理解执行流程和结果处理

**进阶开发者（关注架构与扩展）：**
4. [Backend](./Backend.md) —— 拓扑图、保真度、硬件特性
5. [Task](./Task.md) —— 任务抽象与适配器设计模式
6. [provider_runtime](./provider_runtime.md) —— Provider 动态注册与初始化

**集成新 Provider：**
7. [providers](./providers.md) —— 参考现有 Quafu/TianYan/GuoDun/Tencent/Origin/FieldQuantum 实现
8. [cqlib](./cqlib.md) —— TianYan/GuoDun 共享的 HTTP 客户端层
9. [platform_credentials](./platform_credentials.md) —— 认证配置管理

## 参数详解速查

各文档中的关键方法参数说明如下：

- **QuantumHardwareClient**
  - `run_auto(...)` 的 16 个参数（不含 self）及返回 RunResult 字段 —— see [QuantumHardwareClient.md](./QuantumHardwareClient.md#%E5%8F%82%E6%95%B0)
  - `build_circuit(kind, **kwargs)` 的 4 种线路及其参数 —— see [QuantumHardwareClient.md](./QuantumHardwareClient.md#%E5%85%B3%E9%94%AE%E6%96%B9%E6%B3%95) 下的展开说明

- **hardware_discovery**
  - `discover_hardware(num_qubits, prefer_hardware)` —— see [hardware_discovery.md](./hardware_discovery.md#%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3)
  - `resolve_backend(...)` —— see [hardware_discovery.md](./hardware_discovery.md#%E8%BF%94%E5%9B%9E%E5%80%BC%E8%AF%A6%E8%A7%A3)

- **Backend**
  - `Backend(chip)` + 拓扑操作方法 —— see [Backend.md](./Backend.md#%E6%A0%B8%E5%BF%83%E7%B1%BB)

- **Task**
  - `TaskAdapter.submit_openqasm()` / `query_status()` / `fetch_result()` —— see [Task.md](./Task.md#%E5%85%B3%E9%94%AE%E6%96%B9%E6%B3%95)

- **providers**
  - Quafu/TianYan/GuoDun 特定方法 —— see [providers.md](./providers.md#%E7%89%B9%E5%AE%9A%E4%BA%8E-provider-%E7%9A%84%E6%96%B9%E6%B3%95)

## 相关页面

- [docs 索引](../README.md)
- [VQERunner.run_model](../algorithms/vqe_runner.md)
