# 云端高性能模拟器接入方案

## 背景

本软件包内置了轻量级本地模拟器（statevector / MPS），支持单卡 GPU 加速和 PyTorch autograd 自动微分。现需接入一个外部**多卡并行高性能模拟器**，以私有化集群部署的形式提供。

核心挑战：VQE 等变分算法的**主循环在用户本地**运行，而**模拟执行在云端**，需要在保证梯度计算效率的前提下打通这一链路。

---

## 本地客户端接入方式

### 通信机制

本地 Python 用户端通过 **HTTP REST** 与云端服务器通信，底层使用 `requests.Session`（Python `requests` 库的持久会话对象，支持连接复用）。每次调用封装为 JSON 格式的 HTTP 请求。

**开发团队需提供云端服务器 URL**

```python
# 例如：
FIELDQUANTUM_SERVER_URL="http://192.168.1.100:8765"
FIELDQUANTUM_SERVER_URL="https://sim.your-cluster.example.com"
```

---

## REST API 规范

### 端点一览

| 方法 | 路径 | 用途 |
|------|------|------|
| `GET` | `/health` | 存活探测 |
| `POST` | `/run` | 提交计算任务，立即返回 `task_id` |
| `GET` | `/task/{task_id}/status` | 轮询任务状态 |
| `GET` | `/task/{task_id}/result` | 获取计算结果 |

---

### `GET /health`

存活探测，客户端在建立连接前调用。

**响应（200）**：
```json
{"status": "ok"}
```

---

### `POST /run`

提交一个计算任务。**立即返回 `task_id`，不等待执行完成。**

请求头：`Content-Type: application/json`

#### mode = `"sample"` — 采样模式

**请求体**：
```jsonc
{
  "mode": "sample",
  "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[4];\nrx(0.31) q[0];\ncz q[0],q[1];\nmeasure q -> c;",
  "shots": 1024,
}
```

#### mode = `"expectation"` — 期望值 + 梯度模式

**请求体**：
```jsonc
{
  "mode": "expectation",
  "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\nrx(theta_0) q[0];\nry(theta_1) q[1];\ncz q[0],q[1];\nrx(theta_2) q[2];",
  "param_names": ["theta_0", "theta_1", "theta_2"],
  "param_values": [0.31, 1.72, 0.85],
  "hamiltonian": [
    {"coeff": -1.0, "pauli": "Z0 Z1"},
    {"coeff": -0.5, "pauli": "X0"}
  ],
  "shots": 8192
}
```

#### 成功响应（200）

```json
{"task_id": "550e8400-e29b-41d4-a716-446655440000"}
```

#### 错误响应（400 / 500）

```json
{"error": "<人类可读的错误描述>"}
```

> 客户端通过检测响应体中 `"error"` 键是否存在判断失败，**不仅依赖 HTTP 状态码**，两者应同时设置。

---

### `GET /task/{task_id}/status`

轮询任务执行状态。

**响应（200）**：
```json
{"task_id": "550e8400-...", "status": "running"}
```

`status` 取值及客户端映射：

| 服务端值 | 含义 | 客户端映射 |
|----------|------|-----------|
| `"pending"` | 在队列中等待 | `"Running"` |
| `"running"` | 正在执行 | `"Running"` |
| `"finished"` | 执行完成 | `"Finished"` |
| `"error"` | 执行失败 | `"Failed"` |

**任务不存在（404）**：
```json
{"error": "task not found: <task_id>"}
```

---

### `GET /task/{task_id}/result`

取回已完成任务的结果。

**`sample` 模式响应（200）**：
```json
{"counts": {"0000": 512, "0011": 302, "1100": 210}}
```

**`expectation` 模式响应（200）**：
```json
{
  "energy": -1.23,
  "expectations": {
    "Z0 Z1": -0.80,
    "X0": 0.31,
    "I": 1.0
  },
  "gradients": [0.12, -0.34, 0.07]
}
```

`gradients[i]` = $\partial E / \partial \theta_i$，顺序与请求中 `param_names` 严格一致。

**任务处于 error 状态（500）**：
```json
{"error": "<错误信息>"}
```

---

### 数据格式

采用 **OpenQASM 2.0** 作为线路序列化格式，通过请求体的 `"qasm"` 字段传入。

参数化线路中，符号参数以占位符形式保留（如 `rx(theta_0) q[0];`），实际数值通过 `param_names` / `param_values` 字段传入，服务端负责按全词匹配替换（`\btheta_0\b`）后再交给模拟器。

`hamiltonian` 为数组，每项：

| 字段 | 类型 | 含义 |
|------|------|------|
| `coeff` | `float` | 该 Pauli 项的系数 |
| `pauli` | `string` | Pauli 算符串，格式为 `"X0 Z1 Y3"`（`<算符><比特索引>` 以空格分隔）；纯恒等项用 `"I"` |

---

## 功能清单

| 优先级 | 功能 | 说明 |
|--------|------|------|
| **P0** | OpenQASM 2.0 解析 | 支持 `qelib1.inc` 标准门集|
| **P0** | 采样模拟 (`sample`) | 输入 QASM + shots，返回 counts |
| **P0** | 期望值计算 (`expectation`) | 输入 QASM + Hamiltonian，返回 energy + per-observable 期望值 |
| **P0** | 参数偏移梯度 | 输入参数化 QASM + param_values + Hamiltonian，返回 gradients |

---

## 集群部署建议

### 推荐架构

```
用户 Python 进程
  │  HTTP REST (requests.Session)
  │  由 FIELDQUANTUM_SERVER_URL 指定地址
  ▼
┌──────────────────────┐
│   API Gateway / LB   │  TLS 终止、负载均衡、（可选）认证
└──────────┬───────────┘
           │
     ┌─────▼──────┐
     │  API 服务  │  无状态，接收 POST /run 后入队立即返回
     │ (FastAPI)  │  提供 /task/* 状态与结果查询
     └─────┬──────┘
           │
     ┌─────▼──────────────┐
     │  任务队列 + 状态存储 │  Redis（推荐）：任务分发 + 结果持久化
     └─────┬──────────────┘
           │
  ┌────────┴──────────────┐
  │      Worker 集群       │  从队列取任务 → 调用模拟器 → 写回结果
  └───────────────────────┘
```

### 任务状态机

```
POST /run → [pending] → [running] → [finished]
                                 ↘ [error]
```

