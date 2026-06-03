# 贡献指南

感谢你对 **fieldqkit** 的关注！无论是反馈问题、完善文档，还是提交代码，我们都非常欢迎。本文档说明参与贡献的流程与约定。

---

## 反馈问题（Issues）

在 [GitHub Issues](https://github.com/FieldQuantum/fieldqkit/issues) 提交前，请先搜索是否已有相同问题。

**报告 Bug** 时，请尽量提供：

- fieldqkit 版本（`python -c "import fieldqkit; print(fieldqkit.__version__)"`）、Python 版本、操作系统；
- 最小可复现代码片段；
- 期望行为与实际行为（含完整报错堆栈）；
- 涉及的 provider / 后端（如 `simulator`、`quafu`、`tianyan` 等）。

> ⚠️ **请勿在 Issue 中粘贴任何真实的 API token 或凭证。** 如涉及安全漏洞，请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开提交。

**提交功能建议** 时，请说明使用场景与动机，便于我们评估优先级。

---

## 本地开发环境

### 1. 获取源码

```bash
git clone https://github.com/FieldQuantum/fieldqkit.git
cd fieldqkit
```

### 2. 安装（可编辑模式 + 开发依赖）

推荐使用独立的 Python 环境（venv 或 conda）。本地模拟器依赖 PyTorch，建议在带 PyTorch 的环境中开发：

```bash
pip install -e ".[sim,test]"        # 模拟器 + 测试依赖
```

> 要求 Python >= 3.10（支持 3.10 / 3.11 / 3.12）。
> 接入本源量子云的相关代码需额外安装 `pip install -e ".[origin]"`（`pyqpanda3`）。

### 3. 运行测试

```bash
pytest                       # 运行全部测试
pytest tests/test_sim.py     # 只跑某个文件
pytest -k qcis               # 按关键字筛选
```

测试套件大量覆盖模拟器与算法，**需要安装 `[sim]` 依赖组（PyTorch）才能完整通过**。提交前请确保 `pytest` 全绿。

### 4. 预览文档（可选）

```bash
mkdocs serve     # 本地 http://127.0.0.1:8000 实时预览
```

---

## 提交 Pull Request

1. 从 `main` 分支创建你的特性分支：
   ```bash
   git checkout -b fix/some-bug      # 或 feature/xxx
   ```
2. 进行修改。**新增功能或修复 Bug 时，请同时补充对应测试**；如改动了公开 API，请同步更新 `docs/` 下的相关文档。
3. 确保本地 `pytest` 全部通过。
4. 提交并推送，然后在 GitHub 上发起 Pull Request，关联相关 Issue（如有），并在描述中说明改动内容与动机。

### 代码约定

- **风格**：遵循 [PEP 8](https://peps.python.org/pep-0008/)；与所在文件的既有风格保持一致（命名、缩进、注释密度）。
- **类型注解 / docstring**：公开函数请带类型注解与 docstring（参数、返回值、可能抛出的异常），与现有代码风格一致。
- **注释与文档**：注释用于解释「为什么」，而非复述代码；文档与代码行为必须一致。
- **提交信息**：使用清晰的英文或中文祈使句，简述本次改动（如 `Fix QCIS u-gate decomposition`）。
- **改动范围**：一个 PR 聚焦一件事，便于审查。

### 量子正确性

本项目对**数值正确性**要求较高。涉及门分解、模拟器、误差缓解、端序（bit ordering）等改动时，请尽量：

- 用酉矩阵重构 / 与参考实现对比的方式编写测试（参见 `tests/test_circuit_to_qcis.py` 中的酉重构测试）；
- 明确并保持本包的端序约定（**big-endian：q[0] 在最左/最高位**）。

---

## 许可证

提交贡献即表示你同意你的贡献以本项目的 [Apache License 2.0](LICENSE) 许可证发布。

如果你在改动中引入了改编自第三方项目的代码，请在 PR 中说明，并相应更新 [NOTICE](NOTICE) 与 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。

再次感谢你的贡献！🎉
