# 安全政策

我们非常重视 fieldqkit 的安全。感谢你帮助我们以负责任的方式发现并修复安全问题。

## 受支持的版本

本项目处于早期阶段（`0.x`），安全修复仅针对**最新发布版本**提供。请在报告前升级到最新版本确认问题是否仍存在。

| 版本 | 是否维护安全更新 |
|------|------------------|
| 最新 `0.x` 发布版 | ✅ |
| 更早版本 | ❌ |

## 报告安全漏洞

**请不要通过公开的 GitHub Issue、Pull Request 或社交渠道披露安全漏洞**，以免在修复前暴露给潜在攻击者。

请通过以下任一私密渠道报告：

- 发送邮件至 **guoyuchen@fieldquanta.com**，邮件标题请以 `[SECURITY]` 开头；
- 或使用 GitHub 的 **[Private vulnerability reporting](https://github.com/FieldQuantum/fieldqkit/security/advisories/new)**（仓库 Security 标签页 → "Report a vulnerability"）。

报告时请尽量包含：

- 漏洞类型与影响范围；
- 复现步骤或最小可复现示例（**请勿包含任何真实 token / 凭证**）；
- 受影响的版本、模块或 provider；
- 如有，建议的修复方向。

## 凭证与密钥安全提示

fieldqkit 通过 API token 接入各量子云平台。使用时请注意：

- **切勿将真实 token 提交到版本库**。`.quantum_hw.yaml` 已在 `.gitignore` 中排除；请使用 `.quantum_hw.example.yaml` 作为模板，或通过环境变量（如 `QUAFU_API_TOKEN` 等）配置。
- 在 Issue、日志、报错堆栈、Notebook 输出中分享内容前，请确认其中不含 token 或其他敏感凭证。
- 凭证查找优先级与配置方式详见 [配置文档](docs/configuration.md)。

修复发布后，如你愿意，我们会在致谢中注明你的贡献。
