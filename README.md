# 机器学习多因子选股：最小研究范式

这是一个可运行、可复用的教学型范式，用少量代表性量价因子演示完整研究链路：

`OHLCV 面板数据 → 时序因子 → 时间切分 → 机器学习预测 → 截面 Top-N → 简化回测`

> 本仓库只公开研究范式，并非完整生产系统。示例输出不构成投资建议，也不代表任何实际策略表现。

## 开源边界

本公开版本包含：

- 6 个代表性量价因子与基础时序算子；
- 避免随机打乱的训练/验证/测试时间切分；
- 一个基于 `HistGradientBoostingRegressor` 的回归基线；
- 按预测分数选择 Top-N、softmax 配权和简化的周期收益统计；
- 完全由程序生成的合成行情示例和最小测试。

以下内容有意不公开：

- 完整因子库、实际因子组合、筛选阈值和生产模型参数；
- 真实股票池、行情数据、交易信号、回测结果及持仓记录；
- 数据库、云服务、消息机器人、自动交易、部署和客户交付代码；
- 任何账号、密钥、Webhook、内网/本机路径或个人联系方式。

因此，本项目适合学习流程、验证接口和二次开发，不应被理解为可直接用于实盘的完整系统。

## 快速开始

需要 Python 3.9 或更高版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.run_demo
```

运行测试：

```bash
python -m pytest
```

示例只使用固定随机种子生成的合成数据，不下载、不上传、也不连接任何外部服务。

## 数据接口

`multifactor_demo.calculate_factor_frame` 接收长表格式的 `pandas.DataFrame`，至少包含：

| 列名 | 含义 |
|---|---|
| `date` | 交易日期 |
| `symbol` | 证券标识；示例使用虚构编号 |
| `open` / `high` / `low` / `close` | OHLC 价格 |
| `volume` | 成交量 |

输入必须按证券和日期唯一。实现会在每只证券内部排序并计算因子，避免跨证券串值。

## 重要的研究限制

- 示例标签是未来若干期收益，仅用于监督学习演示；生产研究必须额外处理停牌、涨跌停、复权、交易成本和可成交性。
- 简化回测直接汇总标签周期收益；当持有期大于 1 时可能存在重叠区间，不能当作严格的逐日资金曲线。
- 示例没有处理行业/市值中性化、幸存者偏差、数据发布时间和样本外滚动训练。
- 历史回测和模型分数均不保证未来收益。

## 隐私与安全

仓库不需要 `.env` 或任何凭据。`.gitignore` 默认排除常见密钥文件、数据集、模型和研究产物。提交前建议至少执行：

```bash
git grep -n -I -E "(password|secret|token|api[_-]?key|webhook|private[_-]?key)"
git status --short
```

若发现安全问题，请通过本仓库的 GitHub Security Advisory 私下报告，不要在公开 Issue 中粘贴敏感值。

## 许可证

[MIT](LICENSE)
