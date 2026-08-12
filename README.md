# 机器学习多因子选股研究框架

这是一个可运行、可测试、适合二次开发的教学型研究框架，展示从 OHLCV 面板数据到样本外信号与成本回测的完整链路：

`数据校验 → 时序因子 → 截面预处理 → 因子诊断 → 防泄漏切分 → 机器学习 → Top-N 信号 → 成本回测`

> 本仓库公开的是研究范式，不是完整生产系统。示例结果来自合成数据，不构成投资建议，也不代表任何实际策略表现。

## 项目截图

<div align="center">

<img width="48%" height="auto" alt="量化投研系统截图 1" src="https://github.com/user-attachments/assets/043c0340-7e5d-4cf7-97c8-dc663d4c0e62" />

<img width="48%" height="auto" alt="量化投研系统截图 2" src="https://github.com/user-attachments/assets/c556c11e-0a76-4d5d-ae6a-cd96ff9ce1fd" />
<img width="48%" height="auto" alt="量化投研系统截图 3" src="https://github.com/user-attachments/assets/587865b1-9123-4e8a-9586-d603fdb015e0" />
<img width="48%" height="auto" alt="量化投研系统截图 4" src="https://github.com/user-attachments/assets/f6f297b3-c2e5-44a0-ac91-e3ce5f7168f1" />

<img width="48%" height="auto" alt="量化投研系统截图 5" src="https://github.com/user-attachments/assets/32809db5-e655-4b92-9216-17329c0c9a7c" />
</div>

图片展示的是原项目产品形态；本仓库公开代码仅提供研究范式。图片仅供参考，不构成投资建议。

## 为什么这个版本有实际参考价值

它不只是一个模型调用示例，而是覆盖了多因子研究中容易做错的关键环节：

- 每只证券内部独立计算时序因子，防止跨证券串值；
- 每个交易日独立进行 MAD 去极值和截面 z-score 标准化；
- 同时计算日度 IC、RankIC、ICIR、正 IC 比例和分组收益；
- 时间切分时根据 `target_date` 清除跨越边界的标签，降低前视泄漏；
- 支持固定训练/验证/测试切分与 expanding/rolling walk-forward；
- 回测输出持仓、换手率、交易成本、净收益、累计收益和最大回撤；
- 示例完全使用固定随机种子生成的合成行情，可离线复现。

## 公开内容

### 1. 24 个通用量价因子

公开因子覆盖以下类别：

| 类别 | 示例 |
|---|---|
| 趋势与反转 | 多周期反转、均线偏离、动量衰减 |
| 成交量 | 短长周期量比、成交量 z-score、量能冲击 |
| 量价关系 | VWAP 偏离、量价相关性、成交量加权收益 |
| 波动与风险 | ATR、收益波动率、极端收益频率 |
| K 线结构 | 上下影线、隔夜跳空、价格离散度 |
| 技术指标 | TRIX、时序排名、Force Index 代理 |

因子名称保留 `alpha_NNN` 接口，便于替换、扩展和批量评估。公开版本不声称这些因子在真实市场中有效。

### 2. 研究流程组件

```text
multifactor_demo/
├── factors.py          # 24 个因子、时序算子、面板数据校验
├── preprocessing.py    # 按日期 MAD 去极值与 z-score
├── evaluation.py       # IC / RankIC / ICIR / 分组收益
└── pipeline.py         # 标签、切分、滚动训练、信号与成本回测

examples/
└── run_demo.py         # 合成数据端到端示例

tests/
└── test_pipeline.py    # 数据隔离、防泄漏、评估与回测测试
```

## 开源边界

以下生产内容有意不公开：

- 完整因子库、真实入选因子组合、因子权重和准入阈值；
- 生产模型、调参结果、模型文件和实际训练配置；
- 真实股票池、行情数据、持仓、交易信号和回测产物；
- 数据库、云服务、消息机器人、自动交易和部署代码；
- 账号、密钥、Webhook、内网/本机路径及个人联系方式。

因此，本仓库可以用于学习、研究接口验证和策略原型开发，但不能直接视为可上线交易的完整系统。

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

示例不会下载、上传或连接任何外部服务。

## 输入数据

`calculate_factor_frame` 接收长表格式的 `pandas.DataFrame`：

| 列名 | 含义 | 约束 |
|---|---|---|
| `date` | 交易日期 | 可被 pandas 解析 |
| `symbol` | 证券标识 | 与日期组合后唯一 |
| `open` | 开盘价 | 正数 |
| `high` | 最高价 | 不低于其他 OHLC 值 |
| `low` | 最低价 | 不高于其他 OHLC 值 |
| `close` | 收盘价 | 正数 |
| `volume` | 成交量 | 非负数 |

框架会拒绝重复证券日期、非有限数值以及不合法的 OHLC 关系。

## 基本用法

```python
from multifactor_demo import (
    backtest_top_n,
    factor_report,
    make_supervised_dataset,
    walk_forward_predict,
)

dataset = make_supervised_dataset(panel, forward_periods=5)
diagnostics = factor_report(dataset)

scored = walk_forward_predict(
    dataset,
    min_train_dates=120,
    test_block_dates=20,
    max_train_dates=240,
)

holdings, returns, summary = backtest_top_n(
    scored,
    top_n=5,
    rebalance_every=5,
    transaction_cost_bps=10,
)
```

`target_date` 表示未来收益标签真正变为已知的日期。训练样本只有在其 `target_date` 早于测试块起点时才会进入模型。

## 输出说明

- `factor_report`：每个因子的 IC、RankIC、稳定性和正 IC 比例；
- `walk_forward_predict`：每条样本的严格样本外分数、fold 和训练截止日期；
- `latest_signal`：最新日期的 Top-N 标的、分数和归一化目标权重；
- `backtest_top_n`：逐期持仓、换手、成本、净收益、权益曲线和摘要指标。

## 研究限制

- 合成数据只用于验证代码链路，不能证明因子在真实市场有效；
- 公开模型是保守的 sklearn 基线，不是生产模型或调参结果；
- 成本回测使用未来周期收益标签，不模拟盘中成交、冲击成本、停牌或涨跌停；
- 当 `rebalance_every` 小于标签周期时会形成重叠持仓，解释结果时需要谨慎；
- 真实研究还需要处理复权、退市、幸存者偏差、数据发布时间、行业市值暴露和基准比较；
- 历史统计、IC 和回测均不保证未来收益。

## 隐私与安全

仓库不需要 `.env` 或任何凭据。`.gitignore` 默认排除常见密钥文件、数据集、模型、Notebook 和研究产物。

提交前建议执行：

```bash
git grep -n -I -E "(password|secret|token|api[_-]?key|webhook|private[_-]?key)"
git status --short
```

若发现安全问题，请通过 GitHub 仓库的 **Security → Report a vulnerability** 私下报告，不要在公开 Issue 中粘贴敏感值。

## 许可证

[MIT](LICENSE)
