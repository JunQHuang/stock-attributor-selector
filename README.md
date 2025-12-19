# stock-attributor-selector

本项目构建了一个面向 A 股市场的系统化股票风格分类与归因分析平台，并集成了基于 LightGBM + 77个Alpha因子的量化交易系统。

## 项目概述

本项目构建了一套完整的A股量化投研系统，涵盖三大模块：（1）基于财务因子（PE/PB/ROE/增速等）的股票风格分类与随机森林归因分析，结合LLM对研报公告进行语义解读，生成可解释的涨跌归因；（2）季频财务数据驱动的选股验证框架；（3）基于LightGBM + 77个自研因子的日频量化交易系统，采用滚动训练机制（训练数据截止信号日前7天以规避未来信息），T-1信号T日执行的严格回测框架，在2025年5-12月测试期实现夏普比率2.74、最大回撤15.90%、超额收益55.46%（vs沪深300），并集成GitHub Gist + BigQuant/MiniQMT的自动化交易链路，支持实盘部署。

---

## 03 多因子量化交易系统（日频数据）

基于 LightGBM + 77个Alpha因子的量化选股策略，支持滚动训练、自动交易。

### 策略表现

| 指标                | 数值           |
| ------------------- | -------------- |
| 策略总收益          | 83.30%         |
| 超额收益(vs沪深300) | 55.46%         |
| 夏普比率            | 2.74           |
| 最大回撤            | 15.90%         |
| Calmar              | 6.65           |
| 测试区间            | 2025年5月-12月 |

### 快速开始

```bash
# 一键更新数据 + 生成最新信号
python run_daily.py
```

### 核心模块

| 文件                  | 说明                     |
| --------------------- | ------------------------ |
| run_daily.py          | 一键运行脚本             |
| multi_factor_model.py | 主模型（训练+回测+信号） |
| alpha_factors.py      | 77个Alpha因子计算        |
| upload_signal.py      | 上传信号到GitHub Gist    |
| bigquant_strategy.py  | BigQuant策略代码         |
| auto_trade.py         | MiniQMT自动交易          |

### 模型训练策略

**数据集划分**：

- 训练集：2020-2023年
- 验证集：2024年 + 2025年1-4月
- 测试集：2025年5月后

**调参 vs 回测/实盘**：

| 阶段      | 训练方式   | 说明                                |
| --------- | ---------- | ----------------------------------- |
| 调参      | 固定训练集 | 用2020-2023数据训练，在验证集上调参 |
| 回测/实盘 | 滚动训练   | 每个信号日用该日之前的数据重新训练  |

**防止未来信息泄露**：

- Label = shift(-6) / shift(-1) - 1，需要6天后的价格
- 滚动训练cutoff = 信号日 - 7天，确保label完整
- T-1日生成信号 → T日执行，无前视偏差

### 调仓规则

- 每 **5个交易日** 调仓一次
- 持仓 **15只** 股票
- 按预测值开根号分配仓位

### 自动交易（BigQuant）


##1 股票归因分析（年频数据）

✅ **环境准备与数据库连接**

- 引入库：pandas, numpy, matplotlib, sqlalchemy, re, warnings
- 配置：设置中文字体 SimHei，创建 MySQL 数据连接 connection

✅ **年度财务样本构建**

- 函数：`build_financial_samples(years, connection)`
- 内容：提取各股票市值、市盈率（PE）、市净率（PB）及财报核心指标
- 时间点：每年4月30日后首个交易日为采样时间

✅ **市值过滤**

- 函数：`filter_by_market_cap(samples, drop_percent=0.3)`
- 内容：剔除市值底部30%的小盘股

✅ **ROE三年滚动均值与波动率**

- 函数：`calculate_3y_rolling_roe()`
- 清洗极值：`clean_roe_by_iqr()`
- 稳定性阈值：`compute_roe_stability_threshold()`
- 输出：ROE_3Y_AVG, ROE_3Y_STD

✅ **股票风格分类（成长 / 价值 / 其他）**

- 函数：`classify_and_label_stocks()`
- 分类依据：PE、PB、ROE波动、净利润同比、收入同比
- 标签说明：
  - 1: 高估值 + 高增长 → 成长股
  - 0: 低估值 + 稳定ROE → 价值股
  - -1: 其他

✅ **相对收益计算**

- 函数：`calculate_relative_stock_returns()`
- 内容：计算个股相对中证全指的年内表现：RELATIVE_TO_MARKET

✅ **样本合并与标签整理**

- 函数：`merge_stock_return_with_labels()`
- 输出：包含收益 + 风格标签的训练样本（X + Y）

✅ **极端样本选取（用于归因）**

- 函数：`select_extreme_performance_stocks(top_n=50)`
- 按"涨跌 + 风格"四象限选取代表性个股，打上 PERFORMANCE_TAG

✅ **财务因子派生与差值计算**

- 函数：`build_detailed_financial_samples()`
- 内容：拼接当前期与上一期数据，生成差值（_DIFF）与同比（_YOY）

✅ **报告纪要与研报合并**

- 函数：`merge_meeting_content_batch_fast()`
- 来源：券商研报（字段：report_extract_text）、公司公告纪要（字段：meeting_content）

✅ **行业信息合并**

- 函数：`get_basicinfo_all()`
- 内容：每只股票匹配申万一级行业 INDUSTRY_SW_I

✅ **多维归因分析（行业 + 风格 + 涨跌）**

- 函数：`analyze_by_group_and_industry()`
- 维度：PERFORMANCE_TAG × INDUSTRY_SW_I × label

✅ **随机森林归因强度评估**

- 函数：`compute_feature_importance_by_style()`
- 内容：按风格训练 RandomForestClassifier，输出因子重要性评分图

✅ **AI自然语言解释**

- 函数：`ask_ai()`
- 内容：基于 combined_content + 财务数据 + 风格标签，自动生成上涨/下跌原因

---

## 02 选股验证分析（季频数据）

✅ **季度财务样本构建**

- 函数：`build_financial_samples_q()`
- 时间：以3月31日为基准
- 提取：净利润、营收、研发、现金流等指标的季度同比增速

✅ **季度ROE均值与波动**

- 函数：`calculate_3y_rolling_roe_q()`

✅ **财务宽表拼接 + 清洗**

- 包括市值过滤、异常值处理、估值指标、同比增速、差值等统一合并

✅ **自然语言解释生成以及选股结果（季度）**

- 函数：`ask_ai()`
- 内容：结合文本与因子输出解释，供投资判断参考

---

## 联系方式

详细结果与文件需要请联系：qq1006775897@163.com
