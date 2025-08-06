# stock-attributor-selector
本项目构建了一个面向 A 股市场的系统化股票风格分类与归因分析平台。 通过提取财务报表中的横截面因子（如市盈率 PE、市净率 PB、ROE、净利润/营收同比增长等），结合年报、季报、研报与公告纪要等文本数据，系统将股票划分为高估值成长型与低估值价值型，并集成机器学习与自然语言处理方法，进行多维度涨跌归因分析与选股预测验证。  项目采用随机森林模型对各类股票进行归因，量化主要财务因子的解释力，同时侧重于通过大模型（如 GPT/Claude）对企业公告与研报内容进行语义理解，自动生成个股上涨/下跌的自然语言解释，从而提高策略的可解释性与投资实用性。最后成功站在2025.03.01节点没有未来信息的情况下，选出诸如：胜宏科技、英科医疗、大博医疗、太辰光等优质牛股。


01 股票归因分析（年频数据）
✅ 环境准备与数据库连接
引入库：pandas, numpy, matplotlib, sqlalchemy, re, warnings

配置：设置中文字体 SimHei，创建 MySQL 数据连接 connection

✅ 年度财务样本构建
函数：build_financial_samples(years, connection)

内容：提取各股票市值、市盈率（PE）、市净率（PB）及财报核心指标

时间点：每年4月30日后首个交易日为采样时间

✅ 市值过滤
函数：filter_by_market_cap(samples, drop_percent=0.3)

内容：剔除市值底部30%的小盘股

✅ ROE三年滚动均值与波动率
函数：calculate_3y_rolling_roe()

清洗极值：clean_roe_by_iqr()

稳定性阈值：compute_roe_stability_threshold()

输出：ROE_3Y_AVG, ROE_3Y_STD

✅ 股票风格分类（成长 / 价值 / 其他）
函数：classify_and_label_stocks()

分类依据：PE、PB、ROE波动、净利润同比、收入同比

标签说明：

1: 高估值 + 高增长 → 成长股

0: 低估值 + 稳定ROE → 价值股

-1: 其他

✅ 相对收益计算
函数：calculate_relative_stock_returns()

内容：计算个股相对中证全指的年内表现：RELATIVE_TO_MARKET

✅ 样本合并与标签整理
函数：merge_stock_return_with_labels()

输出：包含收益 + 风格标签的训练样本（X + Y）

✅ 极端样本选取（用于归因）
函数：select_extreme_performance_stocks(top_n=50)

按“涨跌 + 风格”四象限选取代表性个股，打上 PERFORMANCE_TAG

✅ 财务因子派生与差值计算
函数：build_detailed_financial_samples()

内容：拼接当前期与上一期数据，生成差值（_DIFF）与同比（_YOY）

✅ 报告纪要与研报合并
函数：merge_meeting_content_batch_fast()

来源：

券商研报（字段：report_extract_text）

公司公告纪要（字段：meeting_content）

✅ 行业信息合并
函数：get_basicinfo_all()

内容：每只股票匹配申万一级行业 INDUSTRY_SW_I

✅ 多维归因分析（行业 + 风格 + 涨跌）
函数：analyze_by_group_and_industry()

维度：PERFORMANCE_TAG × INDUSTRY_SW_I × label

✅ 随机森林归因强度评估
函数：compute_feature_importance_by_style()

内容：按风格训练 RandomForestClassifier，输出因子重要性评分图

✅ AI自然语言解释
函数：ask_ai()

内容：基于 combined_content + 财务数据 + 风格标签，自动生成上涨/下跌原因

02 选股验证分析（季频数据）
✅ 季度财务样本构建
函数：build_financial_samples_q()

时间：以3月31日为基准

提取：净利润、营收、研发、现金流等指标的季度同比增速

✅ 季度ROE均值与波动
函数：calculate_3y_rolling_roe_q()

✅ 财务宽表拼接 + 清洗
包括市值过滤、异常值处理、估值指标、同比增速、差值等统一合并

✅ 自然语言解释生成以及选股结果（季度）
函数：ask_ai()

内容：结合文本与因子输出解释，供投资判断参考




