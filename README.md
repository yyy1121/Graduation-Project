# 电商用户行为画像与购买倾向预测系统

本项目为毕业设计项目，基于电商用户点击、订单、商品及用户基础信息，完成 **用户画像、用户分群、购买倾向预测与可视化分析**。

系统通过 RFM 和用户行为特征构建用户画像，使用 K-Means 进行用户分群，并利用 XGBoost、LightGBM、CatBoost 等模型预测用户未来购买倾向，最终通过 FastAPI + Plotly.js 实现可视化展示。

## 主要功能

- **用户画像分析**：年龄、性别、活跃度、RFM、消费金额、购买频率、转化率
- **用户分群**：基于 RFM 和行为特征进行 K-Means 聚类
- **购买倾向预测**：预测用户未来 7 天是否发生购买，支持 LightGBM、XGBoost、CatBoost
- **行为分析**：点击与购买行为、品类转化、用户活跃趋势、商品与价格偏好
- **可视化展示**：用户画像、用户分群、购买倾向、行为路径、热销商品

## 模型效果

当前最佳模型为 **XGBoost**。

| 指标 | 结果 |
| --- | ---: |
| AUC | 0.8452 |
| AP | 0.4656 |
| Precision | 0.3690 |
| Recall | 0.6820 |
| F1 | 0.4789 |

经过特征筛选后，最终使用 **53 个特征**进行模型训练。

## 技术栈

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost / LightGBM / CatBoost
- SHAP
- FastAPI
- Plotly.js
- Streamlit

## 项目结构

```text
Graduation-Project/
├── Preprocessing.ipynb
├── Feature_Engineering.ipynb
├── purchase_pipeline.py
├── improved_modeling.py
├── prepare_dashboard_data.py
├── backend_api.py
├── dashboard.py
├── static/
│   └── dashboard.html
├── 图表/
├── requirements_viz.txt
├── 项目开发文档.md
└── 项目可行性分析报告.md
```

## 运行项目

### 1. 安装依赖

```bash
pip install -r requirements_viz.txt
```

### 2. 数据预处理

依次运行：

```text
Preprocessing.ipynb
Feature_Engineering.ipynb
```

### 3. 训练模型

```bash
python improved_modeling.py
```

### 4. 生成可视化数据

```bash
python prepare_dashboard_data.py
```

### 5. 启动系统

```bash
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

浏览器访问：

```text
http://localhost:8000
```

API 文档：

```text
http://localhost:8000/docs
```

## 项目流程

```text
原始数据
  ↓
数据清洗
  ↓
特征工程
  ↓
用户画像 / K-Means 分群
  ↓
购买倾向模型
  ↓
模型评估与 SHAP 分析
  ↓
FastAPI + Plotly.js 可视化
```

## 详细文档

更详细的项目设计和实现说明可查看：

- [项目开发文档](./项目开发文档.md)
- [项目可行性分析报告](./项目可行性分析报告.md)

## 项目说明

本项目主要用于毕业设计及电商用户行为分析、用户画像和购买倾向预测相关学习研究。
