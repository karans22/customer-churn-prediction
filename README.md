# 📉 Customer Churn Prediction

An end-to-end data science project predicting customer churn using feature engineering, model selection, hyperparameter tuning, and SHAP explainability. Includes a full EDA, feature importance analysis, and a business impact calculator.

## 🎯 Objective
Build a production-ready churn prediction model that not only maximises predictive accuracy but also provides interpretable insights — enabling business teams to take targeted retention actions.

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Modelling | Scikit-learn, XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Notebooks | Jupyter |

## 📁 Project Structure
```
customer-churn-prediction/
├── src/
│   ├── features/
│   │   ├── feature_engineering.py  # RFM, tenure bands, interaction terms
│   │   └── feature_selection.py    # Correlation, VIF, SHAP-based selection
│   ├── models/
│   │   ├── trainer.py              # Train / tune multiple models
│   │   └── evaluator.py            # Metrics, calibration, threshold tuning
│   └── evaluation/
│       └── shap_analysis.py        # SHAP waterfall, beeswarm, summary plots
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_selection.ipynb
│   └── 04_shap_explainability.ipynb
├── data/
│   ├── raw/                        # Raw input CSVs
│   └── processed/                  # Feature-engineered datasets
├── reports/
│   └── churn_analysis_report.md    # Business-readable findings
├── requirements.txt
└── README.md
```

## 📌 Feature Engineering Highlights
| Feature | Description |
|---|---|
| `tenure_band` | Binned tenure (New / Mid / Senior) |
| `revenue_per_month` | Avg monthly spend over customer lifetime |
| `product_diversity` | Number of distinct products used |
| `support_ticket_rate` | Tickets per month (frustration proxy) |
| `recency_days` | Days since last transaction |
| `contract_risk_score` | Composite score: month-to-month + no auto-renew |

## 📊 Model Comparison
| Model | AUC-ROC | F1 | Precision@0.4 | Recall@0.4 |
|---|---|---|---|---|
| Logistic Regression | 0.821 | 0.763 | 0.74 | 0.79 |
| Random Forest | 0.871 | 0.812 | 0.80 | 0.82 |
| XGBoost | 0.891 | 0.838 | 0.83 | 0.85 |
| **LightGBM (tuned)** | **0.903** | **0.851** | **0.84** | **0.86** |

## 💼 Business Impact
With 10,000 customers and 15% churn rate:
- **Without model**: All 1,500 churners cost ~$2.25M in lost revenue
- **With model (top 300 flagged)**: Targeted retention saves ~$820K (36% recovery)
- **ROI on retention campaign**: 4.1x (assuming $50/customer intervention cost)

## 🚀 How to Run
```bash
git clone https://github.com/karans22/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

## 👤 Author
**Karan S** | Aspiring Data Scientist  
[LinkedIn](https://linkedin.com/in/karans22) | [GitHub](https://github.com/karans22)
