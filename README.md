# Retail Sales Forecasting & Analytics

## Overview
This project delivers an end-to-end retail analytics workflow using the Kaggle Store Sales dataset. It covers data ingestion, cleaning, feature engineering, multi-table joins, KPI analysis, visual reporting outputs, and a 30-day sales forecast. The result is a portfolio-ready analytics project that can be extended into business intelligence dashboards and operational forecasting pipelines.

## Tools & Technology
- Python: pandas, numpy, matplotlib
- SQL: aggregate and trend analysis queries
- Power BI: recommended for interactive dashboarding on top of prepared outputs

## Project Workflow
1. Load source datasets from `data/`:
   - `train.csv`
   - `stores.csv`
   - `transactions.csv`
   - `oil.csv`
   - `holidays_events.csv`
2. Standardize and clean data:
   - normalize column names to snake_case
   - convert date columns to datetime
   - handle missing values and remove invalid rows
3. Engineer time-based features:
   - `year`, `month`, `day`, `day_of_week`
4. Merge datasets for unified analytics modeling:
   - train + stores on `store_nbr`
   - train + transactions on `date` + `store_nbr`
   - train + oil on `date`
5. Compute core analytics:
   - total sales
   - sales by month
   - sales by store
   - sales by product family
   - top-performing stores
6. Generate charts in `outputs/charts/`.
7. Run a simple 30-day forecasting model and save actual vs predicted results.
8. Save prepared dataset to `data/cleaned_retail_data.csv`.

## Forecasting
The pipeline uses a simple linear trend regression approach on aggregated daily sales:
- Input: daily total sales
- Output: historical fitted values + 30-day future predictions
- Deliverables:
  - `outputs/charts/forecast.png`
  - `outputs/forecast_actual_vs_predicted.csv`

This baseline model is intentionally lightweight and transparent, making it suitable as a starting point for stronger models such as Prophet, ARIMA, or gradient boosting.

## Key Insights (Expected)
- Sales often follow strong weekly and monthly seasonality patterns.
- Store-level performance varies significantly, with top stores driving a disproportionate share of revenue.
- Product family contribution is uneven, highlighting category concentration opportunities.
- External factors such as transactions volume and oil price can provide additional explanatory context for demand shifts.

## Business Recommendations
- Prioritize inventory and staffing for top-performing stores and peak sales periods.
- Build category-level promotional strategies for low-performing product families.
- Use monthly and daily trend analysis to improve replenishment planning.
- Iterate forecasting with richer features (holidays, promotions, local events) to improve accuracy.
- Deploy outputs into Power BI for stakeholder-facing monitoring and decision support.

## Repository Structure
- `data/`: raw and cleaned datasets
- `scripts/`: Python pipeline (`retail_analysis.py`)
- `outputs/`: generated analytics artifacts and forecasts
- `outputs/charts/`: saved visualizations
- `sql/`: reusable SQL analysis queries

## How To Run
From the project root:

```bash
python scripts/retail_analysis.py
```

After execution, check:
- `data/cleaned_retail_data.csv`
- `outputs/charts/`
- `outputs/forecast_actual_vs_predicted.csv`

## Portfolio Value
This project demonstrates practical skills in data engineering, analytical modeling, forecasting, SQL analytics, and business communication. It is designed to showcase an end-to-end data workflow suitable for analytics and business intelligence roles.
