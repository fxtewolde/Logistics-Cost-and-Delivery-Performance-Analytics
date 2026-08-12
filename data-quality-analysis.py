import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. LOAD DATA
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

print(df.shape)
print(df.info())
print(df.head())

## Data Processing and Analysis
df.isna().sum().sort_values(ascending=False)
df.duplicated().sum()
df.describe()


## Cleaning Dataset
analysis_cols = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Delivery Status",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

df_clean = df[analysis_cols].copy()

df_clean = df_clean.rename(columns={
    "Days for shipping (real)": "actual_shipping_days",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Late_delivery_risk": "late_delivery_risk",
    "Delivery Status": "delivery_status",
    "Shipping Mode": "shipping_mode",
    "Order Region": "order_region",
    "Order Country": "order_country",
    "Order Item Total": "order_value",
    "Order Profit Per Order": "profit_per_order",
    "Order Item Profit Ratio": "profit_margin"
})

# Drop exact duplicate rows found during the data quality check above
df_clean = df_clean.drop_duplicates()

# Convert date columns from text to real datetime objects so we can group by month/year
df_clean["order date (DateOrders)"] = pd.to_datetime(df_clean["order date (DateOrders)"])
df_clean["shipping date (DateOrders)"] = pd.to_datetime(df_clean["shipping date (DateOrders)"])


# 2. FEATURE ENGINEERING
# delay_days: positive = late, negative = early, 0 = exactly on schedule
df_clean["delay_days"] = df_clean["actual_shipping_days"] - df_clean["scheduled_shipping_days"]

# on_time flag: 1 if delivered on or before the scheduled number of days, else 0
df_clean["on_time"] = (df_clean["delay_days"] <= 0).astype(int)

# order_month: groups every order into a "2025-06" style bucket for trend analysis
df_clean["order_month"] = df_clean["order date (DateOrders)"].dt.to_period("M").astype(str)


# 3. KPI 1 — ON-TIME DELIVERY RATE BY REGION AND SHIPPING MODE
#    (Data Analysis and Reporting)
otd_by_region = (
    df_clean.groupby("order_region")["on_time"]
    .mean()
    .mul(100)
    .round(1)
    .sort_values()
    .reset_index(name="on_time_pct")
)

otd_by_mode = (
    df_clean.groupby("shipping_mode")["on_time"]
    .mean()
    .mul(100)
    .round(1)
    .sort_values()
    .reset_index(name="on_time_pct")
)


# 4. KPI 2 — AVERAGE DELAY DAYS BY REGION
#    (Data Analysis and Reporting)
delay_by_region = (
    df_clean.groupby("order_region")["delay_days"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
    .reset_index(name="avg_delay_days")
)


# 5. KPI 3 — COST / PROFIT PERFORMANCE BY REGION
#    (Cost Management)
cost_profit_by_region = (
    df_clean.groupby("order_region")
    .agg(
        avg_order_value=("order_value", "mean"),
        avg_profit_per_order=("profit_per_order", "mean"),
        avg_profit_margin=("profit_margin", "mean"),
        total_orders=("order_value", "count")
    )
    .round(2)
    .sort_values("avg_profit_per_order")
    .reset_index()
)


# 6. KPI 4 — MONTHLY TREND
#    (System construction / report system)
monthly_trend = (
    df_clean.groupby("order_month")
    .agg(
        total_orders=("order_value", "count"),
        avg_order_value=("order_value", "mean"),
        on_time_pct=("on_time", "mean"),
        avg_delay_days=("delay_days", "mean")
    )
    .round(2)
    .reset_index()
)
monthly_trend["on_time_pct"] = (monthly_trend["on_time_pct"] * 100).round(1)


# 7. RISK FLAGGING — ANOMALY DETECTION
#    (Risk Management: "proactively identify and assess potential risks")
# Compute on-time % for every region-month combination
region_month_otd = (
    df_clean.groupby(["order_region", "order_month"])["on_time"]
    .mean()
    .mul(100)
    .reset_index(name="on_time_pct")
)

mean_otd = region_month_otd["on_time_pct"].mean()
std_otd = region_month_otd["on_time_pct"].std()

# Flag any region-month where on-time % falls more than 1.5 standard deviations
# below the network-wide average — i.e., a real anomaly, not normal noise
region_month_otd["risk_flag"] = np.where(
    region_month_otd["on_time_pct"] < (mean_otd - 1.5 * std_otd),
    "HIGH RISK",
    "Normal"
)

high_risk_periods = region_month_otd[region_month_otd["risk_flag"] == "HIGH RISK"]


# 8. EXPORT SUMMARY TABLES
#    These feed directly into your Excel/Power BI dashboard
otd_by_region.to_csv("kpi_on_time_by_region.csv", index=False)
otd_by_mode.to_csv("kpi_on_time_by_shipping_mode.csv", index=False)
delay_by_region.to_csv("kpi_avg_delay_by_region.csv", index=False)
cost_profit_by_region.to_csv("kpi_cost_profit_by_region.csv", index=False)
monthly_trend.to_csv("kpi_monthly_trend.csv", index=False)
high_risk_periods.to_csv("risk_flagged_periods.csv", index=False)

print("All KPI summary CSVs exported.")
print(f"\n{len(high_risk_periods)} high-risk region-month combinations flagged.")
print(high_risk_periods)


# 9. QUICK VISUAL CHECKS
plt.figure(figsize=(8, 4))
plt.bar(otd_by_region["order_region"], otd_by_region["on_time_pct"])
plt.title("On-Time Delivery % by Region")
plt.ylabel("On-Time %")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("chart_on_time_by_region.png")
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(monthly_trend["order_month"], monthly_trend["on_time_pct"], marker="o")
plt.title("On-Time Delivery % Trend by Month")
plt.ylabel("On-Time %")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("chart_on_time_trend.png")
plt.close()

print("Charts saved.")


