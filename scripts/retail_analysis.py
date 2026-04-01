"""Retail Sales Forecasting & Analytics pipeline.

This script loads the Kaggle Store Sales dataset files, performs cleaning and feature
engineering, merges core tables, computes business analytics, creates charts, runs a
simple 30-day forecast, and saves outputs for downstream BI/reporting.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"


REQUIRED_FILES = {
    "train": DATA_DIR / "train.csv",
    "stores": DATA_DIR / "stores.csv",
    "transactions": DATA_DIR / "transactions.csv",
    "oil": DATA_DIR / "oil.csv",
    "holidays_events": DATA_DIR / "holidays_events.csv",
}


PALETTE = {
    "blue": "#2D6CDF",
    "teal": "#12B3A8",
    "orange": "#FF8A3D",
    "violet": "#735DFF",
    "pink": "#FF5FA2",
    "slate": "#334155",
    "gray": "#94A3B8",
}


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_chart_style() -> None:
    """Apply consistent visual style for cleaner and more colorful charts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1E293B",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E1",
        }
    )


def format_axis_in_millions(ax: plt.Axes) -> None:
    """Format y-axis in millions for readability on high-value sales charts."""
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake_case."""
    renamed = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    return df.rename(columns=renamed)


def print_df_overview(name: str, df: pd.DataFrame) -> None:
    """Print head, columns, and info for a dataframe."""
    print(f"\n{'=' * 80}\n{name.upper()} - HEAD\n{'=' * 80}")
    print(df.head())

    print(f"\n{name.upper()} - COLUMNS")
    print(df.columns.tolist())

    print(f"\n{name.upper()} - INFO")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())


def load_data() -> Dict[str, pd.DataFrame]:
    """Load required CSV files and print initial structure diagnostics."""
    missing = [str(path) for path in REQUIRED_FILES.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Missing required dataset files. Please place files at:\n" + missing_text
        )

    datasets = {
        name: pd.read_csv(path)
        for name, path in REQUIRED_FILES.items()
    }

    for name, df in datasets.items():
        print_df_overview(name, df)

    return datasets


def clean_dataframes(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean all input dataframes with shared and table-specific logic."""
    cleaned: Dict[str, pd.DataFrame] = {}

    for name, df in datasets.items():
        df = standardize_column_names(df)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
            else:
                if df[col].isna().any():
                    mode_vals = df[col].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
                    df[col] = df[col].fillna(fill_val)

        if "date" in df.columns:
            df = df.dropna(subset=["date"])

        if name == "train" and "sales" in df.columns:
            df = df[df["sales"].notna()]
            df = df[df["sales"] >= 0]

        if name == "transactions" and "transactions" in df.columns:
            df = df[df["transactions"].notna()]
            df = df[df["transactions"] >= 0]

        cleaned[name] = df

    if "oil" in cleaned and "dcoilwtico" in cleaned["oil"].columns:
        oil_df = cleaned["oil"].copy()
        oil_df = oil_df.sort_values("date")
        oil_df["dcoilwtico"] = oil_df["dcoilwtico"].replace(0, np.nan)
        oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
        cleaned["oil"] = oil_df

    return cleaned


def add_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features to the training data."""
    if "date" not in train_df.columns:
        raise KeyError("Column 'date' was not found in training data after cleaning.")

    train_df = train_df.copy()
    train_df["year"] = train_df["date"].dt.year
    train_df["month"] = train_df["date"].dt.month
    train_df["day"] = train_df["date"].dt.day
    train_df["day_of_week"] = train_df["date"].dt.dayofweek
    return train_df


def merge_datasets(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge train with stores, transactions, and oil as requested."""
    train_df = cleaned["train"]
    stores_df = cleaned["stores"]
    transactions_df = cleaned["transactions"]
    oil_df = cleaned["oil"]

    merged = train_df.merge(stores_df, on="store_nbr", how="left", suffixes=("", "_store"))
    merged = merged.merge(
        transactions_df,
        on=["date", "store_nbr"],
        how="left",
        suffixes=("", "_txn"),
    )
    merged = merged.merge(oil_df[["date", "dcoilwtico"]], on="date", how="left")

    if "transactions" in merged.columns:
        merged["transactions"] = merged["transactions"].fillna(0)

    if "dcoilwtico" in merged.columns:
        merged["dcoilwtico"] = merged["dcoilwtico"].ffill().bfill()

    return merged


def run_analysis(merged: pd.DataFrame) -> Dict[str, pd.DataFrame | float]:
    """Compute required business KPIs and aggregations."""
    analytics: Dict[str, pd.DataFrame | float] = {}

    total_sales = float(merged["sales"].sum())
    sales_by_month = (
        merged.groupby(["year", "month"], as_index=False)["sales"].sum()
        .sort_values(["year", "month"])
    )
    sales_by_store = (
        merged.groupby("store_nbr", as_index=False)["sales"].sum()
        .sort_values("sales", ascending=False)
    )

    family_col = "family" if "family" in merged.columns else None
    if family_col:
        sales_by_family = (
            merged.groupby(family_col, as_index=False)["sales"].sum()
            .sort_values("sales", ascending=False)
        )
    else:
        sales_by_family = pd.DataFrame(columns=["family", "sales"])

    top_stores = sales_by_store.head(10)

    analytics["total_sales"] = total_sales
    analytics["sales_by_month"] = sales_by_month
    analytics["sales_by_store"] = sales_by_store
    analytics["sales_by_family"] = sales_by_family
    analytics["top_stores"] = top_stores

    print(f"\nTotal Sales: {total_sales:,.2f}")
    print("\nTop-performing Stores:")
    print(top_stores)

    return analytics


def create_visualizations(
    daily_sales: pd.DataFrame,
    sales_by_store: pd.DataFrame,
    sales_by_family: pd.DataFrame,
) -> None:
    """Generate and save required charts to outputs/charts."""
    apply_chart_style()

    fig, ax = plt.subplots(figsize=(13, 6.5))
    rolling_30 = daily_sales["sales"].rolling(30, min_periods=1).mean()
    ax.plot(daily_sales["date"], daily_sales["sales"], linewidth=1.4, color=PALETTE["blue"], alpha=0.35, label="Daily Sales")
    ax.plot(daily_sales["date"], rolling_30, linewidth=2.4, color=PALETTE["teal"], label="30-Day Trend")
    ax.set_title("Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    format_axis_in_millions(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "sales_over_time.png", dpi=150)
    plt.close()

    store_top = sales_by_store.head(20)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    store_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(store_top)))
    ax.bar(store_top["store_nbr"].astype(str), store_top["sales"], color=store_colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Top 20 Stores by Sales")
    ax.set_xlabel("Store Number")
    ax.set_ylabel("Sales")
    format_axis_in_millions(ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "sales_by_store.png", dpi=150)
    plt.close()

    if not sales_by_family.empty:
        family_top = sales_by_family.head(15)
        fig, ax = plt.subplots(figsize=(13, 7.5))
        family_rev = family_top.iloc[::-1]
        category_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(family_rev)))
        ax.barh(family_rev["family"], family_rev["sales"], color=category_colors)
        ax.set_title("Top Product Families by Sales")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Product Family")
        format_axis_in_millions(ax)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "sales_by_category.png", dpi=150)
        plt.close()


def create_dashboard(
    daily_sales: pd.DataFrame,
    sales_by_store: pd.DataFrame,
    sales_by_family: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    """Create a single dashboard image that combines key visual insights."""
    apply_chart_style()

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Retail Sales Analytics Dashboard", fontsize=20, fontweight="bold", color="#0F172A")

    rolling_30 = daily_sales["sales"].rolling(30, min_periods=1).mean()
    axs[0, 0].plot(daily_sales["date"], daily_sales["sales"], color=PALETTE["blue"], alpha=0.25, linewidth=1.2)
    axs[0, 0].plot(daily_sales["date"], rolling_30, color=PALETTE["teal"], linewidth=2.3)
    axs[0, 0].set_title("Sales Trend Over Time")
    axs[0, 0].set_xlabel("Date")
    axs[0, 0].set_ylabel("Sales")
    format_axis_in_millions(axs[0, 0])

    store_top = sales_by_store.head(12)
    store_colors = plt.cm.coolwarm(np.linspace(0.15, 0.95, len(store_top)))
    axs[0, 1].bar(store_top["store_nbr"].astype(str), store_top["sales"], color=store_colors)
    axs[0, 1].set_title("Top 12 Stores by Sales")
    axs[0, 1].set_xlabel("Store")
    axs[0, 1].set_ylabel("Sales")
    axs[0, 1].tick_params(axis="x", rotation=40)
    format_axis_in_millions(axs[0, 1])

    if not sales_by_family.empty:
        family_top = sales_by_family.head(10).iloc[::-1]
        family_colors = plt.cm.magma(np.linspace(0.2, 0.9, len(family_top)))
        axs[1, 0].barh(family_top["family"], family_top["sales"], color=family_colors)
        axs[1, 0].set_title("Top 10 Product Families")
        axs[1, 0].set_xlabel("Sales")
        axs[1, 0].set_ylabel("Family")
        format_axis_in_millions(axs[1, 0])
    else:
        axs[1, 0].text(0.5, 0.5, "No family data available", ha="center", va="center")
        axs[1, 0].set_axis_off()

    hist = forecast_df[forecast_df["is_forecast"] == False]
    fut = forecast_df[forecast_df["is_forecast"] == True]
    axs[1, 1].plot(hist["date"], hist["actual_sales"], color=PALETTE["blue"], linewidth=1.7, label="Actual")
    axs[1, 1].plot(forecast_df["date"], forecast_df["predicted_sales"], color=PALETTE["orange"], linewidth=2.2, linestyle="--", label="Predicted")
    if not fut.empty:
        axs[1, 1].axvspan(fut["date"].min(), fut["date"].max(), color=PALETTE["orange"], alpha=0.08)
    axs[1, 1].set_title("Forecast: Actual vs Predicted")
    axs[1, 1].set_xlabel("Date")
    axs[1, 1].set_ylabel("Sales")
    format_axis_in_millions(axs[1, 1])
    axs[1, 1].legend(loc="upper left")

    fig.savefig(CHARTS_DIR / "dashboard.png", dpi=170)
    plt.close(fig)


def fit_linear_trend_forecast(daily_sales: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """Forecast daily sales using a simple linear regression trend model."""
    daily_sales = daily_sales.sort_values("date").copy()

    x_hist = np.arange(len(daily_sales), dtype=float)
    y_hist = daily_sales["sales"].values.astype(float)

    if len(daily_sales) < 2:
        raise ValueError("Not enough rows to create a trend forecast. Need at least 2 dates.")

    x_design = np.vstack([x_hist, np.ones(len(x_hist))]).T
    slope, intercept = np.linalg.lstsq(x_design, y_hist, rcond=None)[0]

    future_dates = pd.date_range(
        start=daily_sales["date"].max() + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    x_future = np.arange(len(daily_sales), len(daily_sales) + horizon_days, dtype=float)

    hist_pred = slope * x_hist + intercept
    fut_pred = slope * x_future + intercept

    history = pd.DataFrame(
        {
            "date": daily_sales["date"],
            "actual_sales": daily_sales["sales"],
            "predicted_sales": hist_pred,
            "is_forecast": False,
        }
    )

    future = pd.DataFrame(
        {
            "date": future_dates,
            "actual_sales": np.nan,
            "predicted_sales": fut_pred,
            "is_forecast": True,
        }
    )

    forecast_df = pd.concat([history, future], ignore_index=True)
    forecast_df["predicted_sales"] = forecast_df["predicted_sales"].clip(lower=0)
    return forecast_df


def plot_forecast(forecast_df: pd.DataFrame) -> None:
    """Plot actual sales against predicted sales and save forecast chart."""
    apply_chart_style()
    plt.figure(figsize=(13, 6.5))

    hist = forecast_df[forecast_df["is_forecast"] == False]
    fut = forecast_df[forecast_df["is_forecast"] == True]

    plt.plot(hist["date"], hist["actual_sales"], label="Actual Sales", linewidth=1.7, color=PALETTE["blue"])
    plt.plot(
        forecast_df["date"],
        forecast_df["predicted_sales"],
        label="Predicted Sales",
        linestyle="--",
        linewidth=2.3,
        color=PALETTE["orange"],
    )

    if not fut.empty:
        plt.axvline(fut["date"].min(), color=PALETTE["gray"], linestyle=":", linewidth=1.6)
        plt.axvspan(fut["date"].min(), fut["date"].max(), color=PALETTE["orange"], alpha=0.06)

    plt.title("Actual vs Predicted Sales (30-Day Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    format_axis_in_millions(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "forecast.png", dpi=150)
    plt.close()


def save_outputs(merged: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """Save cleaned merged dataset and forecast dataframe."""
    merged.to_csv(DATA_DIR / "cleaned_retail_data.csv", index=False)
    forecast_df.to_csv(OUTPUT_DIR / "forecast_actual_vs_predicted.csv", index=False)


def main() -> None:
    """Run full ETL + analytics + forecasting workflow."""
    try:
        ensure_directories()
        datasets = load_data()
        cleaned = clean_dataframes(datasets)

        train_with_features = add_features(cleaned["train"])
        cleaned["train"] = train_with_features

        merged = merge_datasets(cleaned)
        analytics = run_analysis(merged)

        daily_sales = merged.groupby("date", as_index=False)["sales"].sum().sort_values("date")
        forecast_df = fit_linear_trend_forecast(daily_sales, horizon_days=30)

        create_visualizations(
            daily_sales=daily_sales,
            sales_by_store=analytics["sales_by_store"],
            sales_by_family=analytics["sales_by_family"],
        )
        plot_forecast(forecast_df)
        create_dashboard(
            daily_sales=daily_sales,
            sales_by_store=analytics["sales_by_store"],
            sales_by_family=analytics["sales_by_family"],
            forecast_df=forecast_df,
        )

        save_outputs(merged, forecast_df)

        print("\nPipeline completed successfully.")
        print(f"Charts saved to: {CHARTS_DIR}")
        print(f"Cleaned data saved to: {DATA_DIR / 'cleaned_retail_data.csv'}")
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
