"""Sales Analytics & Forecasting pipeline.

This script builds a complete analytics workflow for the Kaggle Store Sales dataset:
1) Loads raw files
2) Cleans and standardizes data
3) Engineers date features
4) Merges analytical tables
5) Produces business analysis and visual outputs
6) Forecasts next 30 days with rolling average + linear regression
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"


REQUIRED_FILES = {
    "train": DATA_RAW_DIR / "train.csv",
    "stores": DATA_RAW_DIR / "stores.csv",
    "transactions": DATA_RAW_DIR / "transactions.csv",
    "oil": DATA_RAW_DIR / "oil.csv",
    "holidays_events": DATA_RAW_DIR / "holidays_events.csv",
}


PALETTE = {
    "blue": "#2463EB",
    "teal": "#14B8A6",
    "orange": "#F97316",
    "rose": "#E11D48",
    "slate": "#334155",
    "gray": "#94A3B8",
}


def log_stage(message: str) -> None:
    """Print pipeline progress in a consistent format."""
    print(f"\n[PIPELINE] {message}")


def ensure_directories() -> None:
    """Ensure output and processed-data directories exist."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    """Apply a clean and consistent plotting style."""
    sns.set_theme(style="whitegrid", context="talk")
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
            "grid.linewidth": 0.8,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E1",
        }
    )


def format_millions(axis: plt.Axes, target_axis: str = "y") -> None:
    """Format numeric axis labels in millions for better readability."""
    formatter = ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M")
    if target_axis == "x":
        axis.xaxis.set_major_formatter(formatter)
    else:
        axis.yaxis.set_major_formatter(formatter)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns to lowercase snake_case format."""
    rename_map = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    return df.rename(columns=rename_map)


def print_dataframe_profile(name: str, df: pd.DataFrame) -> None:
    """Print small profile summary for traceability."""
    print(f"\n{name.upper()} - HEAD")
    print(df.head())
    print(f"\n{name.upper()} - COLUMNS")
    print(df.columns.tolist())

    buffer = io.StringIO()
    df.info(buf=buffer)
    print(f"\n{name.upper()} - INFO")
    print(buffer.getvalue())


def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required raw data files."""
    log_stage("Loading raw datasets")

    missing_files = [str(path) for path in REQUIRED_FILES.values() if not path.exists()]
    if missing_files:
        formatted_missing = "\n".join(f"- {file_path}" for file_path in missing_files)
        raise FileNotFoundError(f"Missing required dataset files:\n{formatted_missing}")

    datasets = {name: pd.read_csv(path) for name, path in REQUIRED_FILES.items()}

    for name, dataframe in datasets.items():
        print_dataframe_profile(name, dataframe)

    return datasets


def clean_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean datasets: columns, dates, nulls, and invalid records."""
    log_stage("Cleaning and standardizing datasets")
    cleaned_frames: Dict[str, pd.DataFrame] = {}

    for name, dataframe in datasets.items():
        dataframe = standardize_columns(dataframe)

        if "date" in dataframe.columns:
            dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
            dataframe = dataframe.dropna(subset=["date"])

        for column in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                if dataframe[column].isna().any():
                    median_value = dataframe[column].median()
                    dataframe[column] = dataframe[column].fillna(0 if pd.isna(median_value) else median_value)
            else:
                if dataframe[column].isna().any():
                    mode_values = dataframe[column].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
                    dataframe[column] = dataframe[column].fillna(fill_value)

        if name == "train" and "sales" in dataframe.columns:
            dataframe = dataframe[dataframe["sales"].notna()]
            dataframe = dataframe[dataframe["sales"] >= 0]

        if name == "transactions" and "transactions" in dataframe.columns:
            dataframe = dataframe[dataframe["transactions"].notna()]
            dataframe = dataframe[dataframe["transactions"] >= 0]

        cleaned_frames[name] = dataframe

    if "oil" in cleaned_frames and "dcoilwtico" in cleaned_frames["oil"].columns:
        oil_dataframe = cleaned_frames["oil"].sort_values("date").copy()
        oil_dataframe["dcoilwtico"] = oil_dataframe["dcoilwtico"].replace(0, np.nan).ffill().bfill()
        cleaned_frames["oil"] = oil_dataframe

    return cleaned_frames


def feature_engineering(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add core date features used for aggregation and modeling."""
    log_stage("Applying feature engineering")

    if "date" not in train_df.columns:
        raise KeyError("Training dataframe is missing 'date' column.")

    engineered_df = train_df.copy()
    engineered_df["year"] = engineered_df["date"].dt.year
    engineered_df["month"] = engineered_df["date"].dt.month
    engineered_df["day"] = engineered_df["date"].dt.day
    engineered_df["day_of_week"] = engineered_df["date"].dt.dayofweek
    return engineered_df


def merge_data(cleaned_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge train + stores + transactions + oil into one analytical table."""
    log_stage("Merging datasets")

    merged_df = cleaned_frames["train"].merge(
        cleaned_frames["stores"],
        on="store_nbr",
        how="left",
    )

    merged_df = merged_df.merge(
        cleaned_frames["transactions"],
        on=["date", "store_nbr"],
        how="left",
        suffixes=("", "_transactions"),
    )

    merged_df = merged_df.merge(
        cleaned_frames["oil"][["date", "dcoilwtico"]],
        on="date",
        how="left",
    )

    if "transactions" in merged_df.columns:
        merged_df["transactions"] = merged_df["transactions"].fillna(0)

    if "dcoilwtico" in merged_df.columns:
        merged_df["dcoilwtico"] = merged_df["dcoilwtico"].ffill().bfill()

    return merged_df


def analysis(merged_df: pd.DataFrame) -> Dict[str, pd.DataFrame | float]:
    """Create core business metrics and aggregated tables."""
    log_stage("Running business analysis")

    total_sales = float(merged_df["sales"].sum())
    daily_sales = merged_df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    sales_by_store = merged_df.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    sales_by_family = merged_df.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    sales_by_month = (
        merged_df.groupby(["year", "month"], as_index=False)["sales"].sum().sort_values(["year", "month"])
    )

    print(f"Total Sales: {total_sales:,.2f}")
    print("Top 10 Stores by Sales:")
    print(sales_by_store.head(10))

    return {
        "total_sales": total_sales,
        "daily_sales": daily_sales,
        "sales_by_store": sales_by_store,
        "sales_by_family": sales_by_family,
        "sales_by_month": sales_by_month,
    }


def forecasting(daily_sales: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """Forecast next 30 days using blended rolling average and linear regression."""
    log_stage("Forecasting future sales")

    if len(daily_sales) < 2:
        raise ValueError("At least two daily records are required for forecasting.")

    forecast_source = daily_sales.sort_values("date").copy()
    forecast_source["time_index"] = np.arange(len(forecast_source), dtype=float)

    model = LinearRegression()
    history_index = forecast_source["time_index"].to_numpy().reshape(-1, 1)
    model.fit(history_index, forecast_source["sales"].to_numpy())

    linear_history = model.predict(history_index)
    rolling_window = 30
    rolling_history = forecast_source["sales"].rolling(window=rolling_window, min_periods=1).mean().values

    future_dates = pd.date_range(
        start=forecast_source["date"].max() + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    future_index = np.arange(len(forecast_source), len(forecast_source) + horizon_days, dtype=float)
    linear_future = model.predict(future_index.reshape(-1, 1))

    rolling_seed = forecast_source["sales"].tail(rolling_window).tolist()
    rolling_future = []
    for _ in range(horizon_days):
        rolling_prediction = float(np.mean(rolling_seed[-rolling_window:]))
        rolling_future.append(rolling_prediction)
        rolling_seed.append(rolling_prediction)

    history_blended = 0.65 * linear_history + 0.35 * rolling_history
    future_blended = 0.65 * linear_future + 0.35 * np.array(rolling_future)

    history_frame = pd.DataFrame(
        {
            "date": forecast_source["date"],
            "actual_sales": forecast_source["sales"],
            "linear_prediction": linear_history,
            "rolling_prediction": rolling_history,
            "predicted_sales": history_blended,
            "is_forecast": False,
        }
    )

    future_frame = pd.DataFrame(
        {
            "date": future_dates,
            "actual_sales": np.nan,
            "linear_prediction": linear_future,
            "rolling_prediction": rolling_future,
            "predicted_sales": future_blended,
            "is_forecast": True,
        }
    )

    forecast_df = pd.concat([history_frame, future_frame], ignore_index=True)
    forecast_df["predicted_sales"] = forecast_df["predicted_sales"].clip(lower=0)
    return forecast_df


def create_visuals(
    daily_sales: pd.DataFrame,
    sales_by_store: pd.DataFrame,
    sales_by_family: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    """Create and save portfolio-ready visualization set."""
    log_stage("Creating visual outputs")
    apply_plot_style()

    fig, axis = plt.subplots(figsize=(14, 7))
    rolling_30 = daily_sales["sales"].rolling(window=30, min_periods=1).mean()
    axis.plot(daily_sales["date"], daily_sales["sales"], color=PALETTE["blue"], alpha=0.3, linewidth=1.2, label="Daily Sales")
    axis.plot(daily_sales["date"], rolling_30, color=PALETTE["teal"], linewidth=2.4, label="30-Day Trend")
    axis.set_title("Sales Over Time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales")
    format_millions(axis)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sales_over_time.png", dpi=170)
    plt.close(fig)

    top_20_stores = sales_by_store.head(20)
    fig, axis = plt.subplots(figsize=(14, 7))
    store_colors = plt.cm.plasma(np.linspace(0.1, 0.95, len(top_20_stores)))
    axis.bar(top_20_stores["store_nbr"].astype(str), top_20_stores["sales"], color=store_colors)
    axis.set_title("Top 20 Stores by Sales")
    axis.set_xlabel("Store Number")
    axis.set_ylabel("Sales")
    axis.tick_params(axis="x", rotation=35)
    format_millions(axis)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sales_by_store.png", dpi=170)
    plt.close(fig)

    top_families = sales_by_family.head(15).iloc[::-1]
    fig, axis = plt.subplots(figsize=(14, 8))
    family_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_families)))
    axis.barh(top_families["family"], top_families["sales"], color=family_colors)
    axis.set_title("Top Product Families by Sales")
    axis.set_xlabel("Sales")
    axis.set_ylabel("Product Family")
    format_millions(axis, target_axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sales_by_family.png", dpi=170)
    plt.close(fig)

    history = forecast_df[forecast_df["is_forecast"] == False]
    future = forecast_df[forecast_df["is_forecast"] == True]

    fig, axis = plt.subplots(figsize=(14, 7))
    axis.plot(history["date"], history["actual_sales"], color=PALETTE["blue"], linewidth=1.5, label="Actual Sales")
    axis.plot(forecast_df["date"], forecast_df["predicted_sales"], color=PALETTE["orange"], linestyle="--", linewidth=2.3, label="Predicted Sales")
    if not future.empty:
        axis.axvline(future["date"].min(), color=PALETTE["gray"], linestyle=":", linewidth=1.6)
        axis.axvspan(future["date"].min(), future["date"].max(), color=PALETTE["orange"], alpha=0.08)
    axis.set_title("Actual vs Predicted Sales (30-Day Forecast)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales")
    format_millions(axis)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "forecast.png", dpi=170)
    plt.close(fig)

    # Combined executive-style dashboard with tighter framing for a closer preview.
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    fig.suptitle("Retail Sales Analytics Dashboard", fontsize=26, fontweight="bold", color="#0F172A")

    axes[0, 0].plot(daily_sales["date"], daily_sales["sales"], color=PALETTE["blue"], alpha=0.25, linewidth=1.2)
    axes[0, 0].plot(daily_sales["date"], rolling_30, color=PALETTE["teal"], linewidth=2.2)
    axes[0, 0].set_title("Sales Trend Over Time", fontsize=18)
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Sales")
    axes[0, 0].tick_params(axis="both", labelsize=12)
    format_millions(axes[0, 0])

    top_12_stores = sales_by_store.head(12)
    top_store_colors = plt.cm.coolwarm(np.linspace(0.1, 0.95, len(top_12_stores)))
    axes[0, 1].bar(top_12_stores["store_nbr"].astype(str), top_12_stores["sales"], color=top_store_colors)
    axes[0, 1].set_title("Top 12 Stores by Sales", fontsize=18)
    axes[0, 1].set_xlabel("Store")
    axes[0, 1].set_ylabel("Sales")
    axes[0, 1].tick_params(axis="x", rotation=35, labelsize=12)
    axes[0, 1].tick_params(axis="y", labelsize=12)
    format_millions(axes[0, 1])

    top_10_families = sales_by_family.head(10).iloc[::-1]
    top_family_colors = plt.cm.magma(np.linspace(0.15, 0.9, len(top_10_families)))
    axes[1, 0].barh(top_10_families["family"], top_10_families["sales"], color=top_family_colors)
    axes[1, 0].set_title("Top 10 Product Families", fontsize=18)
    axes[1, 0].set_xlabel("Sales")
    axes[1, 0].set_ylabel("Family")
    axes[1, 0].tick_params(axis="both", labelsize=12)
    format_millions(axes[1, 0], target_axis="x")

    axes[1, 1].plot(history["date"], history["actual_sales"], color=PALETTE["blue"], linewidth=1.5, label="Actual")
    axes[1, 1].plot(forecast_df["date"], forecast_df["predicted_sales"], color=PALETTE["orange"], linestyle="--", linewidth=2.2, label="Predicted")
    if not future.empty:
        axes[1, 1].axvspan(future["date"].min(), future["date"].max(), color=PALETTE["orange"], alpha=0.08)
    axes[1, 1].set_title("Forecast: Actual vs Predicted", fontsize=18)
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Sales")
    axes[1, 1].tick_params(axis="both", labelsize=12)
    format_millions(axes[1, 1])
    axes[1, 1].legend(loc="upper left", fontsize=11)

    fig.savefig(DASHBOARD_DIR / "dashboard.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_outputs(merged_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """Persist cleaned data and forecast results to disk."""
    log_stage("Saving output files")

    merged_df.to_csv(DATA_PROCESSED_DIR / "cleaned_retail_data.csv", index=False)
    forecast_df.to_csv(OUTPUT_DIR / "forecast_actual_vs_predicted.csv", index=False)


def main() -> None:
    """Execute complete ETL, analysis, visualization, and forecasting workflow."""
    try:
        log_stage("Pipeline started")
        ensure_directories()

        datasets = load_data()
        cleaned_frames = clean_data(datasets)

        cleaned_frames["train"] = feature_engineering(cleaned_frames["train"])
        merged_df = merge_data(cleaned_frames)

        analytics = analysis(merged_df)
        forecast_df = forecasting(analytics["daily_sales"], horizon_days=30)

        create_visuals(
            daily_sales=analytics["daily_sales"],
            sales_by_store=analytics["sales_by_store"],
            sales_by_family=analytics["sales_by_family"],
            forecast_df=forecast_df,
        )
        save_outputs(merged_df, forecast_df)

        log_stage("Pipeline completed successfully")
        print(f"Processed data: {DATA_PROCESSED_DIR / 'cleaned_retail_data.csv'}")
        print(f"Charts directory: {CHARTS_DIR}")
        print(f"Dashboard: {DASHBOARD_DIR / 'dashboard.png'}")
    except Exception as error:
        log_stage(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()
