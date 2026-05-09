"""
Module 2: NYC Yellow Taxi Data Analysis & Visualization
Depends on Module 1 (data_pipeline.py) -> run_pipeline()
All figures saved to ./outputs/
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_pipeline import run_pipeline   # Module 1

sns.set_style("whitegrid")
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============ 1. Temporal Demand Pattern ============
def analyze_time_pattern(df: pd.DataFrame):
    """Hourly demand split by weekday / weekend."""
    # 1.1 Average hourly trips: Weekday vs Weekend
    days = (df.assign(date=df["tpep_pickup_datetime"].dt.date)
              .groupby("is_weekend")["date"].nunique())
    grp = df.groupby(["is_weekend", "pickup_hour"]).size().reset_index(name="trips")
    grp["avg_trips"] = grp.apply(lambda r: r["trips"] / days[r["is_weekend"]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for flag, label, color in [(0, "Weekday", "steelblue"),
                               (1, "Weekend", "tomato")]:
        sub = grp[grp["is_weekend"] == flag]
        ax.plot(sub["pickup_hour"], sub["avg_trips"],
                marker="o", label=label, color=color)
    ax.set(xlabel="Hour of Day", ylabel="Average Trips",
           title="Average Hourly Trips: Weekday vs Weekend",
           xticks=range(24))
    ax.legend(title="Day Type")
    _save(fig, "01_hourly_demand.png")

    # 1.2 Weekday × Hour heatmap
    pivot = (df.groupby(["pickup_weekday", "pickup_hour"])
               .size().unstack(fill_value=0))
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Trip Count"})
    ax.set(xlabel="Hour of Day", ylabel="Day of Week",
           title="Trip Volume Heatmap (Day of Week × Hour)")
    _save(fig, "01_weekday_hour_heatmap.png")


# ============ 2. Zone Hotspot Analysis ============
def analyze_zone_hotspot(df: pd.DataFrame):
    """Top 10 pickup / dropoff zones and their peak-hour distribution."""
    if not {"PULocationID", "DOLocationID"}.issubset(df.columns):
        print("Missing LocationID columns, skipped.")
        return

    top_pu = df["PULocationID"].value_counts().head(10)
    top_do = df["DOLocationID"].value_counts().head(10)

    # 2.1 TOP10 bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=top_pu.values, y=top_pu.index.astype(str),
                ax=axes[0], hue=top_pu.index.astype(str),
                palette="Blues_r", orient="h", legend=False)
    axes[0].set(title="Top 10 Pickup Zones",
                xlabel="Number of Trips", ylabel="Pickup Zone ID")
    sns.barplot(x=top_do.values, y=top_do.index.astype(str),
                ax=axes[1], hue=top_do.index.astype(str),
                palette="Greens_r", orient="h", legend=False)
    axes[1].set(title="Top 10 Dropoff Zones",
                xlabel="Number of Trips", ylabel="Dropoff Zone ID")
    _save(fig, "02_top10_zones.png")

    # 2.2 Top10 pickup zones × hour heatmap
    top_ids = top_pu.index.tolist()
    pivot = (df[df["PULocationID"].isin(top_ids)]
               .groupby(["PULocationID", "pickup_hour"])
               .size().unstack(fill_value=0)
               .reindex(top_ids))
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Trip Count"})
    ax.set(xlabel="Hour of Day", ylabel="Pickup Zone ID",
           title="Hourly Trip Distribution of Top 10 Pickup Zones")
    _save(fig, "02_top_zone_hour_heatmap.png")


# ============ 3. Fare Influencing Factors ============
def analyze_fare_factors(df: pd.DataFrame):
    """Relation of distance / period / passenger count to fare."""
    sample = df.sample(min(50000, len(df)), random_state=42).copy()
    sample["Period"] = sample["is_rush_hour"].map({0: "Off-Peak", 1: "Rush Hour"})

    # 3.1 Distance vs Fare scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=sample, x="trip_distance", y="fare_amount",
                    hue="Period", alpha=0.3, s=10, ax=ax,
                    palette={"Off-Peak": "steelblue", "Rush Hour": "tomato"})
    ax.set(xlim=(0, 30), ylim=(0, 150),
           xlabel="Trip Distance (miles)", ylabel="Fare Amount (USD)",
           title="Trip Distance vs Fare (colored by Rush Hour)")
    _save(fig, "03_distance_vs_fare.png")

    # 3.2 Fare by time period (boxplot)
    period_map = {
        "late_night":   "Late Night (0-5)",
        "morning_peak": "Morning Peak (6-9)",
        "daytime":      "Daytime (10-16)",
        "evening_peak": "Evening Peak (17-19)",
        "night":        "Night (20-23)",
    }
    sample["TimePeriod"] = sample["time_period"].map(period_map)
    order = list(period_map.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=sample, x="TimePeriod", y="fare_amount",
                order=order, hue="TimePeriod",
                showfliers=False, ax=ax, palette="Set3", legend=False)
    ax.set(xlabel="Time Period", ylabel="Fare Amount (USD)",
           title="Fare Distribution Across Time Periods")
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, "03_fare_by_period.png")

    # 3.3 Passenger count vs avg fare
    grp = df.groupby("passenger_count")["fare_amount"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=grp, x="passenger_count", y="fare_amount",
                hue="passenger_count", palette="viridis",
                ax=ax, legend=False)
    ax.set(xlabel="Passenger Count", ylabel="Average Fare (USD)",
           title="Average Fare by Passenger Count")
    _save(fig, "03_fare_by_passenger.png")


# ============ 4. Speed vs Time Period ============
def analyze_speed_period(df: pd.DataFrame):
    """Average trip speed across hours and time periods."""
    spd = df[df["avg_speed_mph"].between(1, 60)].copy()   # filter outliers

    # 4.1 Average speed by hour
    hourly = spd.groupby("pickup_hour")["avg_speed_mph"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["pickup_hour"], hourly["avg_speed_mph"],
            marker="o", color="purple", label="Avg Speed")
    for h in [7, 8, 9, 17, 18, 19]:        # highlight rush hours
        ax.axvspan(h - 0.5, h + 0.5, color="red", alpha=0.08)
    ax.set(xlabel="Hour of Day", ylabel="Average Speed (mph)",
           xticks=range(24),
           title="Average Trip Speed by Hour (red = Rush Hours)")
    ax.legend()
    _save(fig, "04_speed_by_hour.png")

    # 4.2 Speed by time period (boxplot)
    period_map = {
        "late_night":   "Late Night (0-5)",
        "morning_peak": "Morning Peak (6-9)",
        "daytime":      "Daytime (10-16)",
        "evening_peak": "Evening Peak (17-19)",
        "night":        "Night (20-23)",
    }
    spd["TimePeriod"] = spd["time_period"].map(period_map)
    order = list(period_map.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=spd, x="TimePeriod", y="avg_speed_mph",
                order=order, hue="TimePeriod",
                showfliers=False, ax=ax, palette="coolwarm", legend=False)
    ax.set(xlabel="Time Period", ylabel="Average Speed (mph)",
           title="Trip Speed Distribution Across Time Periods")
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, "04_speed_by_period.png")


# ============ Main ============
def main():
    df = run_pipeline("yellow_tripdata_2026-01.parquet")
    analyze_time_pattern(df)
    analyze_zone_hotspot(df)
    analyze_fare_factors(df)
    analyze_speed_period(df)
    print(f"\nAll figures saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main() 