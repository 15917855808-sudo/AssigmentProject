"""
NYC Yellow Taxi 数据处理模块
功能：加载 Parquet 数据 -> 质量报告 -> 清洗 -> 特征工程
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============ 1. 数据加载 ============
def load_data(path: str = "yellow_tripdata_2026-01.parquet") -> pd.DataFrame:
    """加载 Parquet 文件（列式存储，读取效率高于 CSV）"""
    table = pq.read_table(path)
    return table.to_pandas()


# ============ 2. 数据质量报告 ============
def quality_report(df: pd.DataFrame) -> dict:
    """生成缺失率与异常值统计"""
    report = {"shape": df.shape}

    # 缺失率
    missing = df.isna().mean().sort_values(ascending=False)
    report["missing_rate"] = missing[missing > 0].round(4).to_dict()

    # 异常值统计（基于业务规则 + IQR）
    anomalies = {}
    if "trip_distance" in df:
        anomalies["distance_le_0"]   = int((df["trip_distance"] <= 0).sum())
        anomalies["distance_gt_100"] = int((df["trip_distance"] > 100).sum())
    if "fare_amount" in df:
        anomalies["fare_le_0"]    = int((df["fare_amount"] <= 0).sum())
        anomalies["fare_gt_1000"] = int((df["fare_amount"] > 1000).sum())
    if "passenger_count" in df:
        anomalies["passenger_invalid"] = int(
            ((df["passenger_count"] <= 0) | (df["passenger_count"] > 6)).sum()
        )
    if {"tpep_pickup_datetime", "tpep_dropoff_datetime"}.issubset(df.columns):
        dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        anomalies["duration_le_0"]    = int((dur <= 0).sum())
        anomalies["duration_gt_300"]  = int((dur > 300).sum())  # 超过5小时

    report["anomalies"] = anomalies
    print("=== 数据质量报告 ===")
    for k, v in report.items():
        print(f"\n[{k}]\n{v}")
    return report


# ============ 3. 数据清洗 ============
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗策略说明：
      - 删除关键字段缺失行：时间/距离/金额是核心字段，缺失无法补全
      - passenger_count 缺失填 1：单人是出租车最常见场景，众数填补
      - 行程时长 (0, 300] 分钟：≤0 为数据错误；>5h 极可能是计价器故障
      - 距离 (0, 100] 英里：纽约市内行程几乎不超过 100 英里
      - 票价 (0, 1000]：负值为退款记录，过大值多为系统异常
      - 乘客数 [1, 6]：超出常规出租车物理容量
    """
    df = df.copy()

    # (1) 关键字段缺失直接删除
    key_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                "trip_distance", "fare_amount"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns])

    # (2) 乘客数缺失用 1 填充（单人乘车最常见）
    if "passenger_count" in df:
        df["passenger_count"] = df["passenger_count"].fillna(1).astype(int)
        df = df[df["passenger_count"].between(1, 6)]

    # (3) 行程时长过滤
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[df["duration_min"].between(0.5, 300, inclusive="right")]

    # (4) 距离与票价过滤
    df = df[df["trip_distance"].between(0.1, 100)]
    df = df[df["fare_amount"].between(0.1, 1000)]

    return df.reset_index(drop=True)


# ============ 4. 特征工程 ============
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基础时间特征 + 衍生特征
    """
    df = df.copy()
    pu = df["tpep_pickup_datetime"]

    # --- 基础时间特征 ---
    df["pickup_hour"]    = pu.dt.hour
    df["pickup_weekday"] = pu.dt.weekday          # 0=周一
    df["is_weekend"]     = (df["pickup_weekday"] >= 5).astype(int)

    # 高峰时段：工作日 7-9 点 或 17-19 点
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0) &
        (df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)

    # --- 衍生特征 1：平均速度 (mph) ---
    # 反映路况；过低可能堵车，过高可能是高速场景
    df["avg_speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60)

    # --- 衍生特征 2：单位里程费率 ($/mile) ---
    # 反映计价合理性，可用于异常订单识别
    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"]

    # --- 衍生特征 3：时段分桶（凌晨/早高峰/日间/晚高峰/夜间） ---
    bins   = [-1, 5, 9, 16, 19, 23]
    labels = ["late_night", "morning_peak", "daytime", "evening_peak", "night"]
    df["time_period"] = pd.cut(df["pickup_hour"], bins=bins, labels=labels)

    # --- 衍生特征 4：是否短途（≤2 英里） ---
    df["is_short_trip"] = (df["trip_distance"] <= 2).astype(int)

    return df


# ============ 主流程 ============
def run_pipeline(path: str = "yellow_tripdata_2026-01.parquet") -> pd.DataFrame:
    df = load_data(path)
    quality_report(df)
    df = clean_data(df)
    df = build_features(df)
    print(f"\n清洗+特征工程完成，最终形状: {df.shape}")
    return df


if __name__ == "__main__":
    result = run_pipeline()
    print(result.head())