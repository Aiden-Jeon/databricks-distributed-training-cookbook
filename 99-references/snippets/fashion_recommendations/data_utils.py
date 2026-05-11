"""
Data loading and manipulation utilities for MLOps pipeline
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from typing import List


def get_spark() -> SparkSession:
    """Get or create Spark session"""
    return SparkSession.builder.getOrCreate()


def load_delta_table(table_name: str) -> DataFrame:
    """
    Load Delta table from catalog

    Args:
        table_name: Fully qualified table name (e.g., "catalog.schema.table")

    Returns:
        Spark DataFrame
    """
    spark = get_spark()
    return spark.table(table_name)


def get_customer_purchase_history(
    customer_id: str, transactions_df: DataFrame
) -> DataFrame:
    """
    Get purchase history for a specific customer

    Args:
        customer_id: Customer identifier
        transactions_df: Transactions DataFrame

    Returns:
        DataFrame with customer's purchase history ordered by date
    """
    return (
        transactions_df.filter(col("customer_id") == customer_id)
        .orderBy("t_dat")
    )


def create_ground_truth_labels(
    transactions_df: DataFrame,
    start_date: str,
    end_date: str,
    top_k: int = 12
) -> DataFrame:
    """
    Create ground truth labels for evaluation

    Args:
        transactions_df: Transactions DataFrame
        start_date: Start date for ground truth period
        end_date: End date for ground truth period
        top_k: Number of top recommendations to evaluate

    Returns:
        DataFrame with customer_id and list of actual article_ids purchased
    """
    ground_truth = (
        transactions_df
        .filter((col("t_dat") >= start_date) & (col("t_dat") <= end_date))
        .groupBy("customer_id")
        .agg(
            collect_list("article_id").alias("actual_articles"),
            count("*").alias("num_purchases")
        )
    )
    return ground_truth


def filter_by_date_range(
    df: DataFrame,
    date_column: str,
    start_date: str = None,
    end_date: str = None
) -> DataFrame:
    """
    Filter DataFrame by date range

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        start_date: Start date (inclusive), None for no lower bound
        end_date: End date (inclusive), None for no upper bound

    Returns:
        Filtered DataFrame
    """
    if start_date and end_date:
        return df.filter((col(date_column) >= start_date) & (col(date_column) <= end_date))
    elif start_date:
        return df.filter(col(date_column) >= start_date)
    elif end_date:
        return df.filter(col(date_column) <= end_date)
    else:
        return df

