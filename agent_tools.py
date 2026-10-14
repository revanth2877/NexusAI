
import pandas as pd


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Return dataset size, columns, data types, and quality metrics."""
    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": [str(column) for column in df.columns],
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def calculate_numeric_summary(df: pd.DataFrame) -> dict:
    """Calculate reliable numeric statistics directly in Python."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "message": "The dataset has no numeric columns."
        }

    return {
        str(column): {
            "sum": float(numeric_df[column].sum()),
            "mean": float(numeric_df[column].mean())
            if numeric_df[column].notna().any() else None,
            "minimum": float(numeric_df[column].min())
            if numeric_df[column].notna().any() else None,
            "maximum": float(numeric_df[column].max())
            if numeric_df[column].notna().any() else None,
        }
        for column in numeric_df.columns
    }


def analyze_category(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    top_n: int = 10,
) -> dict:
    """Aggregate a numeric measure by category."""
    if category_column not in df.columns:
        return {"error": f"Column not found: {category_column}"}

    if value_column not in df.columns:
        return {"error": f"Column not found: {value_column}"}

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"error": f"{value_column} must be a numeric column."}

    top_n = max(1, min(int(top_n), 50))

    grouped = (
        df.groupby(category_column, dropna=False)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return {
        "category_column": category_column,
        "value_column": value_column,
        "results": [
            {
                "category": str(category),
                "total": float(total),
            }
            for category, total in grouped.items()
        ],
    }


def find_data_quality_issues(df: pd.DataFrame) -> dict:
    """Find missing values, duplicate rows, and constant columns."""
    missing = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    return {
        "missing_values_by_column": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_columns,
        "columns_with_missing_values": len(missing),
    }


def run_analysis_tool(
    tool_name: str,
    df: pd.DataFrame,
    arguments: dict | None = None,
) -> dict:
    """Safely dispatch a request to an approved analysis tool."""
    arguments = arguments or {}

    approved_tools = {
        "get_dataset_overview": lambda: get_dataset_overview(df),
        "calculate_numeric_summary": lambda: calculate_numeric_summary(df),
        "find_data_quality_issues": lambda: find_data_quality_issues(df),
        "analyze_category": lambda: analyze_category(
            df,
            category_column=str(arguments.get("category_column", "")),
            value_column=str(arguments.get("value_column", "")),
            top_n=arguments.get("top_n", 10),
        ),
    }

    if tool_name not in approved_tools:
        return {"error": f"Tool is not approved: {tool_name}"}

    try:
        return approved_tools[tool_name]()
    except Exception as error:
        return {
            "error": f"Tool execution failed: {type(error).__name__}: {error}"
        }
