"""Prepare a matched 2024–2025 elevator dataset for Tableau."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/mta_availability_2026-10-07.csv"
OUTPUT = ROOT / "data/processed/mta_elevators_2024_2025_tableau.csv"
YEARS = [2024, 2025]
EXPORT_COLUMNS = [
    "Month", "Year", "Equipment Code", "Borough", "Station Name",
    "Station Complex Name", "Station Complex MRN", "Total Outages",
    "Scheduled Outages", "Unscheduled Outages", "Entrapments",
    "24-Hour Hours Available", "24-Hour Total Hours", "Downtime Hours",
    "Monthly Availability", "Annual Availability", "Top 10 Downtime In Year",
    "Availability Change 2024 to 2025 pp", "Outage Change 2024 to 2025",
    "Fewer Outages But Lower Availability",
]


def load_data(source):
    data = pd.read_csv(source)
    data["Month"] = pd.to_datetime(data["Month"], errors="raise")
    if data[["Equipment Code", "Month"]].isna().any().any():
        raise ValueError("Missing equipment codes or dates.")
    if data.duplicated(["Equipment Code", "Month"]).any():
        raise ValueError("Duplicate equipment-month records.")
    return data


def select_comparison_group(data):
    # MTA identifies external operators with equipment codes ending in X.
    selected = data.loc[
        data["Month"].dt.year.isin(YEARS)
        & data["Equipment Type"].eq("Elevator")
        & ~data["Equipment Code"].str.endswith("X")
    ].copy()
    selected["Year"] = selected["Month"].dt.year
    calendar_hours = selected["Month"].dt.days_in_month * 24
    selected["Full Calendar Month"] = np.isclose(
        selected["24-Hour Total Hours"], calendar_hours, atol=1e-6, rtol=0,
    )
    coverage = selected.groupby(["Equipment Code", "Year"]).agg(
        months=("Month", "nunique"), full_months=("Full Calendar Month", "sum"),
    )
    complete_years = coverage.loc[
        coverage["months"].eq(12) & coverage["full_months"].eq(12)
    ]
    year_counts = complete_years.reset_index().groupby("Equipment Code")["Year"].nunique()
    matched_codes = year_counts.loc[year_counts.eq(len(YEARS))].index
    return selected.loc[selected["Equipment Code"].isin(matched_codes)].copy()


def add_comparison_fields(monthly):
    monthly["Downtime Hours"] = (
        monthly["24-Hour Total Hours"] - monthly["24-Hour Hours Available"]
    )
    monthly["Monthly Availability"] = (
        monthly["24-Hour Hours Available"] / monthly["24-Hour Total Hours"]
    )
    annual = monthly.groupby(["Equipment Code", "Year"]).agg(
        available=("24-Hour Hours Available", "sum"),
        total=("24-Hour Total Hours", "sum"),
        outages=("Total Outages", "sum"),
        downtime=("Downtime Hours", "sum"),
    ).reset_index()
    annual["Annual Availability"] = annual["available"] / annual["total"]

    # Retain this legacy field to preserve the Tableau export schema.
    annual["Top 10 Downtime In Year"] = (
        annual.groupby("Year")["downtime"].rank(ascending=False, method="min") <= 10
    )
    rates = annual.pivot(index="Equipment Code", columns="Year", values="Annual Availability")
    outages = annual.pivot(index="Equipment Code", columns="Year", values="outages")
    changes = pd.DataFrame(index=rates.index)
    changes["Availability Change 2024 to 2025 pp"] = (rates[2025] - rates[2024]) * 100
    changes["Outage Change 2024 to 2025"] = outages[2025] - outages[2024]
    changes["Fewer Outages But Lower Availability"] = (
        changes["Outage Change 2024 to 2025"].lt(0)
        & changes["Availability Change 2024 to 2025 pp"].lt(0)
    )

    # Repeat annual helpers on monthly rows so Tableau can use one data source.
    result = monthly.merge(
        annual[["Equipment Code", "Year", "Annual Availability", "Top 10 Downtime In Year"]],
        on=["Equipment Code", "Year"], validate="many_to_one",
    ).merge(changes.reset_index(), on="Equipment Code", validate="many_to_one")
    return result[EXPORT_COLUMNS].sort_values(["Month", "Equipment Code"])


def prepare(source=SOURCE, output=OUTPUT):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must be different files.")
    monthly = select_comparison_group(load_data(source))
    if monthly.empty:
        raise ValueError("No elevators meet the comparison rules.")
    result = add_comparison_fields(monthly)
    if result.isna().any().any():
        raise ValueError("The prepared dataset contains missing values.")
    if not result.groupby("Equipment Code").size().eq(24).all():
        raise ValueError("Each elevator must have 24 monthly records.")
    if result["Downtime Hours"].lt(0).any():
        raise ValueError("Downtime cannot be negative.")
    if not result["Monthly Availability"].between(0, 1).all():
        raise ValueError("Availability must be between zero and one.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, date_format="%Y-%m-%d", float_format="%.12g")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(f"Saved {len(result):,} monthly records for {result['Equipment Code'].nunique()} elevators.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
