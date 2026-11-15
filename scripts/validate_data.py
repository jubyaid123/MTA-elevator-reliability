"""Compare a newly prepared CSV with the workbook data and dashboard findings."""

from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import io
import json
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "mta_availability_2026-10-07.csv"
CSV_NAME = "mta_elevators_2024_2025_tableau.csv"
SOURCE_SHA256 = "d3caef0322e5e4f84e567a1d630f2143f0ab5cb05a81b6fc93691b3932e302d9"
REFERENCE_SHA256 = "194e7d92d041f76459b31a7ecb0b22c54c0713fa7c7a8047e1e3d1261cab8272"
FLOAT_COLUMNS = [
    "24-Hour Hours Available", "24-Hour Total Hours", "Downtime Hours",
    "Monthly Availability", "Annual Availability", "Availability Change 2024 to 2025 pp",
]
EXPECTED_COLUMNS = [
    "Month", "Year", "Equipment Code", "Borough", "Station Name",
    "Station Complex Name", "Station Complex MRN", "Total Outages",
    "Scheduled Outages", "Unscheduled Outages", "Entrapments",
    "24-Hour Hours Available", "24-Hour Total Hours", "Downtime Hours",
    "Monthly Availability", "Annual Availability", "Top 10 Downtime In Year",
    "Availability Change 2024 to 2025 pp", "Outage Change 2024 to 2025",
    "Fewer Outages But Lower Availability",
]
KEYS = ["Month", "Equipment Code"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_reference(path):
    """A reference is used for comparison only, never as a preparation input."""
    if path.suffix.lower() == ".twbx":
        with ZipFile(path) as archive:
            return archive.read("Data/" + CSV_NAME)
    return path.read_bytes()


def audit_source(source_bytes):
    require(sha256(source_bytes) == SOURCE_SHA256, "The raw file is not the frozen source snapshot.")
    raw = pd.read_csv(io.BytesIO(source_bytes))
    require(raw.shape == (83079, 22), "Unexpected raw snapshot dimensions.")
    dates = pd.to_datetime(raw["Month"], errors="raise")
    require(not raw[KEYS].isna().any().any(), "Missing source equipment-month keys.")
    require(not raw.duplicated(KEYS).any(), "Duplicate source equipment-month keys.")
    available = raw["24-Hour Hours Available"]
    total = raw["24-Hour Total Hours"]
    require(available.notna().all() and total.notna().all(), "Missing source hour values.")
    require(((available >= 0) & (total >= available)).all(), "Invalid source hours.")
    count_columns = ["Total Outages", "Scheduled Outages", "Unscheduled Outages", "Entrapments"]
    counts = raw[count_columns]
    require(counts.notna().all().all(), "Missing source outage counts.")
    require(((counts >= 0) & ((counts % 1) == 0)).all().all(), "Invalid source outage counts.")
    require(
        raw["Total Outages"].eq(raw["Scheduled Outages"] + raw["Unscheduled Outages"]).all(),
        "Total outages do not reconcile to scheduled plus unscheduled outages.",
    )
    # Published percentages are strings such as "98.60215053763%".
    published = pd.to_numeric(raw["24-Hour Availability"].str.rstrip("%"), errors="raise")
    require(published.isna().equals(total.eq(0)), "Unexpected pattern of undefined availability.")
    calculated = available.div(total.replace(0, np.nan)) * 100
    max_difference = float((published - calculated).abs().max())
    require(max_difference <= 1e-8, "Calculated rates do not match published percentages.")
    return {
        "sha256": sha256(source_bytes), "rows": len(raw), "columns": len(raw.columns),
        "first_month": dates.min().strftime("%Y-%m-%d"),
        "last_month": dates.max().strftime("%Y-%m-%d"),
        "missing_24_hour_percentages": int(published.isna().sum()),
        "zero_included_hour_rows": int(total.eq(0).sum()),
        "max_published_rate_difference_pp": max_difference,
        "duplicate_keys": 0, "invalid_hours": 0, "invalid_outage_counts": 0,
    }


def dashboard_metrics(frame):
    """Recompute findings from monthly counts/hours, not from repeated flags."""
    require(frame.shape == (5064, 20), "Expected 5,064 rows and 20 columns.")
    require(not frame.isna().any().any(), "Missing values in the prepared data.")
    require(frame["Equipment Code"].nunique() == 211, "Expected 211 distinct elevators.")
    require(frame.groupby("Year").size().to_dict() == {2024: 2532, 2025: 2532}, "Wrong year coverage.")
    require(frame.groupby("Equipment Code").size().eq(24).all(), "An elevator is missing months.")
    require(not frame["Equipment Code"].str.endswith("X").any(), "External operator in cohort.")
    dates = pd.to_datetime(frame["Month"], errors="raise")
    require(frame["Year"].eq(dates.dt.year).all(), "Year and Month disagree.")
    require(
        np.allclose(frame["24-Hour Total Hours"], dates.dt.days_in_month * 24, atol=1e-6, rtol=0),
        "The cohort contains excluded calendar hours.",
    )
    require(
        np.allclose(
            frame["Downtime Hours"],
            frame["24-Hour Total Hours"] - frame["24-Hour Hours Available"],
            atol=1e-6, rtol=1e-10,
        ), "Prepared downtime does not reconcile to hours.",
    )
    annual = frame.groupby(["Equipment Code", "Year"]).agg(
        available=("24-Hour Hours Available", "sum"),
        total=("24-Hour Total Hours", "sum"),
        outages=("Total Outages", "sum"), downtime=("Downtime Hours", "sum"),
    )
    annual["availability"] = annual["available"] / annual["total"]
    rates = annual["availability"].unstack("Year")
    outages = annual["outages"].unstack("Year")
    downtime = annual["downtime"].unstack("Year")
    availability_change = (rates[2025] - rates[2024]) * 100
    outage_change = outages[2025] - outages[2024]
    fewer = outage_change < 0
    exception = fewer & (availability_change < 0)
    improved = fewer & (availability_change > 0)
    require(int(fewer.sum()) == 102, "Expected 102 elevators with fewer outages.")
    require(int(improved.sum()) == 84 and int(exception.sum()) == 18, "Expected the 84/18 breakdown.")
    values = rates[2025]
    bands = {
        "below_95_percent": int((values < .95).sum()),
        "95_to_under_98_percent": int(((values >= .95) & (values < .98)).sum()),
        "98_percent_or_higher": int((values >= .98).sum()),
    }
    require(list(bands.values()) == [6, 63, 142], "Wrong 2025 availability bands.")
    annual_totals = annual.groupby("Year")[["available", "total"]].sum()
    overall = (annual_totals["available"] / annual_totals["total"]).to_dict()
    require(np.isclose(overall[2024], .9796936013202122, atol=1e-10, rtol=0), "Wrong 2024 availability.")
    require(np.isclose(overall[2025], .9805520139775591, atol=1e-10, rtol=0), "Wrong 2025 availability.")
    exception_outages = outages.loc[exception].sum()
    exception_downtime = downtime.loc[exception].sum()
    require(exception_outages.to_dict() == {2024: 775, 2025: 665}, "Wrong exception outage totals.")
    require(
        np.allclose(exception_downtime.values, [2982.9166666667, 3881.5166666667], atol=1e-6, rtol=0),
        "Wrong exception downtime totals.",
    )
    increases = downtime[2025] - downtime[2024]
    net_increase = float(increases.loc[exception].sum())
    two_increase = float(increases.loc[["EL194", "EL138"]].sum())
    two_share = two_increase / net_increase
    require(np.isclose(two_share, .4698605238, atol=1e-9, rtol=0), "Wrong 47% contribution.")
    require(availability_change.loc["EL506"] < 0 and increases.loc["EL506"] < 0, "Lost EL506 edge case.")
    exception_codes = availability_change.loc[exception].sort_values().index.tolist()
    heatmap = frame.loc[(frame["Year"] == 2025) & frame["Equipment Code"].isin(exception_codes)]
    require(len(heatmap) == 216 and heatmap.groupby("Equipment Code").size().eq(12).all(), "Wrong heatmap population.")
    return {
        "elevators": 211, "rows": 5064, "rows_per_year": {2024: 2532, 2025: 2532},
        "fewer_outages": int(fewer.sum()), "of_those_improved": int(improved.sum()),
        "of_those_declined": int(exception.sum()), "declined_share_of_102": float(exception.sum() / fewer.sum()),
        "availability_bands_2025": bands, "overall_availability_fraction": overall,
        "exception_outages": exception_outages.astype(int).to_dict(),
        "exception_downtime_hours": exception_downtime.to_dict(),
        "exception_downtime_net_increase_hours": net_increase,
        "EL194_and_EL138_net_increase_hours": two_increase,
        "EL194_and_EL138_share_of_exception_net_increase": two_share,
        "exception_codes_in_decline_order": exception_codes, "heatmap_cells": len(heatmap),
        "EL506_availability_change_pp": float(availability_change.loc["EL506"]),
        "EL506_downtime_change_hours": float(increases.loc["EL506"]),
    }


def verify(source, generated, reference):
    source_audit = audit_source(Path(source).read_bytes())
    generated_bytes = Path(generated).read_bytes()
    reference_bytes = read_reference(Path(reference))
    require(sha256(reference_bytes) == REFERENCE_SHA256, "Reference does not match the workbook target hash.")
    result = pd.read_csv(io.BytesIO(generated_bytes))
    expected = pd.read_csv(io.BytesIO(reference_bytes))
    require(result.columns.tolist() == expected.columns.tolist() == EXPECTED_COLUMNS, "Column names/order differ.")
    require(not result[KEYS].isna().any().any(), "Missing generated keys.")
    require(not result.duplicated(KEYS).any() and not expected.duplicated(KEYS).any(), "Duplicate keys.")
    left = result.set_index(KEYS).sort_index()
    right = expected.set_index(KEYS).sort_index()
    require(left.index.equals(right.index), "Equipment-month key sets differ.")
    exact_columns = [column for column in left.columns if column not in FLOAT_COLUMNS]
    require(
        left[exact_columns].eq(right[exact_columns]).all().all(),
        "Text, counts, identifiers or flags differ.",
    )
    max_differences = {}
    for column in FLOAT_COLUMNS:
        require(
            np.allclose(left[column], right[column], atol=1e-9, rtol=1e-10, equal_nan=False),
            "Numeric values differ in " + column,
        )
        max_differences[column] = float((left[column] - right[column]).abs().max())
    byte_identical = generated_bytes == reference_bytes
    return {
        "status": "byte-identical" if byte_identical else "values match; CSV bytes differ",
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "raw_source_checks": source_audit,
        "comparison": {
            "generated_sha256": sha256(generated_bytes), "reference_sha256": sha256(reference_bytes),
            "byte_identical": byte_identical, "columns_and_order_identical": True,
            "key_sets_identical": True, "row_order_identical": result[KEYS].equals(expected[KEYS]),
            "exact_columns_identical": True, "numeric_values_match": True,
            "absolute_tolerance": 1e-9, "relative_tolerance": 1e-10,
            "maximum_absolute_difference_by_column": max_differences,
        },
        "dashboard_metrics": dashboard_metrics(result),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw" / RAW_NAME)
    parser.add_argument("--generated", type=Path, default=ROOT / "data/processed" / CSV_NAME)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference" / CSV_NAME,
                        help="Embedded reference CSV, or a .twbx containing that CSV")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/validation.json")
    args = parser.parse_args()
    try:
        report = verify(args.source, args.generated, args.reference)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print("Verification failed: " + str(error), file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verification: " + report["status"])
    print("5,064 rows; 211 elevators; 102 with fewer outages; 84 improved / 18 declined.")
    print("Report: " + str(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
