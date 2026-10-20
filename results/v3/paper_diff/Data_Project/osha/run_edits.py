import duckdb
import csv

CSV_PATH = "ITA 300A Summary Data 2023 through 12-31-2024_v2.csv"

con = duckdb.connect()
con.execute(f"""
    CREATE TABLE ita AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', header=True, sample_size=-1)
""")

n_rows = con.execute("SELECT COUNT(*) FROM ita").fetchone()[0]
print(f"Total rows: {n_rows}\n")

# Each rule: id, kind (validity=hard/must, quality=soft/should), category, description, SQL violation predicate
rules = [
    # --- Field-level required (hard, "must") ---
    ("R_ein_required", "validity", "null", "ein is a required field (must not be null)",
     "ein IS NULL"),

    # --- Field-level format/domain (hard, "must") ---
    ("R_ein_fmt", "validity", "pattern", "ein must be 9 digits, no dashes",
     "ein IS NOT NULL AND NOT regexp_matches(CAST(ein AS VARCHAR), '^[0-9]{9}$')"),
    ("R_state_fmt", "validity", "pattern", "state must be a 2-letter code",
     "state IS NOT NULL AND NOT regexp_matches(state, '^[A-Za-z]{2}$')"),
    ("R_zip_fmt", "validity", "pattern", "zip must be a 5 or 9 digit number",
     "zip_code IS NOT NULL AND NOT regexp_matches(CAST(zip_code AS VARCHAR), '^[0-9]{5}([0-9]{4})?$')"),
    ("R_naics_fmt", "validity", "pattern", "naics_code must be 6 digits",
     "naics_code IS NOT NULL AND NOT regexp_matches(CAST(naics_code AS VARCHAR), '^[0-9]{6}$')"),
    ("R_size_dom", "validity", "enum", "size must be in {1,21,22,3}",
     "size IS NOT NULL AND size NOT IN (1,21,22,3)"),
    ("R_estype_dom", "validity", "enum", "establishment_type must be in {1,2,3} if present",
     "establishment_type IS NOT NULL AND establishment_type NOT IN (1,2,3)"),
    ("R_noii_dom", "validity", "enum", "no_injuries_illnesses must be in {1,2}",
     "no_injuries_illnesses IS NOT NULL AND no_injuries_illnesses NOT IN (1,2)"),

    # --- Field-level range (hard, "must") ---
    ("R_avgemp_pos", "validity", "range", "annual_average_employees must be > 0",
     "annual_average_employees IS NOT NULL AND annual_average_employees <= 0"),
    ("R_hours_pos", "validity", "range", "total_hours_worked must be > 0",
     "total_hours_worked IS NOT NULL AND total_hours_worked <= 0"),
    ("R_deaths_nonneg", "validity", "range", "total_deaths must be >= 0",
     "total_deaths IS NOT NULL AND total_deaths < 0"),
    ("R_dafw_cases_nonneg", "validity", "range", "total_dafw_cases must be >= 0",
     "total_dafw_cases IS NOT NULL AND total_dafw_cases < 0"),
    ("R_djtr_cases_nonneg", "validity", "range", "total_djtr_cases must be >= 0",
     "total_djtr_cases IS NOT NULL AND total_djtr_cases < 0"),
    ("R_other_cases_nonneg", "validity", "range", "total_other_cases must be >= 0",
     "total_other_cases IS NOT NULL AND total_other_cases < 0"),
    ("R_dafw_days_nonneg", "validity", "range", "total_dafw_days must be >= 0",
     "total_dafw_days IS NOT NULL AND total_dafw_days < 0"),
    ("R_djtr_days_nonneg", "validity", "range", "total_djtr_days must be >= 0",
     "total_djtr_days IS NOT NULL AND total_djtr_days < 0"),
    ("R_injuries_nonneg", "validity", "range", "total_injuries must be >= 0",
     "total_injuries IS NOT NULL AND total_injuries < 0"),
    ("R_skin_nonneg", "validity", "range", "total_skin_disorders must be >= 0",
     "total_skin_disorders IS NOT NULL AND total_skin_disorders < 0"),
    ("R_resp_nonneg", "validity", "range", "total_respiratory_conditions must be >= 0",
     "total_respiratory_conditions IS NOT NULL AND total_respiratory_conditions < 0"),
    ("R_poison_nonneg", "validity", "range", "total_poisonings must be >= 0",
     "total_poisonings IS NOT NULL AND total_poisonings < 0"),
    ("R_hearing_nonneg", "validity", "range", "total_hearing_loss must be >= 0",
     "total_hearing_loss IS NOT NULL AND total_hearing_loss < 0"),
    ("R_otherill_nonneg", "validity", "range", "total_other_illnesses must be >= 0",
     "total_other_illnesses IS NOT NULL AND total_other_illnesses < 0"),

    # --- Column comparison (hard "must" / soft "should") ---
    ("R_hours_per_emp_hard", "validity", "comparison", "hours/employee must be < 8760",
     "annual_average_employees IS NOT NULL AND annual_average_employees > 0 "
     "AND total_hours_worked IS NOT NULL "
     "AND total_hours_worked >= 8760 * annual_average_employees"),
    ("Q_hours_per_emp_soft", "quality", "comparison", "hours/employee should be > 500",
     "annual_average_employees IS NOT NULL AND annual_average_employees > 0 "
     "AND total_hours_worked IS NOT NULL "
     "AND total_hours_worked <= 500 * annual_average_employees"),
    ("Q_avgemp_soft_max", "quality", "range", "annual_average_employees should be < 25000",
     "annual_average_employees IS NOT NULL AND annual_average_employees >= 25000"),

    # --- Cross-field "Additional Data Validations" (hard "must" unless noted) ---
    ("R_sum_types_eq_cases", "validity", "arithmetic",
     "sum(M1..M6) must equal sum(G-J)",
     "(COALESCE(total_injuries,0)+COALESCE(total_skin_disorders,0)+COALESCE(total_respiratory_conditions,0)"
     "+COALESCE(total_poisonings,0)+COALESCE(total_hearing_loss,0)+COALESCE(total_other_illnesses,0)) "
     "<> (COALESCE(total_deaths,0)+COALESCE(total_dafw_cases,0)+COALESCE(total_djtr_cases,0)+COALESCE(total_other_cases,0))"),
    ("Q_avgemp_gt_cases", "quality", "comparison",
     "annual_average_employees should be > total cases (G-J)",
     "annual_average_employees IS NOT NULL AND "
     "annual_average_employees <= (COALESCE(total_deaths,0)+COALESCE(total_dafw_cases,0)+COALESCE(total_djtr_cases,0)+COALESCE(total_other_cases,0))"),
    ("R_dafwcases_implies_days", "validity", "conditional",
     "if dafw_cases>0 then dafw_days>0",
     "COALESCE(total_dafw_cases,0) > 0 AND COALESCE(total_dafw_days,0) = 0"),
    ("R_dafwdays_implies_cases", "validity", "conditional",
     "if dafw_days>0 then dafw_cases>0",
     "COALESCE(total_dafw_days,0) > 0 AND COALESCE(total_dafw_cases,0) = 0"),
    ("R_dafwdays_ge_cases", "validity", "comparison",
     "dafw_days must be >= dafw_cases",
     "COALESCE(total_dafw_days,0) < COALESCE(total_dafw_cases,0)"),
    ("R_djtrcases_implies_days", "validity", "conditional",
     "if djtr_cases>0 then djtr_days>0",
     "COALESCE(total_djtr_cases,0) > 0 AND COALESCE(total_djtr_days,0) = 0"),
    ("R_djtrdays_implies_cases_or_dafw", "validity", "conditional",
     "if djtr_days>0 then (dafw_cases>0 OR djtr_cases>0)",
     "COALESCE(total_djtr_days,0) > 0 AND COALESCE(total_dafw_cases,0) = 0 AND COALESCE(total_djtr_cases,0) = 0"),
]

print(f"{'ID':<28}{'Kind':<10}{'Category':<12}{'Violations':>12}{'%':>8}")
print("-" * 72)
results = []
for rid, kind, cat, desc, pred in rules:
    q = f"SELECT COUNT(*) FROM ita WHERE {pred}"
    n = con.execute(q).fetchone()[0]
    pct = 100.0 * n / n_rows
    print(f"{rid:<28}{kind:<10}{cat:<12}{n:>12}{pct:>7.2f}%")
    results.append((rid, kind, cat, desc, n, round(pct, 3)))

with open("edit_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "kind", "category", "description", "violations", "pct"])
    w.writerows(results)

print(f"\nTotal establishments: {n_rows}")
print("Wrote edit_results.csv")
