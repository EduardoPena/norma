import duckdb
import csv

CSV_PATH = "fmcsa.csv"

con = duckdb.connect()
con.execute(f"""
    CREATE TABLE fm AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', header=True, sample_size=200000, ignore_errors=True)
""")
n_rows = con.execute("SELECT COUNT(*) FROM fm").fetchone()[0]
n_active = con.execute("SELECT COUNT(*) FROM fm WHERE STATUS_CODE='A'").fetchone()[0]
print(f"Total carrier records: {n_rows}  (Active: {n_active})\n")

rules = [
    # --- 49 CFR 390.19: biennial MCS-150 update ---
    ("R_mcs150_stale", "validity", "comparison",
     "Active carrier: MCS-150 last updated > 24 months ago (49 CFR 390.19)",
     "STATUS_CODE='A' AND MCS150_DATE IS NOT NULL AND "
     "datediff('day', strptime(substr(MCS150_DATE,1,8),'%Y%m%d'), current_date) > 730"),
    ("R_mcs150_missing", "validity", "null",
     "Active carrier has no MCS-150 filing date on record",
     "STATUS_CODE='A' AND MCS150_DATE IS NULL"),

    # --- Operating authority for for-hire carriers (49 U.S.C. 13902 / 49 CFR 365) ---
    ("R_forhire_no_docket", "validity", "null",
     "Active for-hire carrier (CLASSDEF contains AUTHORIZED FOR HIRE) has no operating authority docket number",
     "STATUS_CODE='A' AND CLASSDEF LIKE '%AUTHORIZED FOR HIRE%' "
     "AND DOCKET1 IS NULL AND DOCKET2 IS NULL AND DOCKET3 IS NULL"),

    # --- Domain / enum checks ---
    ("R_status_dom", "validity", "enum", "STATUS_CODE must be in {A,I,P}",
     "STATUS_CODE IS NOT NULL AND STATUS_CODE NOT IN ('A','I','P')"),
    ("R_carrieroperation_dom", "validity", "enum", "CARRIER_OPERATION must be in {A,B,C}",
     "CARRIER_OPERATION IS NOT NULL AND CARRIER_OPERATION NOT IN ('A','B','C')"),
    ("R_hmind_dom", "validity", "enum", "HM_Ind must be in {Y,N}",
     "HM_Ind IS NOT NULL AND HM_Ind NOT IN ('Y','N')"),
    ("R_priorrevoke_dom", "validity", "enum", "PRIOR_REVOKE_FLAG must be in {Y,N}",
     "PRIOR_REVOKE_FLAG IS NOT NULL AND PRIOR_REVOKE_FLAG NOT IN ('Y','N')"),
    ("R_safetyrating_dom", "validity", "enum", "SAFETY_RATING must be in {S,C,U} when present",
     "SAFETY_RATING IS NOT NULL AND SAFETY_RATING NOT IN ('S','C','U')"),

    # --- Range checks (fleet/driver counts cannot be negative) ---
    ("R_truckunits_nonneg", "validity", "range", "TRUCK_UNITS must be >= 0",
     "TRUCK_UNITS IS NOT NULL AND TRUCK_UNITS < 0"),
    ("R_powerunits_nonneg", "validity", "range", "POWER_UNITS must be >= 0",
     "POWER_UNITS IS NOT NULL AND POWER_UNITS < 0"),
    ("R_busunits_nonneg", "validity", "range", "BUS_UNITS must be >= 0",
     "BUS_UNITS IS NOT NULL AND BUS_UNITS < 0"),
    ("R_totaldrivers_nonneg", "validity", "range", "TOTAL_DRIVERS must be >= 0",
     "TOTAL_DRIVERS IS NOT NULL AND TOTAL_DRIVERS < 0"),
    ("R_totalcdl_nonneg", "validity", "range", "TOTAL_CDL must be >= 0",
     "TOTAL_CDL IS NOT NULL AND TOTAL_CDL < 0"),

    # --- Cross-field consistency (quality-tier: plausible internal consistency, not literal CFR text) ---
    ("Q_cdl_le_drivers", "quality", "comparison",
     "TOTAL_CDL should not exceed TOTAL_DRIVERS",
     "TOTAL_CDL IS NOT NULL AND TOTAL_DRIVERS IS NOT NULL AND TOTAL_CDL > TOTAL_DRIVERS"),
    ("Q_intrastate_le_total", "quality", "comparison",
     "TOTAL_INTRASTATE_DRIVERS should not exceed TOTAL_DRIVERS",
     "TOTAL_INTRASTATE_DRIVERS IS NOT NULL AND TOTAL_DRIVERS IS NOT NULL "
     "AND TOTAL_INTRASTATE_DRIVERS > TOTAL_DRIVERS"),
    ("Q_fleet_zero_but_active", "quality", "conditional",
     "Active for-hire carrier reports zero total power units (truck+bus)",
     "STATUS_CODE='A' AND CLASSDEF LIKE '%AUTHORIZED FOR HIRE%' "
     "AND COALESCE(TRUCK_UNITS,0)=0 AND COALESCE(BUS_UNITS,0)=0 AND COALESCE(POWER_UNITS,0)=0"),
]

print(f"{'ID':<26}{'Kind':<10}{'Category':<12}{'Violations':>12}{'%':>8}")
print("-" * 70)
results = []
for rid, kind, cat, desc, pred in rules:
    q = f"SELECT COUNT(*) FROM fm WHERE {pred}"
    n = con.execute(q).fetchone()[0]
    denom = n_active if ("STATUS_CODE='A'" in pred) else n_rows
    pct = 100.0 * n / denom
    print(f"{rid:<26}{kind:<10}{cat:<12}{n:>12}{pct:>7.2f}%")
    results.append((rid, kind, cat, desc, n, round(pct, 3), denom))

with open("edit_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "kind", "category", "description", "violations", "pct", "denominator"])
    w.writerows(results)

print(f"\nTotal records: {n_rows} (Active: {n_active})")
print("Wrote edit_results.csv")
