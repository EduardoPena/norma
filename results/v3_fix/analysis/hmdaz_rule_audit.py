"""HMDAZ rule-by-rule audit of the E2 mappings (Fig. 5 runs): for each model,
mapper and gold rule, whether the predicted DC matches the gold one and, if
not, why. Single-Prompt from results/v3 (HMDAZ has no pair DCs, so the
re-evaluation does not change it), Multi-Step from results/v3_fix.
Writes results/v3_fix/analysis/hmdaz_rule_audit.csv."""
import ast, csv, glob, json, re

GT = json.load(open("data/hmdaz/hmdaz_text2rule_gt.json"))["mapping"]
SID = {s: e["id"] for e in GT for s in e["sentences"]}
COVERAGE = {"business_or_commercial_purpose", "occupancy_type", "reverse_mortgage", "open_end_line_of_credit"}


def preds(rule):
    out = []
    def walk(n):
        if isinstance(n, str): out.append(n)
        elif isinstance(n, dict): [walk(v) for v in n.values()]
        elif isinstance(n, list): [walk(v) for v in n]
    walk(rule)
    return out


def cols(rule):
    return {c for p in preds(rule) for c in re.findall(r"t[12]\.([A-Za-z_]+)", p)}


def base(c):
    return c[:-5] if c.endswith("_name") else c


def classify(row):
    if not row["pred_constraint"]:
        return "not mapped", ""
    gold, pred = ast.literal_eval(row["gold_constraint"]), ast.literal_eval(row["pred_constraint"])
    gc, pc = cols(gold), cols(pred)
    gb, pb = {base(c) for c in gc}, {base(c) for c in pc}
    if row["pairwise_syntactic_exact_match"] == "True":
        return "match (exact)", ""
    form = []
    if any(c.endswith("_name") for c in gc) and any((c + "_name") in gc and c in pc or (c in pb and c + "_name" in gc) for c in pb):
        form.append("code vs label")
    if any(" IN " in p for p in preds(pred)) != any(" IN " in p for p in preds(gold)):
        form.append("IN vs OR")
    missing_cov = sorted((gb & COVERAGE) - pb)
    missing_other = sorted((gb - COVERAGE) - pb)
    extra = sorted(pb - gb)
    detail = "; ".join(x for x in [
        f"missing coverage: {', '.join(missing_cov)}" if missing_cov else "",
        f"missing condition: {', '.join(missing_other)}" if missing_other else "",
        f"extra columns: {', '.join(extra)}" if extra else "",
        f"form: {', '.join(form)}" if form else ""] if x)
    if row["pairwise_logical_match"] == "True":
        return "match (equivalent form)", detail
    if row["pairwise_violation_exact_match"] == "True":
        return "same tuples, not equivalent", detail
    if missing_cov:
        return "wrong: coverage missing", detail
    return "wrong: other", detail


rows = []
for kind, root in [("Single-Prompt", "results/v3"), ("Multi-Step", "results/v3_fix")]:
    tag = "singleprompt" if kind == "Single-Prompt" else "multistep"
    for f in sorted(glob.glob(f"{root}/e2_mapping/hmda_e2_hmdaz_*{tag}*/metrics/*detailed*.csv")):
        model = f.split("|_")[1].replace("_detailed.csv", "").replace("_", " ")
        for r in csv.DictReader(open(f)):
            verdict, detail = classify(r)
            rows.append({"mapper": kind, "model": model, "id": SID[r["sentence"]], "verdict": verdict,
                         "detail": detail, "gold_violations": r["num_gold_violations"],
                         "pred_violations": r["num_pred_violations"], "pred_dc": r["pred_constraint"]})
with open("results/v3_fix/analysis/hmdaz_rule_audit.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
