"""30-day readmission in diabetic inpatients (UCI 'Diabetes 130-US hospitals', 1999-2008).
Step 1: clean + SQL exploration.  Step 2: logistic regression with odds ratios.  Step 3: charts."""
import sqlite3, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
print("raw encounters:", len(raw), "patients:", raw.patient_nbr.nunique())

# --- cleaning ---------------------------------------------------------------
# keep only each patient's first encounter, so one person is not counted many times
df = raw.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first").copy()
n_first = len(df)
# drop patients who died or went to hospice: they cannot be readmitted
df = df[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])].copy()
print("first encounters:", n_first, "after removing death/hospice:", len(df))
df["readmit30"] = (df.readmitted == "<30").astype(int)

def diag_group(code):
    if pd.isna(code): return "Other"
    c = str(code)
    if c.startswith(("V", "E")): return "Other"
    x = float(c)
    if 390 <= x < 460 or int(x) == 785: return "Circulatory"
    if 460 <= x < 520 or int(x) == 786: return "Respiratory"
    if 520 <= x < 580 or int(x) == 787: return "Digestive"
    if int(x) == 250: return "Diabetes"
    if 800 <= x < 1000: return "Injury"
    if 710 <= x < 740: return "Musculoskeletal"
    if 580 <= x < 630 or int(x) == 788: return "Genitourinary"
    if 140 <= x < 240: return "Neoplasms"
    return "Other"
df["diag_group"] = df.diag_1.map(diag_group)
df["a1c_tested"] = (~df.A1Cresult.isna()).astype(int)
df["med_changed"] = (df.change == "Ch").astype(int)
df["on_diabetes_med"] = (df.diabetesMed == "Yes").astype(int)
df["emergency_admit"] = (df.admission_type_id == 1).astype(int)
df["discharged_home"] = (df.discharge_disposition_id == 1).astype(int)
df["prior_inpatient"] = df.number_inpatient.clip(upper=3)   # 3 means 3 or more
df["age_group"] = df.age.map({"[0-10)":"0-30","[10-20)":"0-30","[20-30)":"0-30","[30-40)":"30-50","[40-50)":"30-50",
                              "[50-60)":"50-70","[60-70)":"50-70","[70-80)":"70-80","[80-90)":"80+","[90-100)":"80+"})
df["gender"] = df.gender.where(df.gender.isin(["Male", "Female"]))

con = sqlite3.connect("data/diabetes.db")
df[["encounter_id","patient_nbr","age_group","gender","race","time_in_hospital","num_medications","number_diagnoses",
    "number_emergency","number_outpatient","prior_inpatient","diag_group","a1c_tested","med_changed","on_diabetes_med",
    "emergency_admit","discharged_home","readmit30"]].to_sql("enc", con, if_exists="replace", index=False)

# --- SQL exploration --------------------------------------------------------
Q = {
 "by_age": "SELECT age_group, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY age_group ORDER BY age_group",
 "by_prior_inpatient": "SELECT prior_inpatient, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY prior_inpatient ORDER BY 1",
 "by_diag_group": "SELECT diag_group, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY diag_group ORDER BY readmit_pct DESC",
 "by_a1c_tested": "SELECT a1c_tested, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY a1c_tested",
 "by_med_changed": "SELECT med_changed, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY med_changed",
 "by_discharge_home": "SELECT discharged_home, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY discharged_home",
 "by_los": "SELECT CASE WHEN time_in_hospital<=2 THEN '1-2' WHEN time_in_hospital<=5 THEN '3-5' WHEN time_in_hospital<=8 THEN '6-8' ELSE '9+' END los_days, COUNT(*) n, ROUND(100.0*AVG(readmit30),1) readmit_pct FROM enc GROUP BY 1 ORDER BY MIN(time_in_hospital)",
}
res = {}
for k, q in Q.items():
    res[k] = pd.read_sql(q, con); res[k].to_csv(f"data/sql_{k}.csv", index=False)
    print("\n", k); print(res[k].to_string(index=False))
overall = df.readmit30.mean()
print("\noverall 30-day readmission:", round(100*overall, 1), "% of", len(df))

# --- logistic regression (numpy IRLS, so no extra packages) -----------------
num = ["time_in_hospital", "num_medications", "number_diagnoses", "number_emergency", "number_outpatient", "prior_inpatient"]
binv = ["a1c_tested", "med_changed", "on_diabetes_med", "emergency_admit", "discharged_home"]
cat = ["age_group", "diag_group", "gender", "race"]
m = df.dropna(subset=cat).copy()
X = pd.get_dummies(m[cat], drop_first=True).astype(float)
Z = m[num].copy(); Z = (Z - Z.mean()) / Z.std()          # standardised: OR is per 1 SD
X = pd.concat([Z, m[binv].astype(float), X], axis=1)
X.insert(0, "const", 1.0)
y = m.readmit30.values
rng = np.random.default_rng(42)
test = rng.random(len(m)) < 0.3
def fit_logit(X, y, iters=30):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8: break
    return b, np.linalg.inv(H)
b, cov = fit_logit(X[~test].values, y[~test])
se = np.sqrt(np.diag(cov))
out = pd.DataFrame({"term": X.columns, "odds_ratio": np.exp(b), "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se),
                    "p_value": 2 * (1 - stats.norm.cdf(np.abs(b / se)))}).iloc[1:]
out.round(4).to_csv("data/odds_ratios.csv", index=False)
print("\nodds ratios (training set):"); print(out.round(3).sort_values("odds_ratio", ascending=False).to_string(index=False))
score = 1 / (1 + np.exp(-X[test].values @ b))
def auc(y, s):
    r = stats.rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
A = auc(y[test], score)
print("\nhold-out AUC:", round(A, 3), "on", test.sum(), "patients; train", (~test).sum())
open("data/model_summary.txt", "w").write(f"patients used: {len(m)}\ntrain: {(~test).sum()}  test: {test.sum()}\nhold-out AUC: {A:.3f}\noverall 30-day readmission: {100*overall:.1f}%\n")

# --- charts -----------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, (k, col, title) in zip(axs.ravel(), [("by_age", "age_group", "By age group"), ("by_prior_inpatient", "prior_inpatient", "By inpatient stays in the previous year (3 = 3+)"),
                                              ("by_diag_group", "diag_group", "By main diagnosis"), ("by_los", "los_days", "By length of stay (days)")]):
    t = res[k]; ax.bar(t[col].astype(str), t.readmit_pct, color="#2a7f9e")
    ax.axhline(100 * overall, color="k", ls="--", lw=1); ax.set_title(title, fontsize=10); ax.set_ylabel("30-day readmission %")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
fig.suptitle(f"30-day readmission, diabetic inpatients (dashed line = overall {100*overall:.1f}%)")
plt.tight_layout(); plt.savefig("figures/readmission_rates.png", dpi=150); plt.close()

o = out.sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(9, 8))
ax.errorbar(o.odds_ratio, range(len(o)), xerr=[o.odds_ratio - o.ci_low, o.ci_high - o.odds_ratio], fmt="o", color="#2a7f9e", ecolor="gray", capsize=2)
ax.axvline(1, color="k", ls="--"); ax.set_yticks(range(len(o))); ax.set_yticklabels(o.term, fontsize=8)
ax.set_xlabel("odds ratio (95% CI). Numeric variables are per 1 standard deviation"); ax.set_title("What goes with 30-day readmission (logistic regression)")
plt.tight_layout(); plt.savefig("figures/odds_ratios.png", dpi=150); plt.close()
