import io
import os
import secrets
import json
import joblib
import numpy as np
import pandas as pd
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_file)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from CODE.utils.auth import init_auth, login_required, role_required
from utils.preprocess import prepare_features

app = Flask(__name__)
# Set SECRET_KEY in the environment to keep users logged in across restarts.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
init_auth(app)  # multi-user login, roles, /admin/users, /account

COMPANY_NAME = "Acme Corp"

# ---------- Data and model ----------
DATA_PATH = "data/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"
df = pd.read_csv(DATA_PATH)
df["EmployeeID"] = df.index + 1
model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Risk levels use calibrated probabilities (about the real chance of leaving).
# Low: under 20%, Medium: 20% to 40%, High: 40% or more.
RISK_LOW, RISK_HIGH = 0.20, 0.40


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def risk_level(p):
    return "High" if p >= RISK_HIGH else "Medium" if p >= RISK_LOW else "Low"


def build_risk_scores(data):
    """Honest risk scores.

    1. Cross-validation: every employee is scored by a model that never saw
       that employee, so scores are not inflated by memorisation.
    2. Calibration: the model is trained with class_weight='balanced', which
       pushes probabilities up. A small logistic fit maps them back to real
       probabilities (average risk is about the real 16% attrition rate).
    """
    X = prepare_features(data, feature_columns)
    y = (data["Attrition"] == "Yes").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(clone(model), X, y, cv=cv,
                            method="predict_proba")[:, 1]
    cal = LogisticRegression().fit(_logit(oof), y)
    return cal, cal.predict_proba(_logit(oof))[:, 1]


calibrator, _scores = build_risk_scores(df)
df_scored = df.copy()
df_scored["RiskProb"] = _scores.round(3)
df_scored["RiskLevel"] = pd.cut(
    df_scored["RiskProb"], [-0.001, RISK_LOW, RISK_HIGH, 1.001],
    labels=["Low", "Medium", "High"], right=False).astype(str)

# Bands (used by the Explorer "Group by" option)
df_scored["AgeBand"] = pd.cut(df_scored["Age"], [0, 24, 34, 44, 54, 100],
                              labels=["Under 25", "25-34", "35-44", "45-54", "55+"])
df_scored["TenureBand"] = pd.cut(df_scored["YearsAtCompany"], [-1, 2, 5, 10, 100],
                                 labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
df_scored["IncomeBand"] = pd.cut(df_scored["MonthlyIncome"], [0, 3000, 6000, 10000, 10**9],
                                 labels=["Under $3k", "$3k-6k", "$6k-10k", "$10k+"])

FEATURES = df.drop(columns=["Attrition", "EmployeeID"])
DEFAULTS = FEATURES.mode().iloc[0].to_dict()
NUMERIC_COLS = FEATURES.select_dtypes("number").columns.tolist()
ROLE_MEDIAN = df.groupby("JobRole")["MonthlyIncome"].median().to_dict()

KEY_FIELDS = ["Age", "MonthlyIncome", "OverTime", "JobRole", "Department",
              "BusinessTravel", "MaritalStatus", "DistanceFromHome",
              "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
              "WorkLifeBalance", "JobSatisfaction", "StockOptionLevel"]
KEY_FIELDS = [c for c in KEY_FIELDS if c in FEATURES.columns]

ORDERS = [["Low", "Medium", "High", "Very High"],
          ["Bad", "Good", "Better", "Best"]]


def order_options(opts):
    for o in ORDERS:
        if set(opts) <= set(o):
            return [x for x in o if x in opts]
    return sorted(opts)


@app.context_processor
def inject_globals():
    return dict(company=COMPANY_NAME)


def filtered_df():
    d = df_scored
    for col in ["Department", "JobRole", "Gender", "OverTime"]:
        val = request.args.get(col)
        if val and val != "All":
            d = d[d[col] == val]
    for key, col in [("age", "Age"), ("inc", "MonthlyIncome")]:
        lo = request.args.get(key + "_min")
        hi = request.args.get(key + "_max")
        lo, hi = num(lo, None), num(hi, None)
        if lo is not None:
            d = d[d[col] >= lo]
        if hi is not None:
            d = d[d[col] <= hi]
    return d


def current_df():
    """Filtered employees who are still at the company (retention view)."""
    d = filtered_df()
    return d[d["Attrition"] == "No"]


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def risk_factors(r):
    """Rule-based risk factors + suggested retention action."""
    f = []
    if str(r.get("OverTime")) == "Yes":
        f.append(("Works overtime", "Review workload and cap overtime hours"))
    if num(r.get("YearsSinceLastPromotion")) >= 3:
        f.append(("3+ years since last promotion",
                  "Schedule a career-growth and promotion review"))
    med = ROLE_MEDIAN.get(r.get("JobRole"))
    if med and num(r.get("MonthlyIncome")) < 0.85 * med:
        f.append(("Income well below role median",
                  "Run a compensation benchmark for this role"))
    if str(r.get("WorkLifeBalance")).lower() in ("bad", "low"):
        f.append(("Poor work-life balance",
                  "Offer flexible hours or remote-work options"))
    if str(r.get("JobSatisfaction")).lower() == "low":
        f.append(("Low job satisfaction",
                  "Hold a 1-on-1 to understand concerns and role fit"))
    if str(r.get("BusinessTravel")) == "Travel_Frequently":
        f.append(("Frequent business travel",
                  "Rebalance travel load across the team"))
    if num(r.get("DistanceFromHome")) >= 20:
        f.append(("Long commute", "Consider hybrid or transport support"))
    if num(r.get("YearsAtCompany"), 99) <= 2:
        f.append(("New joiner (2 years or less)",
                  "Strengthen onboarding and assign a mentor"))
    return [dict(factor=a, action=b) for a, b in f]


# ---------- Pages ----------
@app.route("/")
@login_required
def home():
    return redirect(url_for("overview"))


@app.route("/overview")
@login_required
def overview():
    return render_template("overview.html")


@app.route("/explorer")
@login_required
def explorer():
    return render_template("explorer.html")


@app.route("/predict")
@role_required("admin", "hr")
def predict_page():
    return render_template("predict.html")


@app.route("/high-risk")
@role_required("admin", "hr")
def high_risk_page():
    return render_template("high_risk.html")


@app.route("/model-performance")
@login_required
def model_performance():
    return render_template("model_performance.html")


# ---------- APIs ----------
@app.route("/api/filters")
@login_required
def api_filters():
    out = {c: sorted(df[c].unique().tolist())
           for c in ["Department", "JobRole", "Gender", "OverTime"]}
    out["ranges"] = {
        "Age": [int(df["Age"].min()), int(df["Age"].max())],
        "MonthlyIncome": [int(df["MonthlyIncome"].min()),
                          int(df["MonthlyIncome"].max())],
    }
    return jsonify(out)


@app.route("/api/alerts")
@login_required
def api_alerts():
    cur = df_scored[df_scored["Attrition"] == "No"]
    high = int((cur["RiskLevel"] == "High").sum())
    return jsonify(high=high, pct=round(high / max(len(cur), 1) * 100, 1))


@app.route("/api/kpis")
@login_required
def api_kpis():
    d = filtered_df()
    if len(d) == 0:
        return jsonify(total=0, attrition_rate=0, avg_income=0, avg_tenure=0,
                       high_risk=0, attrition_cost=0)
    left = int((d["Attrition"] == "Yes").sum())
    avg_income = d["MonthlyIncome"].mean()
    return jsonify(
        total=len(d),
        attrition_rate=round(left / len(d) * 100, 1),
        avg_income=int(avg_income),
        avg_tenure=round(d["YearsAtCompany"].mean(), 1),
        high_risk=int(((d["RiskLevel"] == "High") & (d["Attrition"] == "No")).sum()),
        # Assumption: replacing an employee costs ~50% of annual salary
        attrition_cost=int(left * avg_income * 12 * 0.5),
    )


ALLOWED_GROUPS = ["Department", "JobRole", "OverTime", "MaritalStatus",
                  "WorkLifeBalance", "JobSatisfaction", "BusinessTravel",
                  "Gender", "EducationField", "AgeBand", "TenureBand",
                  "IncomeBand"]


@app.route("/api/attrition-by/<col>")
@login_required
def api_attrition_by(col):
    if col not in ALLOWED_GROUPS:
        return jsonify(error="invalid column"), 400
    d = filtered_df()
    g = d.groupby(col, observed=True).agg(
        total=("Attrition", "size"),
        left=("Attrition", lambda s: (s == "Yes").sum()),
    )
    g["rate"] = (g["left"] / g["total"] * 100).round(1)
    return jsonify(labels=[str(x) for x in g.index.tolist()],
                   rate=g["rate"].tolist(),
                   left=g["left"].astype(int).tolist(),
                   total=g["total"].astype(int).tolist())


@app.route("/api/risk-distribution")
@login_required
def api_risk_distribution():
    vc = current_df()["RiskLevel"].value_counts()
    labels = ["Low", "Medium", "High"]
    return jsonify(labels=labels, values=[int(vc.get(k, 0)) for k in labels])


@app.route("/api/heatmap")
@login_required
def api_heatmap():
    d = filtered_df().copy()
    d["left"] = (d["Attrition"] == "Yes").astype(int)
    piv = d.pivot_table(index="JobRole", columns="Department",
                        values="left", aggfunc="mean") * 100
    values = [[None if pd.isna(v) else round(float(v), 1) for v in row]
              for row in piv.values]
    return jsonify(rows=piv.index.tolist(), cols=piv.columns.tolist(),
                   values=values)


@app.route("/api/insights")
@login_required
def api_insights():
    d = filtered_df()
    if len(d) < 30:
        return jsonify([])
    out = []
    is_left = d["Attrition"] == "Yes"

    ot = d[d["OverTime"] == "Yes"]
    no_ot = d[d["OverTime"] == "No"]
    if len(ot) and len(no_ot):
        a = (ot["Attrition"] == "Yes").mean() * 100
        b = (no_ot["Attrition"] == "Yes").mean() * 100
        if b > 0:
            out.append(f"Employees working overtime leave <b>{a / b:.1f}x</b> "
                       f"more often ({a:.0f}% vs {b:.0f}%).")

    role = d.groupby("JobRole")["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)
    out.append(f"Highest-attrition role: <b>{role.index[0]}</b> "
               f"({role.iloc[0]:.0f}%).")

    if is_left.any() and (~is_left).any():
        gap = d[~is_left]["MonthlyIncome"].mean() - d[is_left]["MonthlyIncome"].mean()
        if gap > 0:
            out.append(f"Employees who left earned <b>${gap:,.0f}</b> less per "
                       f"month on average than those who stayed.")

    new = d[d["YearsAtCompany"] <= 2]
    if len(new):
        rate = (new["Attrition"] == "Yes").mean() * 100
        out.append(f"Employees with 2 years or less at the company have "
                   f"<b>{rate:.0f}%</b> attrition.")

    high = int(((d["RiskLevel"] == "High") & (d["Attrition"] == "No")).sum())
    out.append(f"<b>{high}</b> current employees are flagged as High risk.")
    return jsonify(out)


@app.route("/api/form-schema")
@role_required("admin", "hr")
def api_form_schema():
    out = []
    for c in KEY_FIELDS:
        s = FEATURES[c]
        if pd.api.types.is_numeric_dtype(s):
            out.append(dict(name=c, type="number", min=float(s.min()),
                            max=float(s.max()), default=float(s.median())))
        else:
            out.append(dict(name=c, type="select",
                            options=order_options(s.unique().tolist()),
                            default=str(DEFAULTS[c])))
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
@role_required("admin", "hr")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the employee details as a JSON object"), 400
    unknown = [k for k in payload if k not in FEATURES.columns]
    if unknown:
        return jsonify(error=f"Unknown field(s): {', '.join(unknown)}"), 400
    data = dict(DEFAULTS)
    data.update(payload)
    row = pd.DataFrame([data])
    try:
        for c in NUMERIC_COLS:
            row[c] = pd.to_numeric(row[c])
    except (ValueError, TypeError):
        return jsonify(error="Numeric fields must contain numbers"), 400
    raw = model.predict_proba(prepare_features(row, feature_columns))[0, 1]
    prob = float(calibrator.predict_proba(_logit(raw))[0, 1])
    level = risk_level(prob)
    return jsonify(probability=round(prob * 100, 1), level=level,
                   factors=risk_factors(data))


@app.route("/api/employee/<int:emp_id>")
@role_required("admin", "hr")
def api_employee(emp_id):
    row = df_scored[df_scored["EmployeeID"] == emp_id]
    if row.empty:
        return jsonify(error="not found"), 404
    r = row.iloc[0]
    details = {}
    for c in KEY_FIELDS:
        v = r[c]
        details[c] = v.item() if hasattr(v, "item") else v
    return jsonify(id=emp_id, probability=round(float(r["RiskProb"]) * 100, 1),
                   level=r["RiskLevel"], details=details,
                   factors=risk_factors(r))


@app.route("/api/high-risk")
@role_required("admin", "hr")
def api_high_risk():
    try:
        top = max(1, min(int(request.args.get("top", 20)), 500))
    except ValueError:
        top = 20
    cols = ["EmployeeID", "Age", "Department", "JobRole", "MonthlyIncome",
            "OverTime", "YearsAtCompany", "RiskProb", "RiskLevel"]
    d = current_df().sort_values("RiskProb", ascending=False).head(top)
    return jsonify(d[cols].to_dict(orient="records"))


@app.route("/download/high-risk")
@role_required("admin", "hr")
def download_high_risk():
    d = current_df()
    d = d[d["RiskLevel"] == "High"].sort_values("RiskProb", ascending=False)
    buf = io.BytesIO(d.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="high_risk_employees.csv")


def model_drivers(n=10):
    """Top drivers of the model in use (standardised coefficients if linear)."""
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:
        return None
    imp = pd.Series(values, index=feature_columns).sort_values(ascending=False).head(n)
    return {"labels": [i.replace("_", ": ", 1) for i in imp.index],
            "values": imp.round(3).tolist()}


@app.route("/api/model-metrics")
@login_required
def api_model_metrics():
    with open("models/metrics.json") as f:
        data = json.load(f)
    drivers = model_drivers()
    if drivers:
        data["feature_importance"] = drivers
    return jsonify(data)


@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")