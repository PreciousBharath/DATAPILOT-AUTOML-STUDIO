"""DataPilot — AutoML & EDA studio: upload any CSV, get an instant visual data profile, train & compare
5 ML models with cross-validation in one click, inspect feature importance and make live predictions.
Run:  python app.py   →  http://127.0.0.1:5000   (demo: demo@example.com / Demo@1234)
"""
import json
import os
import pickle

import pandas as pd
from flask import abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from core import create_app, execute, login_required, now, query
from engine import SAMPLES, profile, run_automl, sample_dataset

UI = dict(
    name="DataPilot", short="AutoML & EDA Studio", icon="🧪", accent1="#F59E0B", accent2="#EF4444", home="datasets",
    tagline="Drop in a CSV and get a complete exploratory analysis, then train and compare machine-learning models with one click — no code.",
    hero='From raw CSV to <span class="gradient-text">trained model in 60 seconds.</span>', badge="No-code data science",
    features=[("bi-cloud-upload", "Drag & drop CSV", "Automatic types, missing values, outliers & duplicates"),
              ("bi-bar-chart-line", "Visual EDA", "Histograms, correlation heatmap and auto insights"),
              ("bi-cpu", "One-click AutoML", "5 models, cross-validation, leaderboard & live predictions")],
    float=['🏆 Best model: <b>Gradient Boosting</b> · R² 0.93', '📊 <b class="gradient-text">800 rows × 7 cols</b> profiled'],
    nav=[("Datasets", "datasets", "bi-database"), ("Experiments", "experiments", "bi-trophy")],
    search_hint="Search datasets…", promo_title="Tip", promo="Try a sample dataset first — then upload your own CSV (up to 16 MB).",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, path TEXT NOT NULL, rows INTEGER, cols INTEGER, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE, target TEXT NOT NULL, task TEXT, best_model TEXT,
    best_score REAL, result TEXT, created_at TEXT NOT NULL);
"""


def data_dir(uid):
    d = os.path.join(current_app.instance_path, "data", str(uid))
    os.makedirs(d, exist_ok=True)
    return d


def save_df(uid, name, df):
    did = execute("INSERT INTO datasets(user_id,name,path,rows,cols,created_at) VALUES(?,?,?,?,?,?)", (uid, name, "", len(df), df.shape[1], now()))
    path = os.path.join(data_dir(uid), f"{did}.csv")
    df.to_csv(path, index=False)
    execute("UPDATE datasets SET path=? WHERE id=?", (path, did))
    return did


def seed_demo(uid):
    with app.app_context():
        for key in SAMPLES:
            save_df(uid, SAMPLES[key][0], sample_dataset(key))


app = create_app(__name__, UI, SCHEMA)
with app.app_context():  # seed after app exists (needs instance path)
    from core import DEMO_EMAIL
    demo = query("SELECT id FROM users WHERE email=?", (DEMO_EMAIL,), one=True)
    if demo and not query("SELECT 1 FROM datasets WHERE user_id=?", (demo["id"],), one=True):
        seed_demo(demo["id"])

_models = {}


def dataset_or_404(did):
    ds = query("SELECT * FROM datasets WHERE id=? AND user_id=?", (did, g.user["id"]), one=True)
    if not ds or not os.path.exists(ds["path"]):
        abort(404)
    return ds


def model_path(eid):
    return os.path.join(data_dir(g.user["id"]), f"model_{eid}.pkl")


@app.route("/datasets")
@login_required
def datasets():
    rows = query("SELECT d.*, (SELECT COUNT(*) FROM experiments e WHERE e.dataset_id=d.id) n_exp FROM datasets d WHERE user_id=? ORDER BY id DESC", (g.user["id"],))
    return render_template("datasets.html", rows=rows, SAMPLES=SAMPLES, page_title="Datasets")


@app.post("/upload")
@login_required
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        flash("Please choose a .csv file.", "danger")
        return redirect(url_for("datasets"))
    try:
        df = pd.read_csv(f)
    except Exception as e:  # malformed CSV
        flash(f"Could not read CSV: {e}", "danger")
        return redirect(url_for("datasets"))
    if df.empty or df.shape[1] < 2:
        flash("The file needs at least 2 columns and some rows.", "danger")
        return redirect(url_for("datasets"))
    did = save_df(g.user["id"], secure_filename(f.filename)[:80], df.head(200000))
    flash(f"Uploaded {len(df):,} rows × {df.shape[1]} columns ✨", "success")
    return redirect(url_for("eda", did=did))


@app.post("/sample/<key>")
@login_required
def add_sample(key):
    if key not in SAMPLES:
        abort(404)
    did = save_df(g.user["id"], SAMPLES[key][0], sample_dataset(key))
    return redirect(url_for("eda", did=did))


@app.post("/dataset/<int:did>/delete")
@login_required
def delete_dataset(did):
    ds = dataset_or_404(did)
    os.remove(ds["path"])
    execute("DELETE FROM datasets WHERE id=?", (did,))
    flash("Dataset deleted.", "info")
    return redirect(url_for("datasets"))


@app.route("/eda/<int:did>")
@login_required
def eda(did):
    ds = dataset_or_404(did)
    return render_template("eda.html", ds=ds, p=profile(pd.read_csv(ds["path"])), page_title=ds["name"][:40])


@app.route("/automl/<int:did>")
@login_required
def automl(did):
    ds = dataset_or_404(did)
    cols = pd.read_csv(ds["path"], nrows=200).columns.tolist()
    return render_template("automl.html", ds=ds, cols=cols, page_title="AutoML")


@app.post("/api/automl/<int:did>")
@login_required
def api_automl(did):
    ds = dataset_or_404(did)
    d = request.get_json(force=True)
    df = pd.read_csv(ds["path"])
    if d.get("target") not in df.columns:
        return jsonify(error="Choose a target column"), 400
    try:
        res, pipe = run_automl(df, d["target"], d.get("drop", []))
    except Exception as e:
        return jsonify(error=f"Training failed: {e}"), 400
    eid = execute("INSERT INTO experiments(user_id,dataset_id,target,task,best_model,best_score,result,created_at) VALUES(?,?,?,?,?,?,?,?)",
                  (g.user["id"], did, d["target"], res["task"], res["best"], res["board"][0]["cv"], json.dumps(res), now()))
    with open(model_path(eid), "wb") as fh:
        pickle.dump(pipe, fh)
    res["experiment_id"] = eid
    return jsonify(res)


@app.post("/api/predict/<int:eid>")
@login_required
def api_predict(eid):
    ex = query("SELECT * FROM experiments WHERE id=? AND user_id=?", (eid, g.user["id"]), one=True)
    if not ex or not os.path.exists(model_path(eid)):
        return jsonify(error="Model not found — train again"), 404
    with open(model_path(eid), "rb") as fh:
        pipe = pickle.load(fh)  # model files are created by this app only
    schema = json.loads(ex["result"])["schema"]
    vals = request.get_json(force=True)
    row = {c["name"]: (float(vals.get(c["name"], c["default"])) if c["type"] == "numeric" else str(vals.get(c["name"], c["default"]))) for c in schema}
    X = pd.DataFrame([row])
    out = dict(prediction=str(pipe.predict(X)[0]) if ex["task"] == "classification" else round(float(pipe.predict(X)[0]), 3))
    if ex["task"] == "classification" and hasattr(pipe, "predict_proba"):
        out["proba"] = {str(c): round(float(p), 3) for c, p in zip(pipe.classes_, pipe.predict_proba(X)[0])}
    return jsonify(out)


@app.route("/experiments")
@login_required
def experiments():
    rows = query("SELECT e.*, d.name dname FROM experiments e JOIN datasets d ON d.id=e.dataset_id WHERE e.user_id=? ORDER BY e.id DESC", (g.user["id"],))
    return render_template("experiments.html", rows=rows, page_title="Experiments")


@app.route("/experiment/<int:eid>")
@login_required
def experiment(eid):
    ex = query("SELECT e.*, d.name dname FROM experiments e JOIN datasets d ON d.id=e.dataset_id WHERE e.id=? AND e.user_id=?", (eid, g.user["id"]), one=True)
    if not ex:
        abort(404)
    ds = dataset_or_404(ex["dataset_id"])
    cols = pd.read_csv(ds["path"], nrows=5).columns.tolist()
    return render_template("automl.html", ds=ds, cols=cols, preload=dict(json.loads(ex["result"]), experiment_id=eid, target=ex["target"]), page_title="Experiment")


if __name__ == "__main__":
    app.run(debug=True)
