"""EDA profiling + AutoML engine (pandas / scikit-learn)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score, mean_squared_error)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ------------------------------------------------------------------ sample data
def sample_dataset(name, seed=1):
    rng = np.random.default_rng(seed)
    if name == "student":
        n = 600
        hours = rng.gamma(3, 1.6, n).round(1)
        att = rng.normal(82, 10, n).clip(40, 100).round()
        prev = rng.normal(65, 12, n).clip(20, 100).round()
        sleep = rng.normal(7, 1.1, n).clip(4, 10).round(1)
        internet = rng.choice(["Yes", "No"], n, p=[.8, .2])
        parent = rng.choice(["School", "Graduate", "Postgraduate"], n, p=[.4, .4, .2])
        extra = rng.choice(["Sports", "Music", "Coding club", "None"], n)
        score = (4 + 3.2 * hours + .32 * att + .28 * prev + 1.5 * (sleep - 7) + 3 * (internet == "Yes") + 2 * (parent == "Postgraduate")
                 + rng.normal(0, 6, n)).clip(0, 100).round()
        df = pd.DataFrame(dict(study_hours=hours, attendance=att, previous_score=prev, sleep_hours=sleep, internet=internet,
                               parent_education=parent, activity=extra, final_score=score))
        df["result"] = np.where(df.final_score >= 60, "Pass", "Fail")
        df.loc[rng.choice(n, 18, replace=False), "sleep_hours"] = np.nan
        return df
    if name == "housing":
        n = 800
        city = rng.choice(["Bengaluru", "Chennai", "Hyderabad", "Pune", "Madurai"], n, p=[.3, .2, .2, .2, .1])
        area = rng.normal(1200, 380, n).clip(400, 3200).round()
        bhk = np.clip((area / 520).round() + rng.integers(-1, 2, n), 1, 5)
        bath = np.clip(bhk - rng.integers(0, 2, n), 1, 5)
        age = rng.integers(0, 30, n)
        parking = rng.choice(["Yes", "No"], n, p=[.7, .3])
        mult = pd.Series(city).map({"Bengaluru": 1.35, "Chennai": 1.1, "Hyderabad": 1.15, "Pune": 1.2, "Madurai": .7}).values
        price = (area * 6200 * mult * (1 - age * .008) + bhk * 3e5 + (parking == "Yes") * 4e5 + rng.normal(0, 6e5, n)) / 1e5
        return pd.DataFrame(dict(city=city, area_sqft=area, bhk=bhk, bathrooms=bath, age_years=age, parking=parking, price_lakhs=price.round(1)))
    if name == "heart":
        n = 500
        age = rng.integers(29, 78, n)
        chol = rng.normal(235, 45, n).round()
        bp = rng.normal(130, 17, n).round()
        hr = rng.normal(150, 22, n).round()
        smoker = rng.choice(["Yes", "No"], n, p=[.3, .7])
        exang = rng.choice(["Yes", "No"], n, p=[.33, .67])
        logit = -9 + .07 * age + .008 * chol + .018 * bp - .02 * hr + .9 * (smoker == "Yes") + 1.3 * (exang == "Yes")
        risk = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "High", "Low")
        return pd.DataFrame(dict(age=age, cholesterol=chol, resting_bp=bp, max_heart_rate=hr, smoker=smoker, exercise_angina=exang, risk=risk))
    raise ValueError(name)


SAMPLES = {"student": ("🎓 Student Performance", "Predict final score / pass-fail from study habits"),
           "housing": ("🏠 Indian House Prices", "Regression on city, area, BHK & amenities"),
           "heart": ("❤️ Heart Risk (synthetic)", "Classify cardiovascular risk level")}


# ------------------------------------------------------------------ EDA
def profile(df):
    cols = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        info = dict(name=c, type="numeric" if num else "categorical", missing=int(s.isna().sum()),
                    missing_pct=round(100 * s.isna().mean(), 1), unique=int(s.nunique()))
        if num:
            d = s.dropna()
            q1, q3 = d.quantile(.25), d.quantile(.75)
            iqr = q3 - q1
            counts, edges = np.histogram(d, bins=min(20, max(5, d.nunique())))
            info.update(mean=round(float(d.mean()), 2), std=round(float(d.std()), 2), min=round(float(d.min()), 2), max=round(float(d.max()), 2),
                        median=round(float(d.median()), 2), skew=round(float(d.skew()), 2),
                        outliers=int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum()),
                        hist=dict(labels=[f"{edges[i]:.4g}" for i in range(len(counts))], counts=counts.tolist()))
        else:
            vc = s.value_counts().head(10)
            info.update(top=str(vc.index[0]) if len(vc) else "", hist=dict(labels=[str(x) for x in vc.index], counts=vc.values.tolist()))
        cols.append(info)
    num = df.select_dtypes("number")
    corr = num.corr().round(2) if num.shape[1] >= 2 else pd.DataFrame()
    insights = []
    if not corr.empty:
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
        for a, b, v in sorted(pairs, key=lambda p: -abs(p[2]))[:3]:
            insights.append(("bi-link-45deg", f"<b>{a}</b> and <b>{b}</b> are {'strongly' if abs(v) > .6 else 'moderately'} {'positively' if v > 0 else 'negatively'} correlated (r = {v})."))
    for c in cols:
        if c["missing"]:
            insights.append(("bi-exclamation-triangle", f"<b>{c['name']}</b> has {c['missing']} missing values ({c['missing_pct']}%) — will be imputed automatically."))
        if c["type"] == "numeric" and abs(c["skew"]) > 1:
            insights.append(("bi-graph-up", f"<b>{c['name']}</b> is {'right' if c['skew'] > 0 else 'left'}-skewed (skew {c['skew']}) — consider a log transform."))
        if c["type"] == "numeric" and c["outliers"] > len(df) * .02:
            insights.append(("bi-bullseye", f"<b>{c['name']}</b> has {c['outliers']} IQR outliers."))
    dup = int(df.duplicated().sum())
    if dup:
        insights.append(("bi-files", f"{dup} duplicate rows found."))
    return dict(rows=len(df), cols=df.shape[1], numeric=int(num.shape[1]), categorical=int(df.shape[1] - num.shape[1]),
                missing=int(df.isna().sum().sum()), duplicates=dup, memory=round(df.memory_usage(deep=True).sum() / 1024, 1),
                columns=cols, corr=dict(cols=list(corr.columns), values=corr.values.tolist()), insights=insights[:8],
                preview=df.head(8).fillna("").astype(str).values.tolist(), headers=list(df.columns))


# ------------------------------------------------------------------ AutoML
def detect_task(y):
    return "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15 else "classification"


def build_pre(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ]), num, cat


def run_automl(df, target, drop=()):
    df = df.dropna(subset=[target]).drop(columns=[c for c in drop if c in df.columns and c != target])
    X, y = df.drop(columns=[target]), df[target]
    task = detect_task(y)
    pre, num, cat = build_pre(X)
    if task == "classification":
        y = y.astype(str)
        models = {"Logistic Regression": LogisticRegression(max_iter=2000), "Random Forest": RandomForestClassifier(250, random_state=0, n_jobs=-1),
                  "Gradient Boosting": GradientBoostingClassifier(random_state=0), "K-Nearest Neighbours": KNeighborsClassifier(7),
                  "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0)}
        scoring = "f1_macro"
    else:
        models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(1.0), "Random Forest": RandomForestRegressor(250, random_state=0, n_jobs=-1),
                  "Gradient Boosting": GradientBoostingRegressor(random_state=0), "K-Nearest Neighbours": KNeighborsRegressor(7)}
        scoring = "r2"
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=0, stratify=strat)
    board, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("m", m)])
        cv = cross_val_score(pipe, Xtr, ytr, cv=3, scoring=scoring)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        if task == "classification":
            row = dict(model=name, cv=round(float(cv.mean()), 4), cv_std=round(float(cv.std()), 4), accuracy=round(accuracy_score(yte, pred), 4), f1=round(f1_score(yte, pred, average="macro"), 4))
        else:
            row = dict(model=name, cv=round(float(cv.mean()), 4), cv_std=round(float(cv.std()), 4), r2=round(r2_score(yte, pred), 4),
                       mae=round(mean_absolute_error(yte, pred), 3), rmse=round(float(np.sqrt(mean_squared_error(yte, pred))), 3))
        board.append(row)
        fitted[name] = (pipe, pred)
    board.sort(key=lambda r: -r["cv"])
    best = board[0]["model"]
    pipe, pred = fitted[best]
    names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
    m = pipe.named_steps["m"]
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "coef_"):
        imp = np.abs(np.atleast_2d(m.coef_)).mean(axis=0)
    else:
        imp = np.zeros(len(names))
    order = np.argsort(imp)[::-1][:12]
    detail = dict(importance=dict(labels=[names[i] for i in order], values=[round(float(imp[i]), 4) for i in order]))
    if task == "classification":
        labels = sorted(y.unique().tolist())
        detail["cm"] = dict(labels=labels, values=confusion_matrix(yte, pred, labels=labels).tolist())
    else:
        idx = np.random.default_rng(0).choice(len(yte), min(150, len(yte)), replace=False)
        detail["scatter"] = [dict(x=round(float(np.asarray(yte)[i]), 3), y=round(float(pred[i]), 3)) for i in idx]
    schema = [dict(name=c, type="numeric", default=round(float(X[c].median()), 2)) if c in num else
              dict(name=c, type="categorical", options=sorted(X[c].dropna().astype(str).unique().tolist())[:30], default=str(X[c].mode()[0])) for c in X.columns]
    return dict(task=task, metric=scoring, board=board, best=best, detail=detail, schema=schema, n_train=len(Xtr), n_test=len(Xte)), pipe
