"""
Spend-Wise AI  -  ML core
=========================
Sab kuch lab manual ke concepts se:
  Lab 1  : Data Preprocessing (cleaning, encoding, scaling, train/test split)
  Lab 2  : Linear Regression  (monthly spending forecast)
  Lab 4  : Decision Tree      (transaction category classifier)
  Lab 5  : Random Forest      (better classifier + comparison with Decision Tree)
  Lab 10 : K-Means Clustering (spending-behaviour of days)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, silhouette_score)

SEED = 42          # models ke liye
DATA_SEED = 9      # synthetic data ke liye
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Education", "Health"]
PAYMENTS = ["Cash", "Card", "Wallet"]
NUM_COLS = ["amount", "hour", "day_of_week", "day_of_month"]

# amounts in PKR; peaks = (hour, spread) jab us category ka kharcha zyada hota hai
PROFILES = {
    "Food":          dict(mu=700,  sd=157,  floor=80,  peaks=[(13, 1.5), (20, 2.0)], pay=[0.50, 0.20, 0.30]),
    "Transport":     dict(mu=400,  sd=112,  floor=50,  peaks=[(8, 1.5), (17, 1.5)],  pay=[0.40, 0.10, 0.50]),
    "Shopping":      dict(mu=4500, sd=1125, floor=500, peaks=[(18, 3.0)],            pay=[0.20, 0.60, 0.20]),
    "Bills":         dict(mu=6000, sd=1350, floor=800, peaks=[(12, 2.5)],            pay=[0.20, 0.30, 0.50]),
    "Entertainment": dict(mu=1800, sd=405,  floor=300, peaks=[(21, 2.0)],            pay=[0.30, 0.40, 0.30]),
    "Education":     dict(mu=3500, sd=675, floor=500, peaks=[(11, 2.5)],            pay=[0.30, 0.30, 0.40]),
    "Health":        dict(mu=2200, sd=675, floor=200, peaks=[(15, 3.0)],            pay=[0.50, 0.30, 0.20]),
}
BASE_W = dict(zip(CATEGORIES, [0.38, 0.22, 0.10, 0.05, 0.08, 0.08, 0.09]))


# ----------------------------------------------------------------------------
# 0. DATA (synthetic, taake internet / download ki zaroorat na ho)
# ----------------------------------------------------------------------------
def _hour_probs(peaks):
    h = np.arange(24)
    p = np.full(24, 0.01)
    for c, w in peaks:
        p += np.exp(-0.5 * ((h - c) / w) ** 2)
    return p / p.sum()


def generate_data(seed=DATA_SEED, start="2024-01-01", end="2025-12-31"):
    """start se end tak ke personal transactions. Jaanbujh kar missing values
    aur duplicates daale gaye hain taake Lab 1 (cleaning) dikhaya ja sake."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    hour_p = {c: _hour_probs(PROFILES[c]["peaks"]) for c in CATEGORIES}
    rows = []
    for d in dates:
        midx = (d.year - dates[0].year) * 12 + d.month - 1
        growth = 1 + 0.025 * midx                       # mehngai / lifestyle: har mahine ~2% barhti hai
        w = np.array([BASE_W[c] for c in CATEGORIES], float)
        if d.dayofweek >= 5:                           # weekend: shopping/entertainment zyada
            w *= [1.1, 0.6, 1.8, 1.0, 2.0, 0.2, 0.8]
        w[3] *= 3 if d.day <= 10 else 0.3              # bills mahine ki shuru ki tareekhon mein
        w /= w.sum()
        for _ in range(rng.poisson(4.5)):
            cat = str(rng.choice(CATEGORIES, p=w))
            p = PROFILES[cat]
            season = 1.6 if (cat == "Shopping" and d.month in (3, 12)) else 1.0   # Eid / Dec sale
            mean_ = p["mu"] * growth * season
            amt = min(mean_ + 2 * p["sd"] * growth, max(p["floor"], rng.normal(mean_, p["sd"] * growth)))   # outliers capped
            rows.append((d, cat, float(round(amt / 10) * 10),
                         int(rng.choice(24, p=hour_p[cat])),
                         str(rng.choice(PAYMENTS, p=p["pay"]))))
    df = pd.DataFrame(rows, columns=["date", "category", "amount", "hour", "payment_method"])
    n = len(df)
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "amount"] = np.nan
    df.loc[rng.choice(n, int(0.04 * n), replace=False), "payment_method"] = np.nan
    df.loc[rng.choice(n, int(0.01 * n), replace=False), "category"] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)])     # duplicates
    return df.sort_values("date", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# LAB 1 : DATA PREPROCESSING
# ----------------------------------------------------------------------------
def clean_data(raw):
    """Cleaning: duplicates, missing values (median / mode / drop) + feature engineering."""
    rep = {"rows_raw": len(raw), "missing_before": raw.isna().sum().to_dict()}
    df = raw.drop_duplicates().copy()
    rep["duplicates_removed"] = len(raw) - len(df)

    n0 = len(df)                                   # Embarked jaisa: category missing -> row drop
    df = df.dropna(subset=["category"]).copy()
    rep["dropped_missing_category"] = n0 - len(df)

    rep["amount_filled"] = int(df["amount"].isna().sum())      # Age jaisa: numeric -> median
    df["amount"] = df["amount"].fillna(df.groupby("category")["amount"].transform("median"))

    rep["payment_filled"] = int(df["payment_method"].isna().sum())   # categorical -> mode
    df["payment_method"] = df["payment_method"].fillna(df["payment_method"].mode()[0])

    df["date"] = pd.to_datetime(df["date"])
    start = df["date"].min()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["month_idx"] = (df["date"].dt.year - start.year) * 12 + df["date"].dt.month - start.month
    rep["missing_after"] = df.isna().sum().to_dict()
    rep["rows_clean"] = len(df)
    return df.reset_index(drop=True), rep


def prepare_classification(df):
    """Encoding (One-Hot payment, Label category) + StandardScaler + 80/20 split."""
    pay = pd.get_dummies(df["payment_method"], prefix="pay")
    for p in PAYMENTS:
        if f"pay_{p}" not in pay:
            pay[f"pay_{p}"] = 0
    pay = pay[[f"pay_{p}" for p in PAYMENTS]]
    X = pd.concat([df[NUM_COLS + ["is_weekend"]], pay], axis=1).astype(float)

    le = LabelEncoder().fit(CATEGORIES)
    y = le.transform(df["category"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_tr[NUM_COLS])             # sirf train pe fit (no leakage)
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[NUM_COLS] = scaler.transform(X_tr[NUM_COLS])
    X_te_s[NUM_COLS] = scaler.transform(X_te[NUM_COLS])
    return dict(X_train=X_tr_s, X_test=X_te_s, y_train=y_tr, y_test=y_te,
                scaler=scaler, classes=list(le.classes_), columns=list(X.columns))


# ----------------------------------------------------------------------------
# LAB 4 + LAB 5 : DECISION TREE vs RANDOM FOREST
# ----------------------------------------------------------------------------
def train_classifiers(prep):
    models = {
        "Decision Tree (no limit)": DecisionTreeClassifier(random_state=SEED),
        "Decision Tree (max_depth=6)": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    }
    out = {}
    for name, m in models.items():
        m.fit(prep["X_train"], prep["y_train"])
        pred = m.predict(prep["X_test"])
        out[name] = dict(
            model=m,
            train_acc=accuracy_score(prep["y_train"], m.predict(prep["X_train"])),
            test_acc=accuracy_score(prep["y_test"], pred),
            cm=confusion_matrix(prep["y_test"], pred),
            report=pd.DataFrame(classification_report(
                prep["y_test"], pred, target_names=prep["classes"],
                output_dict=True, zero_division=0)).T,
            importances=pd.Series(m.feature_importances_, index=prep["columns"]).sort_values(),
        )
    return out


def predict_category(model, prep, amount, hour, day_of_week, day_of_month, payment):
    row = {"amount": amount, "hour": hour, "day_of_week": day_of_week,
           "day_of_month": day_of_month, "is_weekend": int(day_of_week >= 5)}
    for p in PAYMENTS:
        row[f"pay_{p}"] = int(p == payment)
    x = pd.DataFrame([row])[prep["columns"]].astype(float)
    x[NUM_COLS] = prep["scaler"].transform(x[NUM_COLS])
    return pd.Series(model.predict_proba(x)[0], index=prep["classes"]).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# LAB 2 : LINEAR REGRESSION  (spending forecast)
# ----------------------------------------------------------------------------
def train_forecast(df):
    """Simple LR (sirf month_idx)  vs  Multiple LR (month_idx + is_festive).
    is_festive = March (Eid) / December (sale) -> pehle se maloom hota hai, isliye forecast mein use ho sakta hai."""
    monthly = df.groupby("month_idx").agg(total=("amount", "sum"), month=("month", "first")).reset_index()
    monthly["is_festive"] = monthly["month"].isin([3, 12]).astype(int)
    y = monthly["total"]
    feats = {"Simple LR": ["month_idx"], "Multiple LR": ["month_idx", "is_festive"]}

    models, fitted = {}, {}
    for name, cols in feats.items():
        X = monthly[cols]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)  # time order
        pred = LinearRegression().fit(X_tr, y_tr).predict(X_te)
        mse = mean_squared_error(y_te, pred)
        final = LinearRegression().fit(X, y)                   # poore data pe refit
        models[name] = dict(cols=cols, final=final, mse=mse, rmse=float(np.sqrt(mse)),
                            r2=r2_score(y_te, pred), n_test=len(X_te),
                            mape=float(np.mean(np.abs(y_te - pred) / y_te)),     # average % error (hold-out)
                            r2_all=r2_score(y, final.predict(X)),                # fit quality on all months
                            coef=dict(zip(cols, final.coef_)), intercept=float(final.intercept_))
        fitted[name] = final.predict(X)

    nxt = int(monthly["month_idx"].max()) + 1
    nxt_month = int(monthly["month"].iloc[-1]) % 12 + 1
    row = pd.DataFrame({"month_idx": [nxt], "is_festive": [int(nxt_month in (3, 12))]})
    best = models["Multiple LR"]
    next_total = float(best["final"].predict(row[best["cols"]])[0])

    cat = df.pivot_table(index="month_idx", columns="category", values="amount",
                         aggfunc="sum", fill_value=0).reset_index()
    cat["is_festive"] = cat["month_idx"].map(monthly.set_index("month_idx")["is_festive"])
    cat_fc = {}
    for c in CATEGORIES:
        m = LinearRegression().fit(cat[["month_idx", "is_festive"]], cat[c])
        cat_fc[c] = max(0.0, float(m.predict(row[["month_idx", "is_festive"]])[0]))

    return dict(monthly=monthly, models=models, fitted=fitted, next_idx=nxt, next_month=nxt_month,
                next_total=next_total, cat_forecast=pd.Series(cat_fc).sort_values(ascending=False))


# ----------------------------------------------------------------------------
# LAB 10 : K-MEANS  (kaisay din guzra? low / medium / high spending days)
# ----------------------------------------------------------------------------
def cluster_days(df, k=3):
    daily = df.groupby("date").agg(total=("amount", "sum"), n_txn=("amount", "size"),
                                   avg_txn=("amount", "mean")).reset_index()
    cols = ["total", "n_txn", "avg_txn"]
    Xs = StandardScaler().fit_transform(daily[cols])

    ks = list(range(1, 9))
    inertia = [KMeans(n_clusters=i, n_init=10, random_state=SEED).fit(Xs).inertia_ for i in ks]
    sil = {i: silhouette_score(Xs, KMeans(n_clusters=i, n_init=10, random_state=SEED).fit_predict(Xs))
           for i in range(2, 9)}

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    daily["cluster"] = km.labels_
    g = daily.groupby("cluster")[["total", "n_txn", "avg_txn"]].mean()
    if k == 3:                                     # insaani naam: profile dekh kar
        low = g["total"].idxmin()
        rest = g.drop(index=low)
        big = rest["avg_txn"].idxmax()
        busy = [i for i in rest.index if i != big][0]
        mapping = {low: "Low spending day", big: "Big-purchase day", busy: "Busy day (many purchases)"}
        names = ["Low spending day", "Busy day (many purchases)", "Big-purchase day"]
    else:
        order = g["total"].sort_values().index.tolist()
        names = [f"Level {i + 1}" for i in range(k)]
        mapping = dict(zip(order, names))
    daily["cluster_name"] = daily["cluster"].map(mapping)
    summary = daily.groupby("cluster_name").agg(
        days=("date", "size"), avg_total=("total", "mean"),
        avg_txns=("n_txn", "mean"), avg_per_txn=("avg_txn", "mean")).round(0)
    summary = summary.reindex([n for n in names if n in summary.index])
    return dict(daily=daily, ks=ks, inertia=inertia, silhouette=sil,
                k_silhouette=float(sil[k]) if k in sil else None, summary=summary, names=names)