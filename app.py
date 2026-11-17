"""
Spend-Wise AI  -  real-time personal expense tracker with machine learning
Run:  streamlit run app.py
"""
import calendar
import os
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import ml_core as ml

st.set_page_config(page_title="Spend-Wise AI", page_icon="💸", layout="wide")
sns.set_theme(style="whitegrid")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "transactions.csv")          # transaction history lives here
TODAY = date.today()
CUR = pd.Timestamp(TODAY).to_period("M")             # current month
AUTO = "🤖 Auto (AI decides)"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def rs(x):
    return f"Rs {x:,.0f}"


# ============================================================ storage
def seed_history():
    """Pehli baar chalane par ek sample history bana deta hai (last ~2 saal, aaj tak)."""
    raw = ml.generate_data(start=f"{TODAY.year - 1}-01-01", end=TODAY.isoformat())
    df, _ = ml.clean_data(raw)
    df = df[["date", "category", "amount", "hour", "payment_method"]].reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    df["source"] = "sample"
    df["ai_guess"] = np.nan
    df.to_csv(DB, index=False)


def read_db():
    if not os.path.exists(DB):
        seed_history()
    return pd.read_csv(DB, parse_dates=["date"])


def add_transaction(d, hour, amount, category, payment, ai_guess):
    tx = read_db()
    row = {"id": int(tx["id"].max()) + 1, "date": pd.Timestamp(d), "category": category,
           "amount": float(amount), "hour": int(hour), "payment_method": payment,
           "source": "you", "ai_guess": ai_guess}
    pd.concat([tx, pd.DataFrame([row])], ignore_index=True).to_csv(DB, index=False)


def undo_last():
    tx = read_db()
    mine = tx[tx["source"] == "you"]
    if mine.empty:
        return False
    tx.drop(index=mine["id"].idxmax()).to_csv(DB, index=False)
    return True


# ============================================================ ML pipeline (retrains when history changes)
@st.cache_resource(show_spinner="🤖 AI is learning from your latest data...")
def build(version):
    tx = read_db()
    df, _ = ml.clean_data(tx[["id", "date", "category", "amount", "hour", "payment_method"]])
    df["ym"] = df["date"].dt.to_period("M")
    prep = ml.prepare_classification(df)
    clf = ml.train_classifiers(prep)
    fc = ml.train_forecast(df[df["ym"] < CUR])        # forecast sirf poore mahinon se seekhta hai
    km = ml.cluster_days(df)
    return tx, df, prep, clf, fc, km


read_db()                                             # file maujood honi chahiye
tx, df, prep, clf, fc, km = build(os.path.getmtime(DB))
rf = clf["Random Forest"]["model"]

if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

# ============================================================ sidebar : add transaction
with st.sidebar:
    st.title("💸 Spend-Wise AI")
    st.caption("Real-time expense tracker powered by machine learning")
    st.subheader("➕ Add a transaction")
    amount = st.number_input("Amount (Rs)", min_value=10, max_value=1000000, value=1000, step=100)
    choice = st.selectbox("Category", [AUTO] + ml.CATEGORIES)
    payment = st.selectbox("Payment method", ml.PAYMENTS)
    d = st.date_input("Date", value=TODAY, max_value=TODAY)
    hour = st.slider("Hour of day", 0, 23, datetime.now().hour)

    probs = ml.predict_category(rf, prep, float(amount), int(hour), d.weekday(), d.day, payment)
    guess = probs.index[0]
    st.info(f"🤖 AI suggests **{guess}** ({probs.iloc[0]:.0%} sure)")

    if st.button("Add transaction", type="primary"):
        final = guess if choice == AUTO else choice
        add_transaction(d, hour, amount, final, payment, None if choice == AUTO else guess)
        st.session_state["flash"] = f"✅ Saved {rs(amount)} under {final}"
        st.rerun()

    st.markdown("---")
    budget = st.number_input("Monthly budget (Rs)", min_value=10000, max_value=5000000,
                             value=400000, step=10000)
    with st.expander("⚙️ Data options"):
        if st.button("↩️ Undo my last entry"):
            st.session_state["flash"] = "↩️ Last entry removed" if undo_last() else "Nothing to undo"
            st.rerun()
        if st.button("🗑️ Reset to sample data"):
            if os.path.exists(DB):
                os.remove(DB)
            st.cache_resource.clear()
            st.rerun()

# ============================================================ numbers for the dashboard
mdf = df[df["ym"] == CUR]
mtd = mdf["amount"].sum()
dim = calendar.monthrange(TODAY.year, TODAY.month)[1]
days_left = dim - TODAY.day
pace = mtd / TODAY.day * dim                                          # agar yehi raftaar rahi
prev_same = df[(df["ym"] == CUR - 1) & (df["date"].dt.day <= TODAY.day)]["amount"].sum()
today_spend = df[df["date"].dt.date == TODAY]["amount"].sum()
forecast = fc["next_total"]
remaining = budget - mtd
safe_daily = max(remaining, 0) / days_left if days_left > 0 else max(remaining, 0)

st.title("💸 Spend-Wise AI")
st.caption(f"{calendar.month_name[TODAY.month]} {TODAY.year}  •  day {TODAY.day} of {dim}  •  "
           f"{len(tx):,} transactions in history")

k1, k2, k3, k4, k5 = st.columns(5)
delta = f"{(mtd - prev_same) / prev_same:+.0%} vs same days last month" if prev_same > 0 else None
k1.metric("Spent this month", rs(mtd), delta, delta_color="inverse")
k2.metric("Budget left", rs(max(remaining, 0)))
k3.metric("Safe daily limit", rs(safe_daily))
k4.metric("Current pace → month end", rs(pace))
k5.metric("AI forecast (month)", rs(forecast))

ratio = mtd / budget
st.progress(min(ratio, 1.0), text=f"{ratio:.0%} of monthly budget used ({rs(mtd)} of {rs(budget)})")
if mtd > budget:
    st.error(f"🚨 You are already {rs(mtd - budget)} over this month's budget.")
elif max(pace, forecast) > budget:
    st.warning(f"⚠️ Month-end looks like {rs(max(pace, forecast))} (pace {rs(pace)}, AI forecast "
               f"{rs(forecast)}) - that is over your budget. Try to keep to {rs(safe_daily)} per day.")
else:
    st.success(f"✅ On track. Pace {rs(pace)} and AI forecast {rs(forecast)} are both within your budget.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "🧾 Transactions", "📈 Forecast", "🏷️ AI Categorizer", "👥 Spending Habits"])

# ============================================================ TAB 1 : DASHBOARD
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly spending** (orange = AI's estimate for the rest of this month)")
        mon = df.groupby("ym")["amount"].sum()
        mon = mon.reindex(pd.period_range(mon.index.min(), CUR, freq="M"), fill_value=0)
        chart = pd.DataFrame({"Spent": mon.values, "Expected rest of month": 0.0},
                             index=mon.index.to_timestamp())
        chart.iloc[-1, 1] = max(forecast - mon.iloc[-1], 0.0)
        st.bar_chart(chart, color=["#4C78A8", "#F58518"])
    with c2:
        st.markdown("**Last 30 days**")
        start = pd.Timestamp(TODAY) - pd.Timedelta(days=29)
        daily = (df[df["date"] >= start].groupby("date")["amount"].sum()
                 .reindex(pd.date_range(start, periods=30), fill_value=0))
        st.bar_chart(daily.rename("Spent"), color="#54A24B")

    c3, c4 = st.columns([3, 2])
    with c3:
        st.markdown("**This month by category vs AI expectation**")
        so_far = mdf.groupby("category")["amount"].sum().reindex(ml.CATEGORIES, fill_value=0)
        expected = fc["cat_forecast"].reindex(ml.CATEGORIES)
        cat_tbl = pd.DataFrame({
            "Category": ml.CATEGORIES,
            "So far (Rs)": so_far.values.round(0),
            "AI expected (Rs)": expected.values.round(0),
            "Used": (so_far.values / np.maximum(expected.values, 1) * 100).round(0)})
        st.dataframe(cat_tbl, hide_index=True, column_config={
            "Used": st.column_config.ProgressColumn("Used", min_value=0, max_value=150, format="%d%%")})
    with c4:
        st.markdown("**💡 Smart insights**")
        if mtd > 0:
            top = so_far.idxmax()
            st.markdown(f"- Biggest spend this month: **{top}** ({so_far.max() / mtd:.0%} of total)")
        else:
            st.markdown("- No spending recorded yet this month.")
        recent = df[df["date"] >= pd.Timestamp(TODAY) - pd.Timedelta(days=90)]
        if not recent.empty:
            wd = recent.groupby("date")["amount"].sum().groupby(lambda x: x.dayofweek).mean()
            st.markdown(f"- You spend the most on **{DAYS[int(wd.idxmax())]}s** "
                        f"(avg {rs(wd.max())}/day, last 90 days)")
        st.markdown(f"- Today so far: **{rs(today_spend)}**")
        if days_left > 0:
            st.markdown(f"- To stay within budget, spend at most **{rs(safe_daily)}/day** "
                        f"for the next {days_left} days")

    st.markdown("**Latest transactions**")
    latest = tx.sort_values(["date", "id"], ascending=False).head(8).copy()
    latest["Source"] = latest["source"].map({"you": "Added by you", "sample": "Sample data"})
    st.dataframe(latest[["date", "category", "amount", "payment_method", "Source"]], hide_index=True,
                 column_config={"date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                                "amount": st.column_config.NumberColumn("Amount (Rs)", format="%d")})

# ============================================================ TAB 2 : TRANSACTIONS
with tab2:
    st.subheader("Transaction history")
    mine = tx[(tx["source"] == "you") & tx["ai_guess"].notna()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total transactions", f"{len(tx):,}")
    m2.metric("Added by you", f"{(tx['source'] == 'you').sum():,}")
    m3.metric("AI accuracy on your own category picks",
              f"{(mine['ai_guess'] == mine['category']).mean():.0%} ({int((mine['ai_guess'] == mine['category']).sum())}/{len(mine)})"
              if len(mine) else "–")

    f1, f2 = st.columns([3, 1])
    cats = f1.multiselect("Filter by category", ml.CATEGORIES)
    src = f2.selectbox("Source", ["All", "Added by you", "Sample data"])
    view = tx.copy()
    if cats:
        view = view[view["category"].isin(cats)]
    if src != "All":
        view = view[view["source"] == ("you" if src == "Added by you" else "sample")]
    view = view.sort_values(["date", "id"], ascending=False)
    view["Source"] = view["source"].map({"you": "Added by you", "sample": "Sample data"})
    view["AI check"] = view.apply(
        lambda r: "" if pd.isna(r["ai_guess"])
        else ("✅ correct" if r["ai_guess"] == r["category"] else f"❌ AI said {r['ai_guess']}"), axis=1)
    st.dataframe(view[["date", "category", "amount", "hour", "payment_method", "Source", "AI check"]],
                 hide_index=True,
                 column_config={"date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                                "amount": st.column_config.NumberColumn("Amount (Rs)", format="%d")})
    st.download_button("⬇️ Download history (CSV)", tx.to_csv(index=False), "transactions.csv", "text/csv")
    st.caption("Every transaction you add is saved here and the AI models retrain on it automatically.")

# ============================================================ TAB 3 : FORECAST
with tab3:
    st.subheader("Spending forecast - Linear Regression")
    mods = fc["models"]
    comp = pd.DataFrame({
        n: {"Features used": ", ".join(v["cols"]),
            "R² (all months)": round(v["r2_all"], 2),
            "Avg error on last 5 months": f"{v['mape']:.1%}",
            "RMSE (Rs)": f"{v['rmse']:,.0f}"}
        for n, v in mods.items()}).T
    st.dataframe(comp)
    st.caption("Simple LR uses only the month number (trend). Multiple LR also knows whether the month is "
               "festive (March / December), so it explains more of the variation. "
               "The forecast learns from completed months only.")

    left, right = st.columns([1, 2])
    with left:
        st.metric(f"Forecast for {calendar.month_name[fc['next_month']]}", rs(forecast))
        st.markdown("**Expected by category**")
        st.dataframe(fc["cat_forecast"].round(0).rename("Rs").to_frame())
    with right:
        first = df["date"].min().to_period("M")
        mt = fc["monthly"]
        labels = [(first + int(i)).strftime("%b %y") for i in mt["month_idx"]]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(mt["month_idx"], mt["total"] / 1000, "o-", label="Actual", color="#4C78A8")
        ax.plot(mt["month_idx"], fc["fitted"]["Simple LR"] / 1000, "--", label="Simple LR", color="#999999")
        ax.plot(mt["month_idx"], fc["fitted"]["Multiple LR"] / 1000, "-", label="Multiple LR", color="#54A24B")
        ax.scatter([fc["next_idx"]], [forecast / 1000], s=130, color="#E45756", zorder=5, label="Forecast")
        ax.axhline(budget / 1000, color="#E45756", ls=":", label="Budget")
        ax.set_xticks(list(mt["month_idx"]) + [fc["next_idx"]])
        ax.set_xticklabels(labels + [calendar.month_abbr[fc["next_month"]] + " (fc)"], rotation=60, fontsize=8)
        ax.set_ylabel("Rs thousand")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================ TAB 4 : AI CATEGORIZER
with tab4:
    st.subheader("AI Categorizer - Decision Tree vs Random Forest")
    st.caption("Given amount, time, day and payment method, the model predicts the category. "
               "The sidebar uses the Random Forest live while you type.")
    acc = pd.DataFrame({n: {"Train accuracy": r["train_acc"], "Test accuracy": r["test_acc"]}
                        for n, r in clf.items()}).T
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(acc.style.format("{:.1%}"))
        st.caption("The unpruned Decision Tree memorizes the training data (overfitting). "
                   "Pruning and Random Forest generalize better.")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(acc))
        ax.bar(x - 0.2, acc["Train accuracy"], 0.4, label="Train", color="#BBBBBB")
        ax.bar(x + 0.2, acc["Test accuracy"], 0.4, label="Test", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in acc.index], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    pick = st.selectbox("Inspect model", list(clf.keys()), index=2)
    r = clf[pick]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=prep["classes"], yticklabels=prep["classes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**What the model looks at (feature importance)**")
        st.bar_chart(r["importances"].rename("importance"), color="#F58518")
    st.markdown("**Precision / Recall / F1 per category**")
    st.dataframe(r["report"].round(2))

# ============================================================ TAB 5 : SPENDING HABITS
with tab5:
    st.subheader("Spending Habits - K-Means clustering of your days")
    st.caption("Each day is described by total spent, number of transactions and average transaction size.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Elbow method**")
        st.line_chart(pd.DataFrame({"Inertia": km["inertia"]}, index=km["ks"]))
        st.metric("Silhouette score (K=3)", f"{km['k_silhouette']:.2f}")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        dd = km["daily"]
        for name, color in zip(km["names"], ["#4C78A8", "#F58518", "#E45756"]):
            s = dd[dd["cluster_name"] == name]
            ax.scatter(s["n_txn"], s["total"] / 1000, s=14, alpha=0.6, label=name, color=color)
        ax.set_xlabel("Transactions per day")
        ax.set_ylabel("Total spent (Rs thousand)")
        ax.set_title("Daily spending clusters")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("**Cluster summary**")
    st.dataframe(km["summary"])
    st.info("Most days are low-spend, but a minority of busy and big-purchase days account for most "
            "of the money. Watching those days is the fastest way to save.")