"""
Credit Card Fraud - Logistic Regression (Streamlit)
הרצה:  streamlit run fraud_app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             accuracy_score)
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "credit_card_fraud_10k.csv"
TARGET = "is_fraud"
NUM_COLS = ["amount", "transaction_hour", "device_trust_score",
            "velocity_last_24h", "cardholder_age"]
BIN_COLS = ["foreign_transaction", "location_mismatch"]
CAT_COLS = ["merchant_category"]
OUTLIER_COLS = ["amount", "velocity_last_24h", "device_trust_score", "cardholder_age"]

st.set_page_config(page_title="זיהוי הונאות אשראי", layout="wide")
st.markdown("<style>.block-container{direction:rtl;text-align:right}</style>",
            unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def remove_outliers_iqr(df, cols, k):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[c].between(q1 - k * iqr, q3 + k * iqr)
    return df[mask], df[~mask]


def build_pipeline(C, fraud_weight):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_COLS),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN_COLS),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    model = LogisticRegression(C=C, class_weight={0: 1, 1: fraud_weight}, max_iter=1000)
    return Pipeline([("pre", pre), ("model", model)])


@st.cache_resource
def train(test_size, iqr_k, keep_fraud, C, seed, fraud_weight, beta):
    df = load_data()
    X = df.drop(columns=["transaction_id", TARGET])
    y = df[TARGET]
    # חלוקה לפני הסרת ערכי קיצון - סט הבדיקה נשאר נקי ומייצג את המציאות
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    train_df = X_train.assign(**{TARGET: y_train})
    if keep_fraud:
        legit_clean, legit_out = remove_outliers_iqr(train_df[train_df[TARGET] == 0], OUTLIER_COLS, iqr_k)
        clean = pd.concat([legit_clean, train_df[train_df[TARGET] == 1]])
        removed = legit_out
    else:
        clean, removed = remove_outliers_iqr(train_df, OUTLIER_COLS, iqr_k)

    pipe = build_pipeline(C, fraud_weight)
    Xc, yc = clean.drop(columns=[TARGET]), clean[TARGET]
    # סף אופטימלי ל-F-beta נבחר ב-Cross Validation על סט האימון בלבד (בלי לגעת בסט הבדיקה)
    cv_proba = cross_val_predict(pipe, Xc, yc, method="predict_proba",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    p, r, t = precision_recall_curve(yc, cv_proba)
    # beta > 1 נותן ל-Recall משקל גבוה יותר מ-Precision (עדיפות לתפוס הונאות)
    fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12)
    best_th = float(t[np.argmax(fb[:-1])])
    pipe.fit(Xc, yc)
    proba = pipe.predict_proba(X_test)[:, 1]
    return pipe, X_test, y_test, proba, len(train_df), removed, best_th


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ הגדרות מודל")
test_size = st.sidebar.slider("גודל סט בדיקה", 0.1, 0.4, 0.2, 0.05)
iqr_k = st.sidebar.slider("מקדם IQR להסרת ערכי קיצון", 1.5, 5.0, 3.0, 0.5,
                          help="ערך קטן = מסיר יותר שורות")
keep_fraud = st.sidebar.checkbox("לא להסיר עסקאות הונאה כערכי קיצון", value=True,
                                 help="יש רק כ-150 הונאות בנתונים; עסקאות חריגות הן לעיתים בדיוק ההונאות")
C = st.sidebar.select_slider("רגולריזציה C", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
fraud_weight = st.sidebar.select_slider("משקל לעסקאות הונאה (חוסר איזון)", [1, 5, 10, 20, 33, 66], value=10,
                                        help="66 ≈ balanced מלא; משקל נמוך יותר = פחות התראות שווא")
auto_th = st.sidebar.checkbox("סף אוטומטי (נבחר ב-CV על סט האימון)", value=True)
beta = st.sidebar.select_slider("עדיפות לתפיסת הונאות (beta)", [1, 2, 3, 5], value=3,
                                help="1 = איזון בין Precision ל-Recall (F1); ערך גבוה = תופס יותר הונאות על חשבון יותר התראות שווא")
seed = 42

pipe, X_test, y_test, proba, n_train, removed, best_th = train(test_size, iqr_k, keep_fraud, C, seed, fraud_weight, beta)
if auto_th:
    threshold = round(best_th, 3)
    st.sidebar.info(f"סף נבחר: {threshold}")
else:
    threshold = st.sidebar.slider("סף החלטה (Threshold)", 0.05, 0.95, 0.5, 0.05)
pred = (proba >= threshold).astype(int)

st.title("💳 זיהוי הונאות בכרטיסי אשראי - רגרסיה לוגיסטית")
tab_data, tab_dash, tab_pred = st.tabs(["📊 נתונים ועיבוד", "📈 דשבורד ביצועים", "🔮 חיזוי"])

# ---------------- Tab 1: data & preprocessing ----------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("שורות", f"{len(df):,}")
    c2.metric("עמודות", df.shape[1])
    c3.metric("שיעור הונאות", f"{df[TARGET].mean():.2%}")
    c4.metric("ערכים חסרים", int(df.isna().sum().sum()))
    st.dataframe(df.head(20), width="stretch")

    st.subheader("1. ערכים חסרים")
    st.dataframe(df.isna().sum().rename("חסרים").to_frame().T, width="stretch")
    st.caption("אין ערכים חסרים בקובץ. למרות זאת ה-Pipeline כולל SimpleImputer "
               "(חציון למשתנים מספריים, ערך שכיח לקטגוריאליים) כך שהמודל יתמודד עם ערכים חסרים בעתיד.")

    st.subheader("2. ערכי קיצון (IQR)")
    st.write(f"מתוך {n_train:,} שורות אימון הוסרו **{len(removed):,}** שורות "
             f"(מתוכן {int(removed[TARGET].sum())} הונאות). סט הבדיקה לא שונה.")
    col = st.selectbox("משתנה להצגה", OUTLIER_COLS)
    st.plotly_chart(px.box(df, x=TARGET, y=col, color=TARGET,
                           labels={TARGET: "הונאה"}), width="stretch")

    st.subheader("3. נרמול וקידוד")
    st.markdown("- משתנים מספריים: **StandardScaler** (ממוצע 0, סטיית תקן 1)\n"
                "- משתנים בינאריים: נשארים 0/1\n"
                "- `merchant_category`: **One-Hot Encoding**\n"
                "- `transaction_id` הוסר (מזהה בלבד)\n"
                "- חוסר איזון: משקל מוגבר להונאות (class_weight) + סף החלטה שממקסם F-beta (עדיפות ל-Recall) שנבחר ב-Cross Validation")

# ---------------- Tab 2: dashboard ----------------
with tab_dash:
    auc = roc_auc_score(y_test, proba)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC AUC", f"{auc:.3f}")
    m2.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    m3.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    m4.metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
    m5.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    st.caption(f"סט בדיקה: {len(y_test):,} עסקאות, {int(y_test.sum())} הונאות. סף החלטה: {threshold}")

    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", fill="tozeroy"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="אקראי"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width="stretch")
    with right:
        cm = confusion_matrix(y_test, pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  x=["חזוי: תקין", "חזוי: הונאה"], y=["בפועל: תקין", "בפועל: הונאה"],
                                  title="Confusion Matrix"), width="stretch")

    # השפעת הסף על המדדים
    ths = np.arange(0.05, 0.96, 0.05)
    rows = [{"threshold": t,
             "Recall": recall_score(y_test, proba >= t, zero_division=0),
             "Precision": precision_score(y_test, proba >= t, zero_division=0),
             "F1": f1_score(y_test, proba >= t, zero_division=0)} for t in ths]
    st.plotly_chart(px.line(pd.DataFrame(rows), x="threshold", y=["Recall", "Precision", "F1"],
                            title="מדדים לפי סף החלטה", markers=True), width="stretch")

    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": [n.split("__")[1] for n in names],
                          "coef": pipe.named_steps["model"].coef_[0]}).sort_values("coef")
    st.plotly_chart(px.bar(coefs, x="coef", y="feature", orientation="h",
                           title="מקדמי המודל (חיובי = מעלה סיכוי להונאה)"), width="stretch")

# ---------------- Tab 3: prediction ----------------
with tab_pred:
    st.subheader("בחר ערכים לעסקה")
    a, b, c = st.columns(3)
    with a:
        amount = st.number_input("סכום (amount)", 0.0, float(df.amount.max() * 2), 150.0, 10.0)
        hour = st.slider("שעת עסקה", 0, 23, 12)
        category = st.selectbox("קטגוריית בית עסק", sorted(df.merchant_category.unique()))
    with b:
        foreign = st.radio("עסקה בחו\"ל", [0, 1], horizontal=True, format_func=lambda v: "כן" if v else "לא")
        mismatch = st.radio("אי-התאמת מיקום", [0, 1], horizontal=True, format_func=lambda v: "כן" if v else "לא")
        trust = st.slider("ציון אמינות מכשיר", 0, 100, 60)
    with c:
        velocity = st.slider("מספר עסקאות ב-24 שעות", 0, 15, 2)
        age = st.slider("גיל בעל הכרטיס", 18, 90, 40)

    sample = pd.DataFrame([{"amount": amount, "transaction_hour": hour,
                            "merchant_category": category, "foreign_transaction": foreign,
                            "location_mismatch": mismatch, "device_trust_score": trust,
                            "velocity_last_24h": velocity, "cardholder_age": age}])
    p = pipe.predict_proba(sample)[0, 1]
    is_fraud = p >= threshold

    r1, r2 = st.columns([1, 2])
    r1.metric("הסתברות להונאה", f"{p:.1%}")
    if is_fraud:
        r2.error(f"🚨 העסקה סווגה כ**הונאה** (סף {threshold})")
    else:
        r2.success(f"✅ העסקה סווגה כ**תקינה** (סף {threshold})")
    st.progress(float(p))
