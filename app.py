import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
/* Main page headings and text */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] {
    color: #172033 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #172033 !important;
}

/* Main page background */
[data-testid="stAppViewContainer"] {
    background-color: #f7f9fc;
}
</style>
""", unsafe_allow_html=True)




# -------------------- Styling --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f7f9ff 0%, #eef4ff 55%, #f8fbff 100%);
        color: #14213d;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10254b 0%, #173b70 100%);
    }
    [data-testid="stSidebar"] * { color: #f5f8ff !important; }
    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background: linear-gradient(120deg, #182b68 0%, #3156bd 58%, #1697c9 100%);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(34, 70, 150, .18);
    }
    .hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.2rem; }
    .hero p { color: #e7efff; margin: 0; font-size: 1.05rem; }
    .section-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #dce7fb;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 5px 18px rgba(35, 68, 130, .06);
        margin-bottom: 16px;
    }
    .small-muted { color: #60718f; font-size: .92rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce7fb;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 14px rgba(35, 68, 130, .05);
    }
    .result-placed {
        background: #e9fbf2; border: 1px solid #a9e7c8; color: #087443;
        padding: 18px 20px; border-radius: 16px; font-size: 1.2rem; font-weight: 700;
    }
    .result-notplaced {
        background: #fff2ed; border: 1px solid #ffc7b5; color: #a43d20;
        padding: 18px 20px; border-radius: 16px; font-size: 1.2rem; font-weight: 700;
    }
    .stButton > button {
        border: 0; border-radius: 11px; padding: .72rem 1rem;
        background: linear-gradient(90deg, #245bd6, #168bc7);
        color: white; font-weight: 700; width: 100%;
    }
    .stButton > button:hover { color: white; border: 0; filter: brightness(1.06); }
    h2, h3 { color: #173565; }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #F3F6FB;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #1F2937 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #CBD5E1;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Placement Predictor")
    st.caption("Your personal placement assistant")
    st.markdown("---")

    st.markdown("### 🏠 Student Portal")
    st.markdown(
        "Enter your academic details to check your estimated "
        "placement chances."
    )

    st.markdown("---")
    st.caption("Predictions are estimates, not guarantees.")

# -------------------- Hero --------------------
st.markdown("""
<div class="hero">
  <h1>🎓 Student Placement Prediction</h1>
  <p>Explore student placement outcomes with machine learning. Enter student details,
  generate a prediction, and compare classification models.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Dataset loading and validation --------------------
required_columns = [
    "CGPA", "Internships", "Projects", "Workshops/Certifications",
    "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities",
    "PlacementTraining", "SSC_Marks", "HSC_Marks", "PlacementStatus"
]
categorical_columns = ["ExtracurricularActivities", "PlacementTraining"]
numeric_columns = [
    "CGPA", "Internships", "Projects", "Workshops/Certifications",
    "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"
]


# -------------------- Load saved dataset --------------------
from pathlib import Path

dataset_path = Path(__file__).parent / "campus_placement.csv"

try:
    df = pd.read_csv(dataset_path)
except Exception as exc:
    st.error(f"Could not read campus_placement.csv: {exc}")
    st.stop()

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(
        "The CSV is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df = df[required_columns].copy()

# Convert numeric columns safely
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

if len(df) < 10:
    st.error("The dataset needs at least 10 complete records.")
    st.stop()

if df["PlacementStatus"].nunique() < 2:
    st.error("PlacementStatus must contain at least two classes.")
    st.stop()

class_counts = df["PlacementStatus"].value_counts()
if class_counts.min() < 2:
    st.error(
        "Each placement-status class needs at least two records "
        "for a stratified train/test split."
    )
    st.stop()

X_raw = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"].astype(str)

X = pd.get_dummies(
    X_raw,
    columns=categorical_columns,
    dtype=int
)


# -------------------- Train and evaluate models --------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as exc:
    st.error(f"Could not create a train/test split: {exc}")
    st.stop()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
predictions = {}
accuracies = {}

for model_name, classifier in models.items():
    try:
        classifier.fit(X_train, y_train)
        predictions[model_name] = classifier.predict(X_test)
        accuracies[model_name] = accuracy_score(y_test, predictions[model_name])
    except Exception as exc:
        st.error(f"Could not train {model_name}: {exc}")
        st.stop()

# Fit the final Random Forest on all available records for interactive predictions.
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X, y)

# -------------------- Summary metrics --------------------
best_model_name = max(accuracies, key=accuracies.get)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Student records", f"{len(df):,}")
m2.metric("Input features", "10")
m3.metric("Models compared", "3")
m4.metric("Best test accuracy", f"{accuracies[best_model_name] * 100:.1f}%")

st.markdown("## 🧑‍🎓 Predict placement")
st.markdown('<p class="small-muted">Enter the student profile below. The app uses the trained Random Forest model for the prediction.</p>', unsafe_allow_html=True)



with st.container(border=True):
    st.markdown("### 🎓 Enter Your Credentials")
    st.caption(
        "Enter your academic details to check your estimated "
        "placement chances."
    )

    with st.form("placement_form"):
        c1, c2 = st.columns(2)

        with c1:
            cgpa = st.number_input(
                "CGPA", min_value=0.0, max_value=10.0,
                value=0.0, step=0.1
            )
            internships = st.number_input(
                "Internships", min_value=0, max_value=20,
                value=0, step=1
            )
            projects = st.number_input(
                "Projects", min_value=0, max_value=30,
                value=0, step=1
            )
            workshops = st.number_input(
                "Workshops / Certifications", min_value=0,
                max_value=30, value=0, step=1
            )
            aptitude = st.number_input(
                "Aptitude Test Score", min_value=0,
                max_value=100, value=0, step=1
            )

        with c2:
            soft_skills = st.number_input(
                "Soft Skills Rating", min_value=0.0,
                max_value=10.0, value=0.0, step=0.1
            )
            extracurricular = st.selectbox("Extracurricular Activities", sorted(df["ExtracurricularActivities"].astype(str).unique()))
            training = st.selectbox("Placement Training", sorted(df["PlacementTraining"].astype(str).unique()))
            ssc = st.number_input("SSC Marks", min_value=0, max_value=100, value=0, step=1)
            hsc = st.number_input("HSC Marks", min_value=0, max_value=100, value=0, step=1)
       
        submitted = st.form_submit_button("🎯 Check My Placement Chances", width="stretch")


if submitted:
    st.markdown("## 📊 Your Placement Assessment")
    st.caption("Based on the academic and preparation details you entered.")
    new_student = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Workshops/Certifications": workshops,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": soft_skills,
        "ExtracurricularActivities": extracurricular,
        "PlacementTraining": training,
        "SSC_Marks": ssc,
        "HSC_Marks": hsc
    }])
    new_student_encoded = pd.get_dummies(
        new_student, columns=categorical_columns, dtype=int
    ).reindex(columns=X.columns, fill_value=0)
    predicted_status = str(final_model.predict(new_student_encoded)[0])
    probabilities = final_model.predict_proba(new_student_encoded)[0]
    predicted_index = list(final_model.classes_).index(predicted_status)
    confidence = float(probabilities[predicted_index])

    if predicted_status.strip().lower() == "placed":
        st.markdown(
            f'<div class="result-placed">✅ Predicted status: {predicted_status}'
            f'<br><span style="font-size: .95rem; font-weight: 500;">Model confidence: {confidence:.1%}</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-notplaced">ℹ️ Predicted status: {predicted_status}'
            f'<br><span style="font-size: .95rem; font-weight: 500;">Model confidence: {confidence:.1%}</span></div>',
            unsafe_allow_html=True
        )
    st.caption("Confidence is the model's estimated probability for its predicted class, not a guarantee of an actual placement.")

# -------------------- Model comparison --------------------
st.markdown("## 📊 Model comparison")
st.markdown('<p class="small-muted">Accuracy is measured on the same held-out test split.</p>', unsafe_allow_html=True)
results = pd.DataFrame({
    "Model": list(accuracies.keys()),
    "Accuracy": list(accuracies.values())
}).sort_values("Accuracy", ascending=False)
chart_col, table_col = st.columns([1.5, 1])
with chart_col:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results, x="Model", y="Accuracy", hue="Model", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_title("Classification model accuracy")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with table_col:
    st.dataframe(
        results.assign(**{"Accuracy (%)": (results["Accuracy"] * 100).round(2)}),
        hide_index=True,
        use_container_width=True
    )
    st.info(f"Best measured result: **{best_model_name}**. Results may vary with different data splits.")

# -------------------- Confusion matrices --------------------
st.markdown("## 🧩 Confusion matrices")
matrix_columns = st.columns(3)
class_labels = sorted(y.unique())
for idx, (model_name, pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, pred, labels=class_labels)
    with matrix_columns[idx]:
        fig, ax = plt.subplots(figsize=(4, 3.3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels, ax=ax
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# -------------------- Feature importance --------------------
st.markdown("## 🔎 Feature importance")
st.markdown('<p class="small-muted">These scores describe how the Random Forest used features in this dataset; they do not prove cause and effect.</p>', unsafe_allow_html=True)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values("Importance", ascending=False)
top_importance = importance.head(10).sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(top_importance["Feature"], top_importance["Importance"])
ax.set_xlabel("Importance score")
ax.set_ylabel("")
ax.set_title("Top 10 features used by Random Forest")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("View dataset preview"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("About this project"):
    st.write(
        "This project demonstrates classification using Logistic Regression, Decision Tree, "
        "and Random Forest on a student placement dataset. Model performance is evaluated on "
        "a held-out test split. The predictions are educational estimates and should not be "
        "used as the sole basis for real placement decisions."
    )
    st.write("Technologies: Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit.")

st.markdown("---")
st.caption("Student Placement Predictor • Educational machine-learning project")
