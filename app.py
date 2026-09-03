"""
Glassdoor Data Science Jobs Dashboard
--------------------------------------
Streamlit dashboard built on top of the engineered dataset produced by:
  01_data_cleaning.ipynb -> 02_feature_engineering.ipynb -> featured_jobs.csv

Chart set matches 03_eda_visualization_refined.ipynb (7 dashboard charts +
Key Metrics). Company Age, Seniority Distribution, Company Ratings histogram,
and the pay-by-role/state tables were dropped from the notebook as low-value
for a public dashboard, so they're left out here too.

NEW: a "Predict Salary" tab, powered by the model trained in
04_model_training.py (salary_prediction_model.pkl + salary_prediction_meta.json).

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Data Science Jobs Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Color Palette (matches the refined EDA notebook)
# ----------------------------------------------------------------------
PURPLE = {
    "dark_purple": "#4B0082",
    "mauve": "#9370DB",
    "light_violet": "#C8A2C8",
}

sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlecolor"] = PURPLE["dark_purple"]

# ----------------------------------------------------------------------
# Data Loading — now relative to this file, not a hardcoded Windows path
# (this was Known Limitation #1 / interview Q14 in your own docs)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "featured_jobs.csv"
MODEL_PATH = BASE_DIR / "salary_prediction_model.pkl"
META_PATH = BASE_DIR / "salary_prediction_meta.json"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path: Path, meta_path: Path):
    """Returns (model, meta) or (None, None) if not trained yet."""
    if not model_path.exists() or not meta_path.exists():
        return None, None
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Make sure you've run notebooks 1 and 2 "
        "first, and that this app is launched from the project root "
        "(the folder that contains `data/` and `app.py`)."
    )
    st.stop()

model, model_meta = load_model(MODEL_PATH, META_PATH)

# ----------------------------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------------------------
st.sidebar.title("🔎 Filters")

job_roles = sorted(df["Job_Simplified"].dropna().unique())
selected_roles = st.sidebar.multiselect(
    "Job Role", options=job_roles, default=job_roles
)

states = sorted(df["Job_State"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "State", options=states, default=states
)

remote_option = st.sidebar.radio(
    "Work Type", options=["All", "Remote Only", "In-Person Only"], index=0
)

min_salary, max_salary = int(df["Avg_Salary"].min()), int(df["Avg_Salary"].max())
salary_range = st.sidebar.slider(
    "Average Salary Range (K USD)",
    min_value=min_salary,
    max_value=max_salary,
    value=(min_salary, max_salary),
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Glassdoor Data Science Jobs (cleaned & engineered)")

# ----------------------------------------------------------------------
# Apply Filters
# ----------------------------------------------------------------------
filtered = df[
    df["Job_Simplified"].isin(selected_roles)
    & df["Job_State"].isin(selected_states)
    & df["Avg_Salary"].between(salary_range[0], salary_range[1])
].copy()

if remote_option == "Remote Only":
    filtered = filtered[filtered["Remote_Job"] == 1]
elif remote_option == "In-Person Only":
    filtered = filtered[filtered["Remote_Job"] == 0]

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Data Science Jobs Dashboard")
st.markdown(
    "An interactive look at salaries, skills, and hiring trends in "
    "data science job postings."
)

if filtered.empty:
    st.warning("No postings match the current filters. Try widening your selection.")
    st.stop()

# ----------------------------------------------------------------------
# Key Metrics (matches the notebook's Key Metrics cell)
# ----------------------------------------------------------------------
skill_cols_all = [c for c in filtered.columns if c.endswith("_yn")]
top_skill = (
    filtered[skill_cols_all].sum().sort_values(ascending=False).index[0].replace("_yn", "")
    if skill_cols_all
    else "N/A"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Job Postings", f"{len(filtered):,}")
k2.metric("Avg. Salary (K USD)", f"{filtered['Avg_Salary'].mean():.0f}")
k3.metric("Avg. Company Rating", f"{filtered['Rating'].mean():.1f} ⭐")
k4.metric("Remote Share", f"{filtered['Remote_Job'].mean() * 100:.0f}%")
k5.metric("Top Skill", top_skill)

st.markdown("---")

# ----------------------------------------------------------------------
# Tabs — 7 EDA charts + new Predict Salary tab
# ----------------------------------------------------------------------
tab_salary, tab_roles, tab_skills, tab_geo, tab_predict, tab_data = st.tabs(
    [
        "💰 Salary",
        "🧑‍💼 Roles & Experience",
        "🛠️ Skills",
        "🗺️ Geography",
        "🔮 Predict Salary",
        "📄 Raw Data",
    ]
)

# --- Salary Tab (Chart 1: Salary Distribution, Chart 2: Salary by Role) ---
with tab_salary:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        filtered["Avg_Salary"], bins=25, kde=True,
        color=PURPLE["light_violet"], ax=ax
    )
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Average Salary (K USD)")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = (
        filtered.groupby("Job_Simplified")["Avg_Salary"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=filtered, x="Job_Simplified", y="Avg_Salary",
        order=order, color=PURPLE["light_violet"], ax=ax
    )
    ax.set_title("Average Salary by Job Role")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --- Roles & Experience Tab (Chart 3, 6, 7) -------------------------------
with tab_roles:
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.countplot(
            data=filtered, y="Job_Simplified",
            order=filtered["Job_Simplified"].value_counts().index,
            color=PURPLE["light_violet"], ax=ax
        )
        ax.set_title("Job Role Distribution")
        st.pyplot(fig)

    with col2:
        exp_order = ["0-2 Years", "3-5 Years", "6-10 Years", "10+ Years", "Unknown"]
        exp_counts = (
            filtered["Experience_Group"]
            .value_counts()
            .reindex(exp_order)
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        bar = sns.barplot(
            x=exp_counts.index, y=exp_counts.values,
            color=PURPLE["light_violet"], ax=ax
        )
        for i, v in enumerate(exp_counts.values):
            bar.text(i, v, int(v), ha="center", va="bottom")
        ax.set_title("Experience Required")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.countplot(
            data=filtered, x="Remote_Job",
            color=PURPLE["light_violet"], ax=ax
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["In-Person", "Remote"])
        ax.set_title("Remote vs In-Person")
        st.pyplot(fig)

# --- Skills Tab (Chart 4: Top Skills Required) ----------------------------
with tab_skills:
    if skill_cols_all:
        skill_counts = (
            filtered[skill_cols_all].sum().sort_values(ascending=False).reset_index()
        )
        skill_counts.columns = ["Skill", "Count"]
        skill_counts["Skill"] = skill_counts["Skill"].str.replace("_yn", "")

        fig, ax = plt.subplots(figsize=(9, 7))
        bar = sns.barplot(
            data=skill_counts, x="Count", y="Skill",
            color=PURPLE["light_violet"], ax=ax
        )
        for i, v in enumerate(skill_counts["Count"]):
            bar.text(v + 1, i, str(v), va="center")
        ax.set_title("Most Requested Skills & Tools")
        st.pyplot(fig)

        st.dataframe(skill_counts, use_container_width=True)
    else:
        st.info("No skill/tool columns (`*_yn`) found in the dataset.")

# --- Geography Tab (Chart 5: Top Hiring States) ---------------------------
with tab_geo:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(
        data=filtered, y="Job_State",
        order=filtered["Job_State"].value_counts().head(15).index,
        color=PURPLE["mauve"], ax=ax
    )
    ax.set_title("Top Hiring States")
    st.pyplot(fig)

# --- Predict Salary Tab ----------------------------------------------------
with tab_predict:
    st.subheader("Estimate a Salary")

    if model is None:
        st.info(
            "No trained model found yet. Run `04_model_training.py` first — "
            "it will produce `salary_prediction_model.pkl` and "
            "`salary_prediction_meta.json` in this folder."
        )
    else:
        st.caption(
            f"Model: {model_meta['model_name']} · "
            f"Test MAE ≈ ${model_meta['test_mae']:.0f}K · "
            f"Test R² ≈ {model_meta['test_r2']:.2f}. "
            "Treat this as a ballpark estimate, not a precise figure — "
            "it's trained on 467 postings."
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            role_input = st.selectbox(
                "Job Role", model_meta["categorical_options"]["Job_Simplified"]
            )
            state_input = st.selectbox(
                "State", model_meta["categorical_options"]["Job_State"]
            )
            experience_input = st.selectbox(
                "Experience Required",
                model_meta["categorical_options"]["Experience_Group"],
            )

        with input_col2:
            rating_range = model_meta["numeric_ranges"].get("Rating", [1.0, 5.0])
            rating_input = st.slider(
                "Company Rating", min_value=float(rating_range[0]),
                max_value=float(rating_range[1]), value=3.5, step=0.1,
            )
            age_range = model_meta["numeric_ranges"].get("Company_Age", [0.0, 100.0])
            age_input = st.slider(
                "Company Age (years)", min_value=0, max_value=int(age_range[1]),
                value=15,
            )
            remote_input = st.checkbox("Remote position")

        st.markdown("**Skills / tools mentioned in the posting:**")
        skill_labels = [c.replace("_yn", "") for c in model_meta["skill_features"]]
        selected_skills = st.multiselect("Skills", skill_labels)

        if st.button("Predict Salary", type="primary"):
            row = {
                "Job_Simplified": role_input,
                "Job_State": state_input,
                "Experience_Group": experience_input,
                "Rating": rating_input,
                "Company_Age": age_input,
                "Remote_Job": int(remote_input),
            }
            for skill_col in model_meta["skill_features"]:
                skill_name = skill_col.replace("_yn", "")
                row[skill_col] = 1 if skill_name in selected_skills else 0

            input_df = pd.DataFrame([row])
            prediction = model.predict(input_df)[0]

            st.metric("Estimated Average Salary", f"${prediction:.0f}K")
            st.caption(
                f"±${model_meta['test_mae']:.0f}K typical error based on holdout "
                "test performance — not a guaranteed range."
            )

# --- Raw Data Tab -----------------------------------------------------------
with tab_data:
    st.subheader("Filtered Dataset")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="filtered_jobs.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built with Streamlit · Data cleaned & engineered via pandas notebooks")
