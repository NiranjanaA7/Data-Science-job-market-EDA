"""
Glassdoor Data Science Jobs Dashboard
--------------------------------------
Streamlit dashboard built on top of the engineered dataset produced by:
  01_data_cleaning.ipynb -> 02_feature_engineering.ipynb -> featured_jobs.csv

Chart set matches 03_eda_visualization_refined.ipynb (7 dashboard charts +
Key Metrics). Company Age, Seniority Distribution, Company Ratings histogram,
and the pay-by-role/state tables were dropped from the notebook as low-value
for a public dashboard, so they're left out here too.

Run with:
    streamlit run app.py
"""

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
# US State Abbreviation → Full Name Mapping
# ----------------------------------------------------------------------
US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
DATA_PATH = "featured_jobs1.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Make sure you've run notebooks 1 and 2 "
        "first, and that this app is launched from the project root "
        "(the folder that contains `data/` and `app.py`)."
    )
    st.stop()

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
# Tabs — one per notebook section, 7 charts total
# ----------------------------------------------------------------------
tab_salary, tab_roles, tab_skills, tab_geo, tab_data = st.tabs(
    ["💰 Salary", "🧑‍💼 Roles & Experience", "🛠️ Skills", "🗺️ Geography", "📄 Raw Data"]
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
    top_states = filtered["Job_State"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(
        data=filtered, y="Job_State",
        order=top_states.index,
        color=PURPLE["mauve"], ax=ax
    )
    ax.set_title("Top Hiring States")
    st.pyplot(fig)

    # State abbreviation -> full name reference table for the chart above
    state_table = pd.DataFrame({
        "State (Abbr.)": top_states.index,
        "Full State Name": [US_STATE_NAMES.get(s, "Unknown") for s in top_states.index],
        "Job Postings": top_states.values,
    })
    st.markdown("**State Name Reference**")
    st.dataframe(state_table, use_container_width=True, hide_index=True)

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
st.caption("Built with Streamlit")
