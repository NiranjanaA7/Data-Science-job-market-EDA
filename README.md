# Glassdoor Data Science Jobs — EDA & Dashboard

An end-to-end analytics project that takes a raw Glassdoor "Data Science Jobs" dataset through cleaning, feature engineering, and exploratory analysis, then surfaces the findings in an interactive Streamlit dashboard.

**Pipeline:** Data Cleaning → Feature Engineering → EDA & Chart Curation → Streamlit Dashboard

---

## 🚀 Live Demo

**[Open the dashboard →](https://data-science-job-market-eda.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-science-job-market-eda.streamlit.app/)

---
## What's in this repo

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Cleans the raw scrape — drops duplicates, parses the salary text into numeric columns, removes unlisted-salary rows. Outputs `cleaned_jobs.csv`. |
| `02_feature_engineering.ipynb` | Derives analysis-ready columns from the cleaned data — job role, seniority, state, skill/tool flags, remote flag, years of experience. Outputs `featured_jobs.csv`. |
| `03_eda_visualization_refined.ipynb` | Explores the engineered data and curates the chart set that made it into the dashboard, with reasoning for what was kept and cut. |
| `app.py` | The Streamlit dashboard — sidebar filters, KPI cards, and 7 charts organized into tabs. |
| `requirements.txt` | Python dependencies needed to run the dashboard. |

---

## Key findings

- Salaries cluster in the mid-range (roughly **$80K–$140K**) with a long right tail
- **Directors** have the highest median salary (**$150K+**) despite being the smallest job category by count — a non-obvious insight easy to miss if you only look at raw counts
- Most postings target **3–5 years** of experience (189 of 467)
- **Python** and **SQL** dominate required skills, far ahead of everything else
- Hiring is concentrated in a handful of states — **CA, MA, NY, VA** lead
- Only about **4%** of postings are remote — the large majority are in-person

---

## Tech stack

Python · pandas · Matplotlib · Seaborn · Streamlit · regex (for text extraction)

---

## Running the dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebooks in order to generate the data
#    01_data_cleaning.ipynb          -> cleaned_jobs.csv
#    02_feature_engineering.ipynb    -> featured_jobs.csv

# 3. Launch the app
streamlit run app.py
```

**`requirements.txt`:**
```
streamlit
pandas
matplotlib
seaborn
```

---

## Dashboard layout

- **Sidebar filters:** Job Role, State, Work Type (All / Remote / In-Person), Salary Range slider
- **KPI row:** Job Postings, Avg. Salary, Avg. Rating, Remote Share, Top Skill — all filter-reactive
- **Tabs:**

| Tab | Chart(s) |
|---|---|
| 💰 Salary | Salary Distribution (histogram + KDE); Average Salary by Job Role (boxplot) |
| 🧑‍💼 Roles & Experience | Job Role Distribution; Experience Required; Remote vs In-Person |
| 🛠️ Skills | Most Requested Skills & Tools (bar chart + sortable table) |
| 🗺️ Geography | Top Hiring States (top 15), with a state abbreviation → full name reference table |
| 📄 Raw Data | Filtered dataset table with CSV export |

---

## Dataset

This is the widely-used public "Glassdoor Data Science Jobs" dataset (956 scraped postings, 15 raw columns), commonly used for learning and portfolio projects. It is not original data collection — the value of this project is in the cleaning, feature engineering, and chart-curation decisions built on top of it.

---

## Known limitations

- File paths are currently hardcoded/absolute in places — should be relative paths from the project root, or config/environment-variable driven
- Rows with unlisted salary (`-1`) were dropped rather than imputed — reduces the sample from 956 to 467 rows, and may bias results if the missingness isn't random
- Title/seniority classification uses simple keyword-based `if/elif` priority matching, not a documented rule set or trained model
- Regex-based skill/tool detection is a first pass and hasn't been validated against a hand-labeled ground truth set
- No predictive model yet — this is descriptive analytics only

---

## Possible next steps

- Switch remaining hardcoded paths to relative, config-driven paths
- Add a simple salary-prediction model (regression on role, seniority, state, and skills)
- Validate the regex skill extraction against hand-labeled postings and report precision/recall
- Investigate whether dropped salary rows are missing at random
