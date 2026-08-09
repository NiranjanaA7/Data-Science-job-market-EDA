# Glassdoor Data Science Jobs — EDA & Dashboard

An end-to-end analytics project that takes a raw Glassdoor "Data Science Jobs" dataset through cleaning, feature engineering, and exploratory analysis, then surfaces the findings in an interactive Streamlit dashboard.

**Pipeline:** `Data Cleaning → Feature Engineering → EDA & Chart Curation → Streamlit Dashboard`

## What's in this repo

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Cleans the raw scrape — drops duplicates, parses the salary text into numeric columns, removes unlisted-salary rows. Outputs `cleaned_jobs.csv`. |
| `02_feature_engineering.ipynb` | Derives analysis-ready columns from the cleaned data — job role, seniority, state, skill/tool flags, remote flag, years of experience. Outputs `featured_jobs.csv`. |
| `03_eda_visualization_refined.ipynb` | Explores the engineered data and curates the chart set that made it into the dashboard, with reasoning for what was kept and cut. |
| `app.py` | The Streamlit dashboard — filters, KPI cards, and the 7 approved charts organized into tabs. |

## Key findings

- Salaries cluster in the mid-range (roughly $80K–$140K) with a long right tail
- Data Scientist and ML Engineer titles command the highest average pay; Directors have the highest median despite fewer postings
- Most postings target 3–5 years of experience
- Python and SQL dominate required skills, by a wide margin
- Hiring is concentrated in a handful of states (CA, MA, NY lead)
- The large majority of postings are in-person; remote roles are a small share

## Tech stack

Python · pandas · Matplotlib · Seaborn · Streamlit

## Running the dashboard

```bash
# 1. Install dependencies
pip install pandas matplotlib seaborn streamlit

# 2. Run the notebooks in order to generate the data
#    01_data_cleaning.ipynb  ->  cleaned_jobs.csv
#    02_feature_engineering.ipynb  ->  featured_jobs.csv

# 3. Update DATA_PATH in app.py to point to your featured_jobs.csv,
#    then launch the app
streamlit run app.py
```

> **Before you run it:** `app.py` currently points to `featured_jobs1.csv`, but the feature engineering notebook saves `featured_jobs.csv` — update `DATA_PATH` in `app.py` (or rename the file) so the filenames match, or the app will raise `FileNotFoundError` on load.

## Dataset

This is the widely-used public "Glassdoor Data Science Jobs" dataset (956 scraped postings, 15 raw columns), commonly used for learning/portfolio projects. It is not original data collection — the value of this project is in the cleaning, feature engineering, and chart-curation decisions built on top of it.

## Known limitations

- File paths are currently hardcoded/absolute in places — should be relative paths from the project root
- No predictive model yet (this is descriptive analytics); a simple regression predicting `Avg_Salary` from role, seniority, state, and skills would be a natural next step
- Rows with unlisted salary were dropped rather than imputed, which reduces sample size (467 of 956 rows remain after cleaning)

## Possible next steps

- Deploy to Streamlit Community Cloud and link the live demo here
- Add a `requirements.txt`
- Add a salary-prediction model
- Fix the `featured_jobs` / `featured_jobs1` filename mismatch and switch to relative paths
