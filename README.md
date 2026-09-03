# Glassdoor Data Science Jobs — EDA, Salary Model & Dashboard

An end-to-end analytics project that takes a raw Glassdoor "Data Science Jobs" dataset through cleaning, feature engineering, and exploratory analysis, trains a salary-prediction model on top of it, and surfaces everything in an interactive Streamlit dashboard.

**Pipeline:** Data Cleaning → Feature Engineering → EDA & Chart Curation → Salary Model → Streamlit Dashboard

---

## 🚀 Live Demo

**[Open the dashboard →](YOUR-APP-LINK-HERE)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-APP-LINK-HERE)

---

## What's in this repo

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Cleans the raw scrape — drops duplicates, parses the salary text into numeric columns, removes unlisted-salary rows. Outputs `cleaned_jobs.csv`. |
| `02_feature_engineering.ipynb` | Derives analysis-ready columns from the cleaned data — job role, seniority, state, skill/tool flags, remote flag, years of experience. Outputs `featured_jobs.csv`. |
| `03_eda_visualization.ipynb` | Explores the engineered data and curates the chart set that made it into the dashboard, with reasoning for what was kept and cut. |
| `04_model_training.py` | Trains and evaluates a salary-prediction model (Ridge, Random Forest, Gradient Boosting), tunes the best one, and exports `salary_prediction_model.pkl` + `salary_prediction_meta.json` for the dashboard. |
| `app.py` | The Streamlit dashboard — sidebar filters, KPI cards, 7 charts, and a "Predict Salary" tab powered by the trained model. |
| `Glassdoor_Project_Interview_Guide.docx` | Walkthrough of how each script/notebook works, why Gradient Boosting was chosen, and an interview question bank for talking through the project. |
| `requirements.txt` | Python dependencies needed to run the pipeline and dashboard. |

---

## Key findings

- Salaries cluster in the mid-range (roughly **$80K–$140K**) with a long right tail
- **Directors** have the highest median salary (**$150K+**) despite being the smallest job category by count — a non-obvious insight easy to miss if you only look at raw counts
- Most postings target **3–5 years** of experience (189 of 467)
- **Python** and **SQL** dominate required skills, far ahead of everything else
- Hiring is concentrated in a handful of states — **CA, MA, NY, VA** lead
- Only about **4%** of postings are remote — the large majority are in-person

### Salary model performance

- **Model:** Gradient Boosting (selected over Ridge and Random Forest on cross-validated MAE, then hyperparameter-tuned)
- **26 input features:** 3 categorical (Job Role, State, Experience Group), 3 numeric (Rating, Company Age, Remote), 20 binary skill/tool flags
- **Test MAE ≈ $22.7K · RMSE ≈ $30.7K · R² ≈ 0.43**
- **56.4%** of predictions land within ±$20K of the true salary; **74.5%** within ±$30K
- Trained on the same 467 cleaned/engineered postings used throughout the project — treat outputs as a ballpark estimate, not a precise figure

---

## Tech stack

Python · pandas · scikit-learn · joblib · Matplotlib · Seaborn · Streamlit · regex (for text extraction)

---

## Running the project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebooks in order to generate the data
#    01_data_cleaning.ipynb          -> cleaned_jobs.csv
#    02_feature_engineering.ipynb    -> featured_jobs.csv

# 3. (Optional but recommended) Train the salary model
#    Produces salary_prediction_model.pkl + salary_prediction_meta.json
#    Without this step, the dashboard still runs — the "Predict Salary"
#    tab just shows an info message instead of a live predictor.
python 04_model_training.py

# 4. Launch the dashboard
streamlit run app.py
```

**`requirements.txt`:**
```
streamlit
pandas
matplotlib
seaborn
scikit-learn
joblib
```

---

## Dashboard layout

- **Sidebar filters:** Job Role, State, Work Type (All / Remote / In-Person), Salary Range slider
- **KPI row:** Job Postings, Avg. Salary, Avg. Rating, Remote Share, Top Skill — all filter-reactive
- **Tabs:**

| Tab | Content |
|---|---|
| 💰 Salary | Salary Distribution (histogram + KDE); Average Salary by Job Role (boxplot) |
| 🧑‍💼 Roles & Experience | Job Role Distribution; Experience Required; Remote vs In-Person |
| 🛠️ Skills | Most Requested Skills & Tools (bar chart + sortable table) |
| 🗺️ Geography | Top Hiring States (top 15) |
| 🔮 Predict Salary | Enter role, state, experience, rating, company age, remote status, and skills to get a live salary estimate (± test MAE) from the trained model |
| 📄 Raw Data | Filtered dataset table with CSV export |

Data and model loading are cached with `@st.cache_data` / `@st.cache_resource` so filter interactions don't re-read the CSV or re-load the model from disk each time. If the model files aren't present, the Predict Salary tab degrades gracefully to an informational message instead of crashing.

---

## Dataset

This is the widely-used public "Glassdoor Data Science Jobs" dataset (956 scraped postings, 15 raw columns), commonly used for learning and portfolio projects. It is not original data collection — the value of this project is in the cleaning, feature engineering, chart-curation, and modeling decisions built on top of it.

---

## Known limitations

- Rows with unlisted salary (`-1`) were dropped rather than imputed — reduces the sample from 956 to 467 rows, and may bias results if the missingness isn't random
- Title/seniority classification uses simple keyword-based `if/elif` priority matching, not a documented rule set or trained model
- Regex-based skill/tool detection is a first pass and hasn't been validated against a hand-labeled ground truth set
- The salary model is trained on only 467 rows — R² ≈ 0.43 and a ±$22.7K typical error mean predictions are a rough guide, not a precise figure
- The model isn't retrained automatically when the underlying data changes — rerun `04_model_training.py` after any changes to the cleaning/feature-engineering steps

---

## Possible next steps

- Validate the regex skill extraction against hand-labeled postings and report precision/recall
- Investigate whether dropped salary rows are missing at random
- Try richer features (e.g. job description length/embeddings, interaction terms) to push model R² higher
- Add automated retraining / a data-versioning check so the dashboard warns if the model was trained on stale data
