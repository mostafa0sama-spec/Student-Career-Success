# Student Career Success — ML, DL & Salary Prediction Project

This project has two stages:

1. **Placement classification** — predict `Placed` vs `Not Placed`.
2. **Starting salary regression** — if the placement prediction is `Placed`, estimate `Starting_Salary_USD`.

## Files
- `Student_Career_Success_ML_DL_Project.ipynb` — full notebook
- `student_career_success_dataset.csv` — dataset
- `streamlit_app.py` — two-stage deployment UI
- `requirements.txt` — dependencies

## Leakage-safe design
The deployment models use pre-placement student features. The classifier excludes `Company_Tier`, `Career_Field`, `Placement_Mode`, and `Starting_Salary_USD`. The salary regressor is trained only on actually placed students and also excludes `Company_Tier`, `Career_Field`, and `Placement_Mode` so it can run from the same form inputs.

## Salary metrics
Regression models are compared using MAE, RMSE, and R². The deployed salary model is selected by the lowest test MAE.

## Run
1. `pip install -r requirements.txt`
2. Open and run `Student_Career_Success_ML_DL_Project.ipynb` from top to bottom.
3. Launch: `streamlit run streamlit_app.py`

The notebook will create the placement-model artifacts plus:
- `best_starting_salary_pipeline.joblib`
- `salary_model_info.json`
- `salary_regression_results.csv`

Educational demonstration only; not for automated hiring, admissions, or compensation decisions.
