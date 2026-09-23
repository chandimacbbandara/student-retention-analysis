# Student Retention Analysis & Prediction (Statistical Modeling)

This project focuses on analyzing student data to predict academic retention, specifically classifying whether a student will ultimately **Graduate**, remain **Enrolled**, or **Dropout**. Following specific assignment rubrics, this project exclusively utilizes **Statistical Modeling** techniques rather than tree-based or ensemble machine learning algorithms.

## 📊 The Dataset

The dataset was gathered by the Research Center for Endogenous Resource Valorization, Polytechnic Institute of Portalegre. It consists of multiple features characterizing each student's academic path, socioeconomic background, and macroeconomic indicators at the time of enrollment.

### Key Feature Categories:
- **Demographics & Family Background**: Age at enrollment, gender, nationality, displaced status, marital status, parental qualifications, and occupations.
- **Academic Path**: Application mode, application order, course selected, daytime/evening attendance, and previous qualifications.
- **Academic Performance**: Number of curricular units credited, enrolled, evaluated, approved, and their respective grades for both the 1st and 2nd semesters.
- **Financial Situation**: Tuition fees up to date, debtor status, and scholarship holder.
- **Macroeconomic Factors**: Unemployment rate, inflation rate, and GDP.

## 🔍 Exploratory Data Analysis (EDA) & Data Preprocessing

Comprehensive Exploratory Data Analysis (EDA) and preprocessing were conducted:
- **Statistical Tests**: ANOVA, Chi-Square contingency, Fisher's Exact test, and Tukey HSD were used to identify the most statistically significant predictors of student success and dropout rates.
- **Multicollinearity Removal**: Features with absolute correlation `|r| > 0.85` were dropped to satisfy statistical model assumptions.
- **Imbalance Fix**: The dataset struggles heavily with predicting "Enrolled" students. To mitigate this, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied exclusively to the training fold to avoid data leakage.

## 📈 Statistical Modeling Pipeline

As per the requirements, complex machine learning algorithms (like Random Forest and XGBoost) were completely removed in favor of the **regularized linear / Generalized Linear Model (GLM) family**.

### Algorithms Evaluated
1. **Logistic Regression (baseline)**: Multinomial GLM without penalty.
2. **LASSO**: GLM with L1 penalty (automatic feature selection).
3. **Elastic Net**: GLM with both L1 and L2 penalties.
4. **Ridge Classifier**: GLM with pure L2 penalty (reference baseline).

### Hyperparameter Fine-Tuning
The statistical models were fine-tuned to extract maximum performance:
- **RandomizedSearchCV**: Explored broad search spaces for both LASSO (regularization strength) and Elastic Net (l1_ratio).
- **GridSearchCV**: Used to exhaustively refine the LASSO parameters.
- **Optuna**: Applied to Elastic Net to efficiently navigate the continuous hyperparameter space via Bayesian Optimization.

### Final Selection & Conclusions
The **regularized logistic-regression family (LASSO and Elastic Net)** achieves highly competitive ROC-AUC scores (~0.83) on this dataset *without* needing complex tree-based models, offering a much more interpretable coefficient set.

**LASSO (Logistic Regression with L1 Penalty)** was selected as the final model because it offers the best trade-off between predictive performance and **interpretability**. By shrinking less important feature coefficients to zero, it isolates the true drivers of student retention.

---

## 🚀 Running the Interactive UI

An interactive Gradio Web UI has been developed for real-time predictions. The UI focuses on the most predictive features, automatically handling imputations for the rest.

### Local Setup
Ensure you have Python 3 installed, then run the following:

```bash
# 1. Activate your virtual environment (if applicable)
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
python3 app.py
```

### Hugging Face Deployment
This repository is configured with **Git LFS** for the `.joblib` model weights and a complete `requirements.txt`. To deploy to Hugging Face Spaces:
1. Create a new **Gradio** Space on Hugging Face.
2. Add the Space as a remote and push the `main` branch. The Hugging Face platform will automatically install dependencies and launch the app.
