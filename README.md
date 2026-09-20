# Student Retention Analysis & Prediction

This project focuses on analyzing student data to predict academic retention, specifically classifying whether a student will ultimately **Graduate**, remain **Enrolled**, or **Dropout**. By leveraging machine learning pipelines and comprehensive statistical inference, this system provides actionable insights into student success and risk factors.

## 📊 The Dataset

The dataset was gathered by the Research Center for Endogenous Resource Valorization, Polytechnic Institute of Portalegre. It consists of multiple features characterizing each student's academic path, socioeconomic background, and macroeconomic indicators at the time of enrollment.

### Key Feature Categories:
- **Demographics & Family Background**: Age at enrollment, gender, nationality, displaced status, marital status, parental qualifications, and occupations.
- **Academic Path**: Application mode, application order, course selected, daytime/evening attendance, and previous qualifications.
- **Academic Performance**: Number of curricular units credited, enrolled, evaluated, approved, and their respective grades for both the 1st and 2nd semesters.
- **Financial Situation**: Tuition fees up to date, debtor status, and scholarship holder.
- **Macroeconomic Factors**: Unemployment rate, inflation rate, and GDP.

## 🔍 Exploratory Data Analysis (EDA) & Statistical Inference

Comprehensive Exploratory Data Analysis (EDA) and rigorous statistical testing were conducted to uncover hidden patterns and significant relationships between the features and the target variable. 

- **Statistical Tests Employed**: ANOVA, Chi-Square contingency, Fisher's Exact test, and Tukey HSD.
- **Insights**: These tests helped identify the most statistically significant predictors of student success and dropout rates, emphasizing the critical role of early academic performance (grades and approved units), financial stability (tuition fees and scholarships), and demographic factors (age at enrollment).

## 🧠 Machine Learning Modeling

The classification task is challenging due to the inherent **class imbalance**, particularly the difficulty in correctly predicting the "Enrolled" class compared to "Graduate" and "Dropout". 

### 1. Handling Imbalance
To address the class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) was integrated into the pipeline to artificially synthesize data points for the minority classes, dramatically improving the F1-scores across all algorithms.

### 2. Algorithms Evaluated
Several classification algorithms were evaluated and compared based on their **Cross-Validation F1-score** (the harmonic mean of precision and recall):
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier 
- XGBoost Classifier
- CatBoost Classifier

**Random Forest** and **XGBoost** consistently outperformed the others and were selected for advanced fine-tuning.

### 3. Hyperparameter Fine-Tuning
A multi-layered approach was taken to squeeze the best performance out of the selected models:
- **RandomizedSearchCV**: Used initially for both Random Forest and XGBoost to broadly explore the hyperparameter space.
- **GridSearchCV**: Applied to **Random Forest** to exhaustively search a narrowed parameter grid, yielding a robust, normally-distributed ROC-AUC score of ~0.828.
- **Optuna**: Applied to **XGBoost**. Because XGBoost has a vast, continuous search space that makes Grid Search computationally infeasible, Optuna's Bayesian optimization was utilized, resulting in a slightly higher ROC-AUC of ~0.829.

### 4. Final Selection & Error Analysis
On the unseen test set, **XGBoost** proved to be the superior model (ROC-AUC: 0.8307, Accuracy: 70.3%, F1-Score: 70.4%). 
- The model excels at identifying **Graduates** (81% F1) and **Dropouts** (71% F1).
- Predicting the transitional **Enrolled** state remains the hardest challenge (41% F1), reflecting real-world ambiguity in intermediate academic statuses.

The final model pipeline (`final_xgboost_dropout_model.joblib`) natively handles scaling, missing value imputation, label encoding, and prediction.

---

## 🚀 Running the Interactive UI

An interactive Gradio Web UI has been developed for real-time predictions. The UI focuses on the 12 most predictive features determined during EDA, automatically handling imputations for the rest.

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
