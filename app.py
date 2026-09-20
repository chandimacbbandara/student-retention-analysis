import gradio as gr
import pandas as pd
import numpy as np
import joblib
import warnings
import sklearn.compose._column_transformer

# Patch for older sklearn versions (solves _RemainderColsList loading issue)
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

warnings.filterwarnings('ignore')

# Load the model bundle
MODEL_PATH = "models/final_xgboost_dropout_model.joblib"
try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["pipeline"]
    le = artifacts["label_encoder"]
    feature_cols = artifacts["feature_names_in"]
except Exception as e:
    model, le, feature_cols = None, None, None
    print(f"Error loading model: {e}")

# The most predictive features specified by the user snippet
INPUT_FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Gender",
    "Displaced",
    "International",
]

def predict_retention(*args):
    if model is None:
        return "Model not loaded.", ""
    
    # args is a tuple of the inputs in order of INPUT_FEATURES
    inputs_dict = dict(zip(INPUT_FEATURES, args))
    
    # Initialize a row with 0s for everything expected by the model
    row = {c: 0.0 for c in feature_cols}
    
    # Update row with user inputs if they exist in feature_cols
    for feature_name, val in inputs_dict.items():
        if feature_name in row:
            row[feature_name] = val
            
    # Create DataFrame enforcing column order
    df = pd.DataFrame([row], columns=feature_cols)
    
    # Predict
    try:
        proba = model.predict_proba(df)[0]
        pred_int = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_int])[0]
        
        # Format probabilities
        prob_text = ""
        for cls, p in zip(le.classes_, proba):
            prob_text += f"{cls}: {p*100:.1f}%\n"
            
        return pred_label, prob_text
    except Exception as e:
        return f"Error: {str(e)}", ""

# Define the Gradio interface
with gr.Blocks(title="Student Retention Prediction") as demo:
    gr.Markdown("# Student Retention Prediction System")
    gr.Markdown("Enter the student's details below to predict if they will **Graduate**, **Enrolled**, or **Dropout**. Any unknown features can be left at 0, the model pipeline will handle missing values through imputation.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Demographics & Financials")
            age = gr.Number(label="Age at enrollment", value=0.0)
            tuition = gr.Number(label="Tuition fees up to date (0/1)", value=0.0)
            scholarship = gr.Number(label="Scholarship holder (0/1)", value=0.0)
            debtor = gr.Number(label="Debtor (0/1)", value=0.0)
            gender = gr.Number(label="Gender (0/1)", value=0.0)
            displaced = gr.Number(label="Displaced (0/1)", value=0.0)
            international = gr.Number(label="International (0/1)", value=0.0)
            
        with gr.Column():
            gr.Markdown("### Academic Performance")
            adm_grade = gr.Number(label="Admission grade", value=0.0)
            cu1_grade = gr.Number(label="Curricular units 1st sem (grade)", value=0.0)
            cu2_grade = gr.Number(label="Curricular units 2nd sem (grade)", value=0.0)
            cu1_appr = gr.Number(label="Curricular units 1st sem (approved)", value=0.0)
            cu2_appr = gr.Number(label="Curricular units 2nd sem (approved)", value=0.0)

    # Order must match INPUT_FEATURES exactly!
    inputs_list = [
        age, adm_grade, cu1_grade, cu2_grade, cu1_appr, cu2_appr, 
        tuition, scholarship, debtor, gender, displaced, international
    ]
            
    predict_btn = gr.Button("Predict Retention Status", variant="primary")
    
    with gr.Row():
        pred_output = gr.Textbox(label="Predicted Outcome")
        prob_output = gr.Textbox(label="Class Probabilities")
        
    predict_btn.click(fn=predict_retention, inputs=inputs_list, outputs=[pred_output, prob_output])

if __name__ == "__main__":
    demo.launch()
