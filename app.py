import os
import glob
import joblib
import pandas as pd
import numpy as np
import gradio as gr
import random

# Patch for older sklearn versions (solves _RemainderColsList loading issue)
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# 1. Load Models
models_cache = {}
model_features = {}
label_encoder = None

def load_models():
    global label_encoder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    le_path = os.path.join(models_dir, "target_label_encoder.joblib")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)

    joblib_files = [
        "final_elastic_net_model.joblib",
        "final_lasso_model.joblib",
        "final_xgboost_model.joblib",
        "final_gradient_boosting_model.joblib",
        "final_stacked_ensemble_model.joblib"
    ]
    for f in joblib_files:
        try:
            full_path = os.path.join(models_dir, f)
            model_name = f.replace(".joblib", "")
            artifacts = joblib.load(full_path)
            
            features = []
            pipeline = None
            
            if isinstance(artifacts, dict):
                pipeline = artifacts.get("pipeline")
                if "feature_names_in" in artifacts:
                    features = list(artifacts["feature_names_in"])
                elif hasattr(pipeline, "feature_names_in_"):
                    features = list(pipeline.feature_names_in_)
            else:
                pipeline = artifacts
                if hasattr(pipeline, "feature_names_in_"):
                    features = list(pipeline.feature_names_in_)
                    
            if pipeline is not None and features:
                models_cache[model_name] = pipeline
                model_features[model_name] = features
        except Exception as e:
            print(f"Failed to load {f}: {e}")

load_models()

# 2. Setup Features
all_features = set()
for feats in model_features.values():
    all_features.update(feats)
all_features = sorted(list(all_features))

# Helper to format names
def format_name(name):
    return name.replace("_", " ").title()

QUALIFICATIONS = [
    ("Secondary Education (12th Year)", 1), ("Higher Ed - Bachelor's", 2),
    ("Higher Ed - Degree", 3), ("Higher Ed - Master's", 4), ("Higher Ed - Doctorate", 5),
    ("Frequency of Higher Ed", 6), ("12th Year - Not Completed", 9),
    ("11th Year - Not Completed", 10), ("7th Year (Old)", 11), ("Other - 11th Year", 12),
    ("2nd year comp. high school", 13), ("10th Year", 14), ("General commerce", 18),
    ("Basic Ed 3rd Cycle (9th-11th)", 19), ("Comp. High School", 20),
    ("Technical-professional", 22), ("Comp. High School - not concluded", 25),
    ("7th year", 26), ("2nd cycle general high school", 27), ("9th Year - Not Completed", 29),
    ("8th year", 30), ("General Course Admin/Commerce", 31), ("Supp. Accounting/Admin", 33),
    ("Unknown", 34), ("Can't read or write", 35), ("Can read (no 4th year)", 36),
    ("Basic ed 1st cycle (4th/5th)", 37), ("Basic Ed 2nd Cycle (6th-8th)", 38),
    ("Technological specialization", 39), ("Higher ed - degree (1st cycle)", 40),
    ("Specialized higher studies", 41), ("Professional higher technical", 42),
    ("Higher Ed - Master (2nd cycle)", 43), ("Higher Ed - Doctorate (3rd cycle)", 44)
]

OCCUPATIONS = [
    ("Student", 0), ("Legislative/Executive/Directors", 1), ("Specialists in Intellectual/Scientific", 2),
    ("Intermediate Technicians/Professions", 3), ("Administrative staff", 4), ("Personal Services/Security/Sellers", 5),
    ("Farmers/Skilled Agriculture", 6), ("Skilled Industry/Construction", 7), ("Installation/Machine Operators", 8),
    ("Unskilled Workers", 9), ("Armed Forces Professions", 10), ("Other Situation", 90), ("Unknown/Blank", 99),
    ("Armed Forces Officers", 101), ("Armed Forces Sergeants", 102), ("Other Armed Forces", 103),
    ("Directors admin/commercial", 112), ("Hotel, catering, trade directors", 114), ("Specialists physical sciences/engineering", 121),
    ("Health professionals", 122), ("Teachers", 123), ("Specialists finance/admin/PR", 124), ("ICT Specialists", 125),
    ("Intermediate science/engineering tech", 131), ("Intermediate health tech", 132), ("Intermediate legal/social tech", 134),
    ("ICT technicians", 135), ("Office workers/secretaries", 141), ("Data/accounting/registry operators", 143),
    ("Other administrative support", 144), ("Personal service workers", 151), ("Sellers", 152), ("Personal care workers", 153),
    ("Protection/security services", 154), ("Market-oriented farmers", 161), ("Subsistence farmers", 163),
    ("Skilled construction workers", 171), ("Skilled workers metallurgy", 172), ("Skilled workers printing/precision", 173),
    ("Skilled workers electricity/electronics", 174), ("Workers food/wood/clothing", 175), ("Fixed plant/machine operators", 181),
    ("Assembly workers", 182), ("Vehicle drivers/mobile equipment", 183), ("Cleaning workers", 191),
    ("Unskilled workers agriculture/fisheries", 192), ("Unskilled workers extractive/construction", 193),
    ("Meal preparation assistants", 194), ("Street vendors/providers", 195)
]

# Options mapping for categorical variables
def get_options(name):
    lower = name.lower()
    if 'gender' in lower: return [("Male", 0), ("Female", 1)]
    if any(k in lower for k in ['debtor', 'tuition', 'scholarship', 'displaced', 'international', 'special needs']): 
        return [("No", 0), ("Yes", 1)]
    if 'attendance' in lower: return [("Evening", 0), ("Daytime", 1)]
    if 'marital' in lower: return [("Single", 1), ("Married", 2), ("Widower", 3), ("Divorced", 4), ("Common-law", 5), ("Legally separated", 6)]
    if 'qualification' in lower: return QUALIFICATIONS
    if 'occupation' in lower: return OCCUPATIONS
    if 'course' in lower: return [
        ("Biofuel Prod", 1), ("Animation", 2), ("Social Service (eve)", 3),
        ("Agronomy", 4), ("Comm Design", 5), ("Vet Nursing", 6),
        ("Informatics Eng", 7), ("Equiniculture", 8), ("Management", 9),
        ("Social Service", 10), ("Tourism", 11), ("Nursing", 12),
        ("Oral Hygiene", 13), ("Advertising", 14), ("Journalism", 15),
        ("Basic Education", 16), ("Management (eve)", 17)
    ]
    if 'application mode' in lower: return [
        ("1st phase (general)", 1), ("Ordinance 612/93", 2), ("1st phase (Azores)", 3),
        ("Other higher courses", 4), ("Ordinance 854-B/99", 5), ("Int. student", 6),
        ("1st phase (Madeira)", 7), ("2nd phase (general)", 8), ("3rd phase (general)", 9),
        ("Ord 533-A/99(b2)", 10), ("Ord 533-A/99(b3)", 11), ("Over 23 years", 12),
        ("Transfer", 13), ("Change in course", 14), ("Tech spec diploma", 15),
        ("Change in inst/course", 16), ("Short cycle diploma", 17), ("Change (Int)", 18)
    ]
    return None

def get_info(name):
    lower = name.lower()
    if 'age' in lower: return "Age of the student at enrollment"
    if 'gender' in lower: return "Student's gender"
    if 'nationality' in lower: return "Student's nationality"
    if 'marital' in lower: return "Student's marital status"
    if 'displaced' in lower: return "Whether the student moved from their hometown to study"
    if 'international' in lower: return "Whether the student is international"
    if 'special needs' in lower: return "Whether the student has special educational needs"
    if 'mother' in lower and 'qualification' in lower: return "Highest education level of the mother"
    if 'father' in lower and 'qualification' in lower: return "Highest education level of the father"
    if 'mother' in lower and 'occupation' in lower: return "Profession of the mother"
    if 'father' in lower and 'occupation' in lower: return "Profession of the father"
    if 'course' in lower: return "Undergraduate degree program the student is enrolled in"
    if 'application mode' in lower: return "Method or quota used by the student to apply"
    if 'application order' in lower: return "Preference order of this course (between 0 - first choice; and 9 - last choice)"
    if 'previous qualification' in lower and 'grade' not in lower: return "Education level obtained prior to enrollment"
    if 'previous qualification' in lower and 'grade' in lower: return "Grade of previous qualification (between 0 and 200)"
    if 'admission grade' in lower: return "Admission grade (between 0 and 200)"
    if 'attendance' in lower: return "Whether the student attends daytime or evening classes"
    if 'debtor' in lower: return "Whether the student has outstanding debts to the institution"
    if 'tuition' in lower: return "Whether the student's tuition fees are up to date"
    if 'scholarship' in lower: return "Whether the student is a scholarship holder"
    if '1st sem' in lower and 'credited' in lower: return "Number of 1st semester curricular units credited"
    if '1st sem' in lower and 'enrolled' in lower: return "Number of 1st semester curricular units enrolled"
    if '1st sem' in lower and 'evaluations' in lower: return "Number of 1st semester evaluations"
    if '1st sem' in lower and 'approved' in lower: return "Number of 1st semester curricular units approved"
    if '1st sem' in lower and 'grade' in lower: return "Average grade in 1st semester (between 0 and 20)"
    if '1st sem' in lower and 'without' in lower: return "Number of 1st semester curricular units without evaluations"
    if '2nd sem' in lower and 'credited' in lower: return "Number of 2nd semester curricular units credited"
    if '2nd sem' in lower and 'enrolled' in lower: return "Number of 2nd semester curricular units enrolled"
    if '2nd sem' in lower and 'evaluations' in lower: return "Number of 2nd semester evaluations"
    if '2nd sem' in lower and 'approved' in lower: return "Number of 2nd semester curricular units approved"
    if '2nd sem' in lower and 'grade' in lower: return "Average grade in 2nd semester (between 0 and 20)"
    if '2nd sem' in lower and 'without' in lower: return "Number of 2nd semester curricular units without evaluations"
    if 'unemployment' in lower: return "National unemployment rate at the time of enrollment"
    if 'inflation' in lower: return "National inflation rate at the time of enrollment"
    if 'gdp' in lower: return "National GDP growth at the time of enrollment"
    if 'approval' in lower and 'rate' in lower: return "Engineered: Percentage of enrolled units approved"
    if 'trend' in lower or 'risk' in lower: return "Engineered feature metric"
    return "Student attribute"

def predict(model_name, *args):
    # args is a tuple corresponding to all_features
    inputs = {f: v for f, v in zip(all_features, args)}
    
    # Get required features
    req_features = model_features[model_name]
    row = {f: inputs.get(f, 0.0) for f in req_features}
    
    df = pd.DataFrame([row], columns=req_features)
    pipeline = models_cache[model_name]
    
    proba = pipeline.predict_proba(df)[0]
    classes = label_encoder.classes_
    
    # Gradio expects a dictionary of {class_name: probability} for Label output
    return {str(cls): float(p) for cls, p in zip(classes, proba)}

def randomize(*args):
    vals = []
    for f in all_features:
        opts = get_options(f)
        if opts:
            # Pick random option value
            vals.append(random.choice(opts)[1])
        else:
            lower = f.lower()
            if 'age' in lower: vals.append(random.randint(18, 47))
            elif 'grade' in lower: vals.append(round(random.uniform(10, 20), 1))
            elif any(k in lower for k in ['rate', 'gdp', 'inflation']): vals.append(round(random.uniform(-2, 8), 1))
            elif any(k in lower for k in ['evaluations', 'enrolled', 'approved', 'credited']): vals.append(random.randint(0, 14))
            elif any(k in lower for k in ['trend', 'risk', 'mean']): vals.append(round(random.uniform(0, 5), 2))
            else: vals.append(round(random.uniform(0, 10), 1))
    return vals

# Force light mode for the whole page
js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f5f7fb",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 10px 35px rgba(15, 23, 42, 0.07)",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f8fafc",
    button_secondary_text_color="#172033",
    input_background_fill="#f8fafc",
    input_border_color="#e2e8f0",
    block_label_text_color="#64748b",
    block_title_text_color="#172033",
    body_text_color="#172033",
    border_color_primary="#e2e8f0"
)

custom_css = """
#app-header { text-align: center; padding: 2rem 0; }
#app-header h1 { color: #172033; font-weight: 700; margin-bottom: 0.5rem; font-size: 2.5rem; }
#app-header h3 { color: #64748b; font-weight: 400; margin-top: 0; font-size: 1.1rem; }

/* Highlight Model Selection */
#model-select { border: 2px solid #2563eb !important; border-radius: 8px; box-shadow: 0 0 10px rgba(37,99,235,0.15) !important; background-color: #eff6ff !important; }

/* Highlight Random Fill Button */
#random-fill-btn { background-color: #ea580c !important; color: white !important; border: none !important; font-weight: 600 !important; font-size: 1.05rem !important; transition: all 0.2s; }
#random-fill-btn:hover { background-color: #c2410c !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(234,88,12,0.2) !important; }

/* Dictionary Table Styles */
#dict-tab table { display: block; overflow-x: auto; white-space: nowrap; width: 100%; border-collapse: collapse; }
#dict-tab th, #dict-tab td { border: 1px solid #e2e8f0; padding: 12px 16px; white-space: normal; vertical-align: top; }
#dict-tab th { background-color: #f8fafc; font-weight: 600; text-align: left; white-space: nowrap; }
#dict-tab td:nth-child(2) { min-width: 180px; font-weight: 600; color: #1e293b; } /* Variable Name */
#dict-tab td:nth-child(6) { min-width: 450px; } /* Description */
#dict-tab tr:nth-child(even) { background-color: #fafafa; }
"""



# 3. Build UI
with gr.Blocks(theme=custom_theme, css=custom_css, js=js_func) as app:
    with gr.Column(elem_id="app-header"):
        gr.Markdown("# 🎓 Student Retention AI\n### Advanced predictive analytics for student success")
    
    with gr.Tabs():
        with gr.Tab("Prediction"):
            
            model_choices = []
            for m in models_cache.keys():
                display_name = format_name(m)
                if m == "final_stacked_ensemble_model": display_name += " (Best Choice)"
                model_choices.append((display_name, m))
                
            default_model = model_choices[-1][1]
            selected_model = gr.Dropdown(choices=model_choices, value=default_model, label="Select Prediction Model", elem_id="model-select")
            
            gr.Markdown("Fill in the student details below, or click Random Fill to generate sample data.")
            random_btn = gr.Button("🎲 Random Fill", elem_id="random-fill-btn")
            
            input_components = []
            with gr.Row():
                # Split into 3 columns for better layout
                cols = [gr.Column() for _ in range(3)]
                
                for idx, feat in enumerate(all_features):
                    with cols[idx % 3]:
                        opts = get_options(feat)
                        is_visible = feat in model_features[default_model]
                        info_text = get_info(feat)
                        if opts:
                            comp = gr.Dropdown(choices=opts, label=format_name(feat), info=info_text, visible=is_visible)
                        else:
                            comp = gr.Number(label=format_name(feat), value=0, info=info_text, visible=is_visible)
                        input_components.append(comp)
            
            submit_btn = gr.Button("Predict Student Outcome", variant="primary")
            output_label = gr.Label(num_top_classes=3, label="Prediction Probability")
            
            def update_visibility(model_name):
                req_feats = model_features.get(model_name, [])
                return [gr.update(visible=(feat in req_feats)) for feat in all_features]
            
            # Wire up events
            selected_model.change(fn=update_visibility, inputs=selected_model, outputs=input_components)
            random_btn.click(fn=randomize, inputs=input_components, outputs=input_components)
            submit_btn.click(fn=predict, inputs=[selected_model] + input_components, outputs=output_label)

        with gr.Tab("How It Works"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "how_it_works.html"), "r", encoding="utf-8") as f:
                html_content = f.read()
            import base64
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            iframe_code = f'<iframe src="data:text/html;base64,{encoded_html}" width="100%" style="height: 85vh; border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>'
            gr.HTML(value=iframe_code)
        with gr.Tab("Dataset Dictionary"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_dictionary.md"), "r", encoding="utf-8") as f:
                dict_content = f.read()
            with gr.Column(elem_id="dict-tab"):
                gr.Markdown(dict_content)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
