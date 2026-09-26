import os
import glob
import joblib
import pandas as pd
import numpy as np
import gradio as gr
import random

# HF ZeroGPU compatibility fix
try:
    import spaces
    @spaces.GPU
    def _hf_spaces_gpu_init():
        pass
except Exception:
    pass

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

def get_options(name):
    lower = name.lower()
    if 'gender' in lower: return [("Male", 0), ("Female", 1)]
    if any(k in lower for k in ['debtor', 'tuition', 'scholarship', 'displaced', 'international', 'special needs', 'zero_approved']): 
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
    if 'displaced' in lower: return "Whether student moved from hometown"
    if 'international' in lower: return "Whether student is international"
    if 'special needs' in lower: return "Whether student has special needs"
    if 'mother' in lower and 'qualification' in lower: return "Mother's highest education"
    if 'father' in lower and 'qualification' in lower: return "Father's highest education"
    if 'mother' in lower and 'occupation' in lower: return "Mother's profession"
    if 'father' in lower and 'occupation' in lower: return "Father's profession"
    if 'course' in lower: return "Degree program enrolled in"
    if 'application mode' in lower: return "Method used to apply"
    if 'application order' in lower: return "Preference order (0-9)"
    if 'previous qualification' in lower and 'grade' not in lower: return "Prior education level"
    if 'previous qualification' in lower and 'grade' in lower: return "Prior qualification grade (0-200)"
    if 'admission grade' in lower: return "Admission grade (0-200)"
    if 'attendance' in lower: return "Daytime or evening classes"
    if 'debtor' in lower: return "Outstanding debt status"
    if 'tuition' in lower: return "Tuition fees up to date"
    if 'scholarship' in lower: return "Scholarship recipient"
    if '1st sem' in lower and 'credited' in lower: return "1st sem units credited"
    if '1st sem' in lower and 'enrolled' in lower: return "1st sem units enrolled"
    if '1st sem' in lower and 'evaluations' in lower: return "1st sem evaluations count"
    if '1st sem' in lower and 'approved' in lower: return "1st sem units approved"
    if '1st sem' in lower and 'grade' in lower: return "1st sem average grade (0-20)"
    if '1st sem' in lower and 'without' in lower: return "1st sem units without evaluations"
    if '2nd sem' in lower and 'credited' in lower: return "2nd sem units credited"
    if '2nd sem' in lower and 'enrolled' in lower: return "2nd sem units enrolled"
    if '2nd sem' in lower and 'evaluations' in lower: return "2nd sem evaluations count"
    if '2nd sem' in lower and 'approved' in lower: return "2nd sem units approved"
    if '2nd sem' in lower and 'grade' in lower: return "2nd sem average grade (0-20)"
    if '2nd sem' in lower and 'without' in lower: return "2nd sem units without evaluations"
    if 'unemployment' in lower: return "Unemployment rate at enrollment"
    if 'inflation' in lower: return "Inflation rate at enrollment"
    if 'gdp' in lower: return "GDP growth at enrollment"
    if 'approval' in lower and 'rate' in lower: return "Percentage of enrolled units approved"
    if 'trend' in lower or 'risk' in lower: return "Engineered performance indicator"
    return "Student attribute"

def predict(model_name, *args):
    inputs = {f: v for f, v in zip(all_features, args)}
    req_features = model_features[model_name]
    row = {f: inputs.get(f, 0.0) for f in req_features}
    
    df = pd.DataFrame([row], columns=req_features)
    pipeline = models_cache[model_name]
    
    proba = pipeline.predict_proba(df)[0]
    classes = label_encoder.classes_
    return {str(cls): float(p) for cls, p in zip(classes, proba)}

def randomize(*args):
    vals = []
    for f in all_features:
        opts = get_options(f)
        if opts:
            vals.append(random.choice(opts)[1])
        else:
            lower = f.lower()
            if 'age' in lower: vals.append(random.randint(18, 45))
            elif 'grade' in lower: vals.append(round(random.uniform(11, 18), 1))
            elif any(k in lower for k in ['rate', 'gdp', 'inflation']): vals.append(round(random.uniform(-1.5, 5.0), 1))
            elif any(k in lower for k in ['evaluations', 'enrolled', 'approved', 'credited']): vals.append(random.randint(2, 10))
            elif any(k in lower for k in ['trend', 'risk', 'mean']): vals.append(round(random.uniform(0.1, 4.0), 2))
            else: vals.append(round(random.uniform(0, 10), 1))
    return vals

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
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 10px 30px rgba(15, 23, 42, 0.04)",
    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color="#0f172a",
    input_background_fill="#f8fafc",
    input_border_color="#cbd5e1",
    block_label_text_color="#475569",
    block_title_text_color="#0f172a",
    body_text_color="#0f172a",
    border_color_primary="#e2e8f0"
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

body, html {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

h1, h2, h3, .hero-title {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom Header Dashboard */
.custom-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: white;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.12);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.custom-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0,0,0,0) 70%);
    border-radius: 50%;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(165, 180, 252, 0.3);
    color: #a5b4fc;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 750px;
    margin-bottom: 24px;
    line-height: 1.6;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin-top: 20px;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: transform 0.2s ease, background 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.12);
}

.kpi-icon {
    font-size: 1.5rem;
}

.kpi-val {
    display: block;
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'Outfit', sans-serif;
}

.kpi-lbl {
    display: block;
    font-size: 0.75rem;
    color: #cbd5e1;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Control Panel Box */
#control-panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.03) !important;
    margin-bottom: 20px !important;
}

#model-select {
    border: 2px solid #6366f1 !important;
    border-radius: 10px !important;
    background-color: #f5f3ff !important;
    transition: all 0.2s ease !important;
}

#model-select:focus-within {
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}

#random-fill-btn {
    background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.25) !important;
    transition: all 0.25s ease !important;
}

#random-fill-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(234, 88, 12, 0.35) !important;
}

#predict-btn {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    padding: 16px !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.28) !important;
    transition: all 0.25s ease !important;
    margin-top: 16px !important;
}

#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.4) !important;
}

/* Category Card Headers */
.category-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.03);
}

.category-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f1f5f9;
}

.category-title span {
    font-size: 1.3rem;
}

/* Form Controls */
.gr-input, .gr-dropdown, select, input {
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

/* Dictionary Table Styles */
#dict-tab table { display: block; overflow-x: auto; white-space: nowrap; width: 100%; border-collapse: collapse; }
#dict-tab th, #dict-tab td { border: 1px solid #e2e8f0; padding: 12px 16px; white-space: normal; vertical-align: top; }
#dict-tab th { background-color: #f8fafc; font-weight: 600; text-align: left; white-space: nowrap; }
#dict-tab td:nth-child(2) { min-width: 180px; font-weight: 600; color: #1e293b; }
#dict-tab td:nth-child(6) { min-width: 450px; }
#dict-tab tr:nth-child(even) { background-color: #fafafa; }
"""

# Helper to classify features into visual categories
def get_feature_category(feat):
    lower = feat.lower()
    if any(k in lower for k in ['1st sem', '2nd sem', 'approval', 'grade', 'eval', 'approved_total']):
        return 'academic'
    elif any(k in lower for k in ['age', 'gender', 'marital', 'mother', 'father', 'displaced', 'special needs', 'nationality', 'international']):
        return 'demographic'
    elif any(k in lower for k in ['course', 'application', 'previous qualification', 'attendance']):
        return 'application'
    else:
        return 'financial'

# 3. Build UI
with gr.Blocks(theme=custom_theme, css=custom_css, js=js_func) as app:
    
    # Custom Dashboard Header
    gr.HTML("""
    <div class="custom-hero">
        <div class="hero-badge">✨ Multi-Model Stacking Ensemble AI</div>
        <h1 class="hero-title">Student Retention Analytics & Prediction</h1>
        <div class="hero-subtitle">
            An advanced machine learning framework analyzing academic performance, socioeconomics, and student background to classify retention outcomes (Graduate, Enrolled, or Dropout).
        </div>
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-icon">⚡</span>
                <div>
                    <span class="kpi-val">89.04%</span>
                    <span class="kpi-lbl">Stacked ROC-AUC</span>
                </div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">🎯</span>
                <div>
                    <span class="kpi-val">77.76%</span>
                    <span class="kpi-lbl">Accuracy Score</span>
                </div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">🤖</span>
                <div>
                    <span class="kpi-val">5 Models</span>
                    <span class="kpi-lbl">Tuned Algorithms</span>
                </div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">📊</span>
                <div>
                    <span class="kpi-val">4,424</span>
                    <span class="kpi-lbl">Dataset Records</span>
                </div>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🎯 Prediction Engine"):
            
            model_choices = []
            for m in models_cache.keys():
                display_name = format_name(m)
                if m == "final_stacked_ensemble_model": display_name += " ★ (Best Performance)"
                model_choices.append((display_name, m))
                
            if not model_choices:
                raise RuntimeError("No machine learning models were loaded successfully. Check dependencies in requirements.txt.")
            default_model = model_choices[-1][1]
            
            with gr.Column(elem_id="control-panel"):
                with gr.Row():
                    selected_model = gr.Dropdown(
                        choices=model_choices, 
                        value=default_model, 
                        label="Choose Prediction Algorithm", 
                        elem_id="model-select",
                        scale=3
                    )
                    random_btn = gr.Button("🎲 Auto-Fill Random Profile", elem_id="random-fill-btn", scale=1)
            
            gr.Markdown("#### Fill in student parameters across the categories below to generate outcome predictions:")
            
            # Map features to categories maintaining exact order in all_features
            academic_feats = [f for f in all_features if get_feature_category(f) == 'academic']
            demographic_feats = [f for f in all_features if get_feature_category(f) == 'demographic']
            application_feats = [f for f in all_features if get_feature_category(f) == 'application']
            financial_feats = [f for f in all_features if get_feature_category(f) == 'financial']
            
            feature_comp_map = {}
            
            with gr.Tabs():
                with gr.Tab("📘 Academic Performance (1st & 2nd Sem)"):
                    with gr.Row():
                        cols = [gr.Column() for _ in range(3)]
                        for idx, feat in enumerate(academic_feats):
                            with cols[idx % 3]:
                                opts = get_options(feat)
                                is_vis = feat in model_features[default_model]
                                info_txt = get_info(feat)
                                if opts:
                                    comp = gr.Dropdown(choices=opts, label=format_name(feat), info=info_txt, visible=is_vis)
                                else:
                                    comp = gr.Number(label=format_name(feat), value=0, info=info_txt, visible=is_vis)
                                feature_comp_map[feat] = comp

                with gr.Tab("👤 Demographics & Family Background"):
                    with gr.Row():
                        cols = [gr.Column() for _ in range(3)]
                        for idx, feat in enumerate(demographic_feats):
                            with cols[idx % 3]:
                                opts = get_options(feat)
                                is_vis = feat in model_features[default_model]
                                info_txt = get_info(feat)
                                if opts:
                                    comp = gr.Dropdown(choices=opts, label=format_name(feat), info=info_txt, visible=is_vis)
                                else:
                                    comp = gr.Number(label=format_name(feat), value=0, info=info_txt, visible=is_vis)
                                feature_comp_map[feat] = comp

                with gr.Tab("📝 Application & Prior Qualifications"):
                    with gr.Row():
                        cols = [gr.Column() for _ in range(3)]
                        for idx, feat in enumerate(application_feats):
                            with cols[idx % 3]:
                                opts = get_options(feat)
                                is_vis = feat in model_features[default_model]
                                info_txt = get_info(feat)
                                if opts:
                                    comp = gr.Dropdown(choices=opts, label=format_name(feat), info=info_txt, visible=is_vis)
                                else:
                                    comp = gr.Number(label=format_name(feat), value=0, info=info_txt, visible=is_vis)
                                feature_comp_map[feat] = comp

                with gr.Tab("💳 Financial & Macroeconomic Indicators"):
                    with gr.Row():
                        cols = [gr.Column() for _ in range(3)]
                        for idx, feat in enumerate(financial_feats):
                            with cols[idx % 3]:
                                opts = get_options(feat)
                                is_vis = feat in model_features[default_model]
                                info_txt = get_info(feat)
                                if opts:
                                    comp = gr.Dropdown(choices=opts, label=format_name(feat), info=info_txt, visible=is_vis)
                                else:
                                    comp = gr.Number(label=format_name(feat), value=0, info=info_txt, visible=is_vis)
                                feature_comp_map[feat] = comp

            # Re-construct input_components in EXACT order of all_features
            input_components = [feature_comp_map[f] for f in all_features]

            submit_btn = gr.Button("⚡ Run Retention Prediction Engine", variant="primary", elem_id="predict-btn")
            output_label = gr.Label(num_top_classes=3, label="Predicted Outcome Probabilities")
            
            def update_visibility(model_name):
                req_feats = model_features.get(model_name, [])
                return [gr.update(visible=(feat in req_feats)) for feat in all_features]
            
            # Wire up events
            selected_model.change(fn=update_visibility, inputs=selected_model, outputs=input_components)
            random_btn.click(fn=randomize, inputs=input_components, outputs=input_components)
            submit_btn.click(fn=predict, inputs=[selected_model] + input_components, outputs=output_label)

        with gr.Tab("💡 How It Works & Architecture"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "how_it_works.html"), "r", encoding="utf-8") as f:
                html_content = f.read()
            import base64
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            iframe_code = f'<iframe src="data:text/html;base64,{encoded_html}" width="100%" style="height: 88vh; border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.06);"></iframe>'
            gr.HTML(value=iframe_code)

        with gr.Tab("📖 Dataset Dictionary & Feature Docs"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_dictionary.md"), "r", encoding="utf-8") as f:
                dict_content = f.read()
            with gr.Column(elem_id="dict-tab"):
                gr.Markdown(dict_content)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
