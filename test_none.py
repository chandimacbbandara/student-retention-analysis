import app
model_name = list(app.models_cache.keys())[0]

# Simulate Gradio passing None for empty dropdowns
dummy_inputs = [None] * len(app.all_features)

try:
    res = app.predict(model_name, *dummy_inputs)
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
