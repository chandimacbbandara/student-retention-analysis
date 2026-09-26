import sys
import app

# Generate dummy inputs
dummy_inputs = app.randomize()

for model_name in app.models_cache.keys():
    print(f"\n--- Testing model: {model_name} ---")
    try:
        res = app.predict(model_name, *dummy_inputs)
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()
