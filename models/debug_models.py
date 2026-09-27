import joblib
import inspect

models = {
    'lasso': '/home/chandima-bandara/Desktop/student-retention-analysis/models/final_lasso_model.joblib',
    'elastic_net': '/home/chandima-bandara/Desktop/student-retention-analysis/models/final_elastic_net_model.joblib',
    'xgboost': '/home/chandima-bandara/Desktop/student-retention-analysis/models/final_xgboost_model.joblib',
    'gradient_boosting': '/home/chandima-bandara/Desktop/student-retention-analysis/models/final_gradient_boosting_model.joblib'
}

for name, path in models.items():
    try:
        model = joblib.load(path)
        print(f"--- {name} ---")
        print(f"Type: {type(model)}")
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # find np.float64
                import numpy as np
                for k, v in params.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, np.float64) or isinstance(v2, np.float32) or type(v2).__name__ == 'float64':
                                print(f"Found np float in dict: {k} -> {k2} = {v2}")
                    if isinstance(v, np.float64) or isinstance(v, np.float32) or type(v).__name__ == 'float64':
                        print(f"Found np float in params: {k} = {v}")
            except Exception as e:
                print(f"Error getting params: {e}")
        else:
            print("No get_params()")
            print(model)
    except Exception as e:
        print(f"Failed to load {name}: {e}")

