import joblib
import glob
import sys
import sklearn.compose._column_transformer

class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

models = glob.glob("/home/chandima-bandara/Desktop/student-retention-analysis/models/*.joblib")
for m in models:
    if "label_encoder" in m:
        continue
    try:
        print(f"Loading {m}...")
        artifacts = joblib.load(m)
        if isinstance(artifacts, dict):
            print("Keys:", artifacts.keys())
            if "feature_names_in" in artifacts:
                print("Features:", artifacts["feature_names_in"])
            elif "pipeline" in artifacts:
                try:
                    print("Pipeline features:", artifacts["pipeline"].feature_names_in_)
                except:
                    pass
        else:
            print("Artifact type:", type(artifacts))
            if hasattr(artifacts, "feature_names_in_"):
                print("Features:", artifacts.feature_names_in_)
        print("-" * 50)
    except Exception as e:
        print(f"Error loading {m}: {e}")
