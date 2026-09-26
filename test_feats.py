import joblib
pipeline = joblib.load('models/final_elasticnet_model.joblib')
print(pipeline.named_steps)
