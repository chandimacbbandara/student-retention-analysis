import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
import joblib
artifacts = joblib.load('models/final_xgboost_dropout_model.joblib')
print("Features:", artifacts['feature_names_in'])
print("Keys:", artifacts.keys())
