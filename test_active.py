import joblib
import numpy as np

for m in ['final_elasticnet_model', 'final_gradient_boosting_model', 'final_lasso_model', 'final_xgboost_model']:
    p = joblib.load(f'models/{m}.joblib')
    clf = p.named_steps['clf']
    # feature names out from preprocessor
    feat_names = p.named_steps['prep'].get_feature_names_out()
    
    if hasattr(clf, 'coef_'):
        active = np.sum(np.abs(clf.coef_), axis=0) > 0
    elif hasattr(clf, 'feature_importances_'):
        active = clf.feature_importances_ > 0
    else:
        active = np.ones(len(feat_names), dtype=bool)
        
    print(f"{m}: {sum(active)}/{len(feat_names)} active features out of preprocessor")
