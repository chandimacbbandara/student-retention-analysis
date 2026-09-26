import app
print("All features:", len(app.all_features))
for m, feats in app.model_features.items():
    print(f"{m}: {len(feats)} features")
