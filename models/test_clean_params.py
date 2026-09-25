import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Create an estimator with a numpy float parameter
gb = GradientBoostingClassifier(learning_rate=np.float64(0.05285382203816726))
pipe = Pipeline([('clf', gb)])

from sklearn.base import clone

try:
    clone(pipe)
    print("Clone worked initially! (Wait, maybe scikit-learn updated?)")
except Exception as e:
    print(f"Clone failed initially: {type(e).__name__}: {e}")

# Now clean it
params = pipe.get_params(deep=True)
cleaned = {}
for k, v in params.items():
    if isinstance(v, (np.floating, np.float32, np.float64)):
        cleaned[k] = float(v)
    elif isinstance(v, (np.integer, np.int32, np.int64)):
        cleaned[k] = int(v)

if cleaned:
    pipe.set_params(**cleaned)

try:
    clone(pipe)
    print("Clone worked after cleaning!")
except Exception as e:
    print(f"Clone failed after cleaning: {type(e).__name__}: {e}")

