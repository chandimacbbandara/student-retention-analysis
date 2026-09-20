import pandas as pd
df = pd.read_csv('data/dataset.csv')
for col in df.columns:
    if col == 'Target': continue
    if df[col].dtype == 'object':
        print(f"{col} (Categorical): {df[col].unique().tolist()}")
    else:
        print(f"{col} (Numeric): min={df[col].min()}, max={df[col].max()}")
