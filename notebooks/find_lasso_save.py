import json

with open('/home/chandima-bandara/Desktop/student-retention-analysis/notebooks/statical_model.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'final_lasso_model.joblib' in source or 'final_elastic_net_model.joblib' in source or 'joblib.dump' in source:
            print("---")
            print(source)
