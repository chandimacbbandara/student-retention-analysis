import json

with open('notebooks/03_model_improvement.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'stack_acc =' in source or 'print(stack_acc)' in source or 'data = {' in source:
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out.get('output_type') == 'stream':
                    print(out.get('text'))
                elif out.get('output_type') == 'execute_result':
                    print(out.get('data', {}).get('text/plain'))
