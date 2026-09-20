import nbformat

for nb_name in ['notebooks/01_Model_Pipeline.ipynb', 'notebooks/02_Statistical Inference.ipynb']:
    print(f"\n--- {nb_name} ---")
    try:
        with open(nb_name, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown':
                print(f"MD [{i}]: {cell.source[:200]}")
            elif cell.cell_type == 'code':
                print(f"Code [{i}]: {cell.source.splitlines()[0] if cell.source else ''}")
    except Exception as e:
        print(f"Error: {e}")
