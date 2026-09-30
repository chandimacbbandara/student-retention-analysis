import json

notebook_path = "/home/chandima-bandara/Desktop/student-retention-analysis/notebooks/03_model_improvement.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences of the Mac path with the Linux path
old_path = "/Users/faizamfairooz/Desktop/SM PRoject/student-retention-analysis"
new_path = "/home/chandima-bandara/Desktop/student-retention-analysis"

content = content.replace(old_path, new_path)

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Path replacements completed successfully.")
