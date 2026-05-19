import json

with open("notebooks/roi_lens_analysis.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            line = line.replace("'outputs/", "'../outputs/")
            line = line.replace("'data/", "'../data/")
            new_source.append(line)
        cell["source"] = new_source

with open("notebooks/roi_lens_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed all paths.")
