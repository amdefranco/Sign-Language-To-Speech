import json

# Path to your JSON file
json_path = "labels.json"

# Load the JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Create index → gloss mapping
index_to_gloss = {}

for idx, entry in enumerate(data):
    gloss = entry.get("gloss")
    if gloss is not None:
        index_to_gloss[idx] = gloss
    else:
        print(f"[Warning] Entry at index {idx} has no 'gloss' key")

output_path = "label_map.json"
with open(output_path, "w") as f:
    json.dump(index_to_gloss, f, indent=2)

print(f"Saved index-to-gloss mapping to {output_path}")
