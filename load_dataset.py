from datasets import load_dataset, Video

# Load the WLASL dataset from Hugging Face
dataset = load_dataset("Voxel51/WLASL")

train_split = dataset["train"]
train_split = train_split.cast_column("video", Video(decode=False))

for sample in train_split:
    print(sample)
    break