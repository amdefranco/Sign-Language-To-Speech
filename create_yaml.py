import os

train_path = os.path.abspath("yolo-wlasl-classify/train")
val_path = os.path.abspath("yolo-wlasl-classify/val")

with open("data.yaml", "w") as f:
    f.write(f"train: {train_path}\n")
    f.write(f"val: {val_path}\n")
    f.write("nc: 119\n")


with open("data.yaml", "a") as f:
    names = [str(i) for i in range(119)]
    f.write(f"names: {names}\n")
