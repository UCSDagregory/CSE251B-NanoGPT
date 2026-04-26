import os
import shutil

# 1. Define your folders
source_dir = "./my_model/checkpoints/"
dest_dir = "./my_model/"  
new_filename = "checkpoint.pt"

# 2. Find the lowest loss checkpoint (Safely filtering for only .pt files!)
all_checkpoints = sorted([f for f in os.listdir(source_dir) if f.endswith('.pt')])
best_ckpt_filename = all_checkpoints[0]

# 3. Build the full file paths
source_path = os.path.join(source_dir, best_ckpt_filename)
dest_path = os.path.join(dest_dir, new_filename)

# 4. Create the destination folder if it doesn't exist, then copy the file
os.makedirs(dest_dir, exist_ok=True)
shutil.copy2(source_path, dest_path)

print(f"Successfully copied {best_ckpt_filename} and renamed it to {new_filename}!")
