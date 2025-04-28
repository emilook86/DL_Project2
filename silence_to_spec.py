# -*- coding: utf-8 -*-

import os
import glob
import random
import shutil

# Œcie¿ka do folderu z gotowymi spektrogramami
input_folder = r"data\background_spectrograms"

# Œcie¿ka do folderu wyjœciowego (tam bêd¹ train/val/test)
output_base_folder = r"data\split_background"

# Ustaw proporcje
train_split = 0.7
val_split = 0.1
test_split = 0.2

# Pobierz wszystkie pliki PNG
all_images = glob.glob(os.path.join(input_folder, "*.png"))
print(f"Znaleziono {len(all_images)} plikow.")

# Pomieszaj kolejnoœæ
random.shuffle(all_images)

# Wyznacz iloœci
total_count = len(all_images)
train_count = int(total_count * train_split)
val_count = int(total_count * val_split)
test_count = total_count - train_count - val_count  # ¿eby siê wszystko zgadza³o

# Podzia³
train_files = all_images[:train_count]
val_files = all_images[train_count:train_count + val_count]
test_files = all_images[train_count + val_count:]

# Funkcja do kopiowania
def copy_files(file_list, subset_name):
    subset_folder = os.path.join(output_base_folder, subset_name)
    os.makedirs(subset_folder, exist_ok=True)
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(subset_folder, filename)
        shutil.copyfile(file_path, dest_path)

# Kopiowanie plików
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"Podzia. zakooczony:")
print(f"- Train: {len(train_files)} plikow")
print(f"- Validation: {len(val_files)} plikow")
print(f"- Test: {len(test_files)} plikow")

