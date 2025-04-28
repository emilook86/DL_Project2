import os
import glob
import numpy as np
import librosa
import math
import matplotlib.pyplot as plt
import shutil

os.chdir(r"D:\repos\DL_Project2")

file_list = glob.glob(r"data_copy\**\*.wav", recursive=True)

file_list.sort()

first_file = file_list[0]


def remove_silence(audio, sr, top_db=15):
    # Wykrycie segmentów z dźwiękiem
    intervals = librosa.effects.split(audio, top_db=top_db)
    # Połączenie wykrytych segmentów
    non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])
    return non_silent_audio


# Nowy folder docelowy dla spektrogramów
# Główna ścieżka
base_input_dir = os.path.join("D:\\repos\\DL_Project2", "data_copy")
base_output_dir = os.path.join("D:\\repos\\DL_Project2", "new_spectograms")
silence_dir = os.path.join(base_output_dir, "silence")

for i in range(len(file_list[37087:])):
    scale, sr = librosa.load(file_list[i+37087], sr=22050)  

    scale_no_silence = remove_silence(scale, sr, top_db=15)

    S = librosa.feature.melspectrogram(y=scale_no_silence, sr=sr, n_mels=256)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='gray', vmin=-30, vmax = 30)
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Wyciągnięcie względnej ścieżki i przygotowanie ścieżki wyjściowej
    rel_path = os.path.relpath(file_list[i+37087], base_input_dir)  # np. 'train/yes/abc.wav'
    rel_path_no_ext = os.path.splitext(rel_path)[0]  # np. 'train/yes/abc'
    output_image_path = os.path.join(base_output_dir, rel_path_no_ext + "_reduction_no_silence.png")

    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"{i}: Saving to {output_image_path}")

    try:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
    except Exception as e:
        print(f"Error saving spectrogram for {file_list[i+37087]}: {e}")
    plt.close()

