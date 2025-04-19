import os
import glob
import numpy as np
import librosa
import math
import matplotlib.pyplot as plt

os.chdir(r"D:\repos\DL_Project2")

file_list = glob.glob(r"data_copy\**\*.wav", recursive=True)

file_list.sort()

first_file = file_list[0]


def remove_silence(audio, sr, top_db=60):
    # Wykrycie segmentów z dźwiękiem
    intervals = librosa.effects.split(audio, top_db=top_db)
    # Połączenie wykrytych segmentów
    non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])
    return non_silent_audio



for i in range(len(file_list[:])):  # Since audio length is now 1 second, process in segments of 1 second

    #other scaling
    scale, sr = librosa.load(file_list[:][i],sr=22050)  
    scale_no_silence = remove_silence(scale, sr, top_db=60)
    S = librosa.feature.melspectrogram(y=scale_no_silence, sr=sr, n_mels=256)

    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='gray', vmin=-30, vmax=30)

    #suffix = "_SILENCE" if is_silence else "_spec"
    #output_image_path = file_list[i].replace(".wav", f"{suffix}.png")
    
    plt.axis('off')  # Remove axes
    plt.gca().set_axis_off()  # Remove axis elements
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove margins
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove X axis
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove Y axis
    # Use both the file index and segment index for the filename
    output_image_path = f'{file_list[:][i][:-4]}_reduction_no_silence.png'
    print(output_image_path)
    # Ensure the directory exists
    output_dir = os.path.dirname(output_image_path)
    print(i)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

#     # # Print to debug
    try:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
    except Exception as e:
        print(f"Error saving spectrogram: {e}")
#     # finally:
    plt.close()