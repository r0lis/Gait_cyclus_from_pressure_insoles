import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import nanmean

output_image_path = "R_Pas.png"

def read_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.replace({',': '.'}, regex=True).astype(float)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    return df, df_numeric, df_numeric.iloc[:, 0]

def find_crossings_and_selected_pairs(mean_data, threshold):
    above_threshold = mean_data >= threshold
    crossings = np.where(np.diff(above_threshold.astype(int)))[0]

    # Filter crossings with spacing greater than 40
    valid_crossings = [crossings[0]]
    for i in range(1, len(crossings)):
        if crossings[i] - crossings[i - 1] > 35:
            valid_crossings.append(crossings[i])

    crossings = np.array(valid_crossings)

    data_pairs = []
    for i in range(0, len(crossings) - 1):
        start_index = crossings[i]
        end_index = crossings[i + 1]
        data_pair = mean_data[start_index:end_index]
        data_pairs.append(data_pair)

    # Updated code to filter pairs based on the condition (greater than 2)
    selected_pairs = [pair for pair in data_pairs[3:18] if any(item > 2 for item in pair)]

    return crossings, selected_pairs
def plot_averaged_data_with_std(t, interpolation_indices, averaged_data, std_deviation, color, label, max_index, max_value):
    normalized_indices = interpolation_indices / interpolation_indices[-1] * 100
    plt.plot(normalized_indices, averaged_data, label=label, color=color)
    plt.plot(normalized_indices, averaged_data - std_deviation, linestyle='--', color=color, alpha=0.7)
    plt.plot(normalized_indices, averaged_data + std_deviation, linestyle='--', color=color, alpha=0.7)
    plt.scatter(normalized_indices[max_index], max_value, color=color, label=f'Max: {max_value:.2f}')

def plot_graphs(df, t, transformed_array, prah, prah_forefoot, prah_midfoot, prah_heel):
    Toes = transformed_array.iloc[:, 1:22]
    Forefoot = transformed_array.iloc[:, 22:59]
    Midfoot = transformed_array.iloc[:, 59:91]
    Heel = transformed_array.iloc[:, 91:117]

    mean_toes = Toes.mean(axis=1)
    mean_forefoot = Forefoot.mean(axis=1)
    mean_midfoot = Midfoot.mean(axis=1)
    mean_heel = Heel.mean(axis=1)

    crossings_toes, selected_pairs_toes = find_crossings_and_selected_pairs(mean_toes, prah)
    max_length_toes = max(len(pair) for pair in selected_pairs_toes)
    selected_pairs_toes = selected_pairs_toes[:-2]
    padded_pairs_toes = [np.pad(pair, (0, max_length_toes - len(pair)), 'constant', constant_values=np.nan) for pair in
                         selected_pairs_toes]
    interpolation_indices_toes = np.arange(0, max_length_toes)
    interpolated_data_toes = [np.interp(interpolation_indices_toes, np.arange(0, len(pair)), pair) for pair in padded_pairs_toes]
    averaged_data_toes = nanmean(interpolated_data_toes, axis=0)
    std_deviation_toes = np.std(interpolated_data_toes, axis=0)
    max_index_toes = np.argmax(averaged_data_toes)
    max_value_toes = averaged_data_toes[max_index_toes]



    plot_averaged_data_with_std(t, interpolation_indices_toes, averaged_data_toes, std_deviation_toes, 'blue', 'Toes', max_index_toes, max_value_toes)

    crossings_forefoot, selected_pairs_forefoot = find_crossings_and_selected_pairs(mean_forefoot, prah_forefoot)
    max_length_forefoot = max(len(pair) for pair in selected_pairs_forefoot)
    padded_pairs_forefoot = [np.pad(pair, (0, max_length_forefoot - len(pair)), 'constant', constant_values=np.nan) for pair in selected_pairs_forefoot]
    interpolation_indices_forefoot = np.arange(0, max_length_forefoot)
    interpolated_data_forefoot = [np.interp(interpolation_indices_forefoot, np.arange(0, len(pair)), pair) for pair in padded_pairs_forefoot]
    averaged_data_forefoot = nanmean(interpolated_data_forefoot, axis=0)
    std_deviation_forefoot = np.std(interpolated_data_forefoot, axis=0)
    max_index_forefoot = np.argmax(averaged_data_forefoot)
    max_value_forefoot = averaged_data_forefoot[max_index_forefoot]

    plot_averaged_data_with_std(t, interpolation_indices_forefoot, averaged_data_forefoot, std_deviation_forefoot, 'green', 'Forefoot', max_index_forefoot, max_value_forefoot)

    crossings_midfoot, selected_pairs_midfoot = find_crossings_and_selected_pairs(mean_midfoot, prah_midfoot)
    max_length_midfoot = max(len(pair) for pair in selected_pairs_midfoot)
    max_length_midfoot = min(max_length_midfoot, 40)
    padded_pairs_midfoot = [np.pad(pair[:40], (0, 40 - len(pair[:40])), 'constant', constant_values=np.nan) for pair in
                         selected_pairs_midfoot]
    interpolation_indices_midfoot = np.arange(0, max_length_midfoot)
    interpolated_data_midfoot = [np.interp(interpolation_indices_midfoot, np.arange(0, len(pair)), pair) for pair in padded_pairs_midfoot]
    averaged_data_midfoot = nanmean(interpolated_data_midfoot, axis=0)
    std_deviation_midfoot = np.std(interpolated_data_midfoot, axis=0)
    max_index_midfoot = np.argmax(averaged_data_midfoot)
    max_value_midfoot = averaged_data_midfoot[max_index_midfoot]

    plot_averaged_data_with_std(t, interpolation_indices_midfoot, averaged_data_midfoot, std_deviation_midfoot, 'yellow', 'Midfoot', max_index_midfoot, max_value_midfoot)

    crossings_heel, selected_pairs_heel = find_crossings_and_selected_pairs(mean_heel, prah_heel)
    max_length_heel = max(len(pair) for pair in selected_pairs_heel)
    max_length_heel = min(max_length_heel, 30)
    padded_pairs_heel = [np.pad(pair[:30], (0, 30 - len(pair[:30])), 'constant', constant_values=np.nan) for pair in selected_pairs_heel]
    interpolation_indices_heel = np.arange(0, max_length_heel)
    interpolated_data_heel = [np.interp(interpolation_indices_heel, np.arange(0, len(pair)), pair) for pair in padded_pairs_heel]
    averaged_data_heel = nanmean(interpolated_data_heel, axis=0)
    std_deviation_heel = np.std(interpolated_data_heel, axis=0)
    max_index_heel = np.argmax(averaged_data_heel)
    max_value_heel = averaged_data_heel[max_index_heel]

    plot_averaged_data_with_std(t, interpolation_indices_heel, averaged_data_heel, std_deviation_heel, 'red', 'Heel', max_index_heel, max_value_heel)

    plt.xlabel('Stance phase[%]')
    plt.ylabel('Pressure [N/cm^2]')
    plt.title('Walking treadmill R')
    plt.legend()
    plt.savefig(output_image_path)

    plt.show()

# File path
file_path = "Shanelova_pas_Rt.csv"

# Thresholds
prah = 0.15
prah_forefoot = 0.15
prah_midfoot = 0.15
prah_heel = 0.15

# Read and preprocess data
df, transformed_array, t = read_and_preprocess_data(file_path)

# Plot graphs
plot_graphs(df, t, transformed_array, prah, prah_forefoot, prah_midfoot, prah_heel)

