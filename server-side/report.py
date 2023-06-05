import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def make_report(confidence_csv):
    # Create a folder for the images if it doesn't exist
    folder_name = 'temp_files/report_outputs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Load data
    df_confidence = pd.read_csv(confidence_csv)
    df_labels_list = df_confidence['label']
    df_length = len(df_confidence)

    # Select columns for analysis
    selected_columns = ['happy', 'sad', 'shy', 'stressed', 'angry', 'surprised', 'bored', 'disgust']

    data = pd.read_csv(confidence_csv)
    selected_data = data[selected_columns]

    # Set the values of zero to NaN for selected_data
    selected_data[selected_data == 0] = float('nan')

    # Define label colors
    label_colors = {
        'happy': 'yellow',
        'sad': 'blue',
        'angry': 'red',
        'stressed': 'indigo',
        'disgust': '#66FF66',
        'shy': 'teal',
        'bored': 'magenta',
        'surprised': 'orange'
    }

    ##### 1st graph #####

    # Plot graph of column values
    fig, ax = plt.subplots(figsize=(len(data) * 0.1, 5))
    selected_data.plot(kind='line', marker='o', alpha=0.7, ax=ax, color=[
        label_colors[column] for column in selected_data.columns])

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Graph of Column Values')

    # plt.show()

    # Save the figure
    fig.savefig(os.path.join(folder_name, 'graph1.png'))

    plt.close(fig)  # Close the figure to release memory

    ##### 2nd graph #####

    # Replace values
    replaced_strings = [
        string for string in df_labels_list if string in selected_columns and string != 'None']

    # Create a list of unique labels
    unique_labels = list(set(replaced_strings))

    # Assign a color to each label
    label_colors = [label_colors[label] for label in unique_labels]

    # Calculate the height of each column
    heights = np.ones(len(replaced_strings))

    # Plot the replaced strings
    fig, ax = plt.subplots(figsize=(len(data['time']) * 0.1, 3))
    bars = ax.bar(data['time'], heights, color=[
        label_colors[unique_labels.index(label)] for label in replaced_strings])

    # Remove y-axis and tick labels
    ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])

    # Add a legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=label_colors[i])
                      for i in range(len(unique_labels))]
    ax.legend(legend_handles, unique_labels)

    # Set the color cycle for the second graph
    ax.set_prop_cycle(color=label_colors)

    # plt.show()

    # Save the figure
    fig.savefig(os.path.join(folder_name, 'graph2.png'))

    plt.close(fig)  # Close the figure to release memory

    ##### 3rd graph #####

    # Calculate the percentage of time each label has been shown
    label_percentages = [replaced_strings.count(
        label) / df_length * 100 for label in unique_labels]

    # Plot the percentage graph
    fig, ax = plt.subplots(figsize=(max(len(unique_labels) * 1, 5), 4))
    ax.bar(unique_labels, label_percentages, color=label_colors)

    # Add labels to the bars
    for i, label in enumerate(unique_labels):
        ax.text(i, label_percentages[i] + 1,
                f"{label_percentages[i]:.1f}%", ha='center')

    plt.xlabel('Label')
    plt.ylabel('Percentage')
    plt.title('Percentage of Time Each Label Has Been Shown')
    # plt.show()

    # Save the figure
    fig.savefig(os.path.join(folder_name, 'graph3.png'))

    plt.close(fig)  # Close the figure to release memory