import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import ast

def create_accuracy_plot(csv_file, title, save_as):
    df = pd.read_csv(csv_file)
    df=df.transpose()
    df.columns = df.iloc[0]
    df = df.drop("Unnamed: 0")
    time_data = list(range(len(df['score'])))
    accuracy_data = df['score'].astype(float).tolist()

    # Extract lower and upper limits for confidence interval (if available)
    lower_limits = df['lower_limit'].astype(float).tolist()
    upper_limits = df['upper_limit'].astype(float).tolist()

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, accuracy_data, marker='o', linestyle='-', label='Accuracy')

    
    # If confidence interval data is available, plot it
    plt.fill_between(time_data, lower_limits, upper_limits, color='gray', alpha=0.2, label='Confidence Interval')

    plt.xlabel("Iteration Number")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Add legend
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_as)

    # Show the plot (optional)
    plt.show()


def create_len_of_policy_plot(csv_file, title, save_as):
    df = pd.read_csv(csv_file)
    df=df.transpose()
    df.columns = df.iloc[0]
    # Drop the first row (optional, if you want to remove the row used as column names)
    df = df.drop("Unnamed: 0")
    # Extract the time and accuracy data
    time_data = list(range(len(df['score'])))
    policy_length_data = df["current_policy"].astype(str).apply(len)

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, policy_length_data, marker='o', linestyle='-')
    plt.xlabel("Iteration Number")
    plt.ylabel("Length of Policy in Characters")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Save the plot to a file
    plt.savefig(save_as)

    # Show the plot (optional)
    plt.show()

def fleiss_visualize(fleiss_file, save_as):

    # Read the CSV file
    df = pd.read_csv(fleiss_file)
    df['Scores'] = df['Scores'].round(3)
    for dataset, subset in df.groupby('dataset'):
        subset = subset.drop_duplicates('idx')
        value_counts = subset['Scores'].value_counts()
        print(f"\nValue counts for {dataset}:")
        print(value_counts)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot KDEs for each dataset
    for dataset, data in df.groupby('dataset'):
        sns.kdeplot(data['Scores'], label=dataset, fill=True)

    # Calculate statistics for each dataset
    result_df = df.groupby('dataset')['Scores'].agg(['mean', 'median', lambda x: x.mode().iloc[0]]).reset_index()
    result_df.columns = ['', 'mean', 'median', 'mode']
    result_df['mean'] = result_df['mean'].round(3)

    # Display the statistics on the plot
    ax.text(0.95, 0.95, result_df, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Customize the plot
    plt.title('Kernel Density Estimate of Scores for Each Dataset')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_as)
    
    # Show the plot
    plt.show()

def visualize_overlap(csv_file, save_as):
    df = pd.read_csv(csv_file)
    df=df.transpose()
    df.columns = df.iloc[0]
    df = df.drop("Unnamed: 0")
    time_data = list(range(len(df['score'])))
    missed_points = [ast.literal_eval(point) if isinstance(point, str) else point for point in df['missed statements'].tolist()] #df['missed statements'].tolist()
    print(missed_points[0])
    # Initialize dictionaries to store results
    overlap_counts = []
    set1_not_set2_counts = []
    set2_not_set1_counts = []

    # Calculate counts for each pair of sets
    for i in range(len(missed_points) - 1):
        set1 = set(missed_points[i])
        set2 = set(missed_points[i + 1])
        print('set1', len(set1), 'expected 40')
        print('set2', len(set2), 'expected 59')
        
        overlap_counts.append(len(set1.intersection(set2)))
        set1_not_set2_counts.append(len(set1.difference(set2)))
        set2_not_set1_counts.append(len(set2.difference(set1)))

    # Plot the results
    x_values = [f"Set{i+1} - Set{i+2}" for i in range(len(missed_points) - 1)]

    plt.plot(x_values, overlap_counts, label='Missed Consecutively', marker='o')
    plt.plot(x_values, set1_not_set2_counts, label='Corrected', marker='o')
    plt.plot(x_values, set2_not_set1_counts, label='Worsened', marker='o')

    plt.title('Shared and Unique Values Between Consecutive Sets')
    plt.xlabel('Set Pairs')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(save_as)
    plt.show()