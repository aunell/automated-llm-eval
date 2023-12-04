import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from automated_llm_eval.model_analysis import analysis
import pandas as pd
from matplotlib.ticker import MultipleLocator
import seaborn as sns

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def stack_from_df(df):
    df.columns= df.columns.get_level_values(1)
    safety_scores = df.at['Avg Safety Score', 'Samples']  # Values for the first set of data
    ethics_scores = df.at['Avg Ethics Score', 'Samples']  # Values for the second set of data
    clinician_scores = df.at['Avg Clinician Score', 'Samples']  # Values for the third set of data
    stacked_lists = list(zip(safety_scores, ethics_scores, clinician_scores))
    averaged_results = np.mean(stacked_lists, axis=0)
    return averaged_results

def make_data(engine_options, engine_judge_options):
    data = []
    for engine in engine_options:
        for judge in engine_judge_options:
            df = analysis(engine, judge)
            stack = stack_from_df(df)
            data.append((engine+'+'+judge, stack))
    return data

def make_spider_plot(engine_options, engine_judge_options):
    N = 3
    
    theta = radar_factory(N, frame='polygon')

    data = make_data(engine_options, engine_judge_options) #example_data(df, engine, engine_judge)
    spoke_labels = ['Safety', 'Ethics', 'Clinician'] #data.pop(0)
    print(data)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))

    colors = ['b', 'r', 'g', 'm', 'y']
    
    for (title, case_data), color in zip(data, colors):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_varlabels(spoke_labels)
        ax.set_title("Engine and Engine Judge Analysis Version 1", weight='bold', size='large')
        ax.plot(theta, case_data, color=color, label=title)
        ax.fill(theta, case_data, facecolor=color, alpha=0.25)

    ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    plt.show()


def create_bar_plots(values, categories, error, title):
    # Create a bar chart
    plt.bar(categories, values, yerr=error, capsize=5, color='skyblue')

    # Add labels and title
    plt.xlabel('Engine and Engine Judge')
    plt.ylabel('Number')
    plt.title(title)
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.savefig(title+'_bar_plot.png')

    # Show the chart
    plt.tight_layout()
    plt.show()

def create_plots(engine_options, judge_options):
    iteration_number = []
    number_answered= []
    names = []
    error_answered= []
    error_iteration = []
    for engine in engine_options:
        for engine_judge in judge_options:
            df = analysis(engine, engine_judge, engine+ ' + ' +engine_judge)
            df.columns= df.columns.get_level_values(1)
            make_spider_plot(df, engine, engine_judge)
            iteration_number.append(df.at['Iterations', 'Mean'])
            number_answered.append(df.at['Number Answered', 'Mean'])
            error_answered.append(np.std(df.at['Number Answered', 'Samples']))
            error_iteration.append(np.std(df.at['Iterations', 'Samples']))
            names.append(engine+'+'+engine_judge)

    make_spider_plot(engine_options, judge_options)

    create_bar_plots(iteration_number, names, error_iteration, 'iteration_number')
    create_bar_plots(number_answered, names, error_answered, 'number_answered') 

# def create_accuracy_plot(csv_file, title, save_as):
#     df = pd.read_csv(csv_file)

# # Transpose the DataFrame to have time data as rows and values as columns
#     df = df.transpose()
#     start_score = df.iloc[-1, 0]
#     end_score = df.iloc[-1, 1]
#     example_length= df.shape[0]-1
#     # Extract the time and value data, and filter out non-integer time values
#     value_data = df.iloc[:example_length,1].astype(float)
#     time_data = df.index[:example_length]
#     # Filter out non-integer time values
#     time_data = time_data[time_data.str.isnumeric()]
#     time_data = time_data.astype(int)
#     print(df.iloc[:-1, -2])
#     lower_limits = df.iloc[:-1, -2].astype(float)
#     upper_limits = df.iloc[:-1, -1].astype(float)

#     # Create a line plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_data, value_data, marker='o', linestyle='-')
#     plt.fill_between(time_data, lower_limits, upper_limits, color='gray', alpha=0.2, label='Confidence Interval')
#     plt.xlabel("Iteration Number")
#     plt.ylabel("Accuracy")
#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     print(start_score, end_score)
#     legend_text = f"Starting Test Accuracy: {start_score}, Ending Test Accuracy: {end_score}"
#     plt.legend([legend_text])
#     # print('plotting')
#     plt.savefig(save_as)

#     # Show the plot or save it to a file
#     plt.show()

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
    missed_points = df['missed statements'].tolist()
    # Initialize dictionaries to store results
    overlap_counts = []
    set1_not_set2_counts = []
    set2_not_set1_counts = []

    # Calculate counts for each pair of sets
    for i in range(len(missed_points) - 1):
        set1 = set(missed_points[i])
        set2 = set(missed_points[i + 1])
        
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