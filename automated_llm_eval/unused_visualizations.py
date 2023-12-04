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
