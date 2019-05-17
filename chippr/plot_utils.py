import numpy as np

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import chippr
from chippr import defaults as d

# cmap = np.linspace(0., 1., d.plot_colors)
# colors = [cm.viridis(i) for i in cmap]
colors = np.array([(213, 94, 0), (0, 158, 115), (0, 114, 178), (86, 180, 233), (230, 159, 0), (204, 121, 167), (240, 228, 66)])/256.
# ['Vermilion', 'Bluish green', 'Blue', 'Sky blue', 'Orange', 'Reddish purple', 'Yellow']

def set_up_plot():
    """
    Sets up plots to look decent
    """
    title = 18
    label = 16
    mpl.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.titlesize'] = title
    mpl.rcParams['axes.labelsize'] = label
    mpl.pyplot.rcParams['xtick.labelsize'] = label
    mpl.pyplot.rcParams['ytick.labelsize'] = label
    mpl.rcParams['figure.subplot.left'] = 0.2
    mpl.rcParams['figure.subplot.right'] = 0.9
    mpl.rcParams['figure.subplot.bottom'] = 0.2
    mpl.rcParams['figure.subplot.top'] = 0.9
    mpl.rcParams['figure.subplot.wspace'] = 0.5
    mpl.rcParams['figure.subplot.hspace'] = 0.5

    return

def plot_step(sub_plot, bin_ends, to_plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Plots a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    to_plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form
        [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized

    Notes
    -----
    Make this not need a subplot
    """

    plot_h(sub_plot, bin_ends, to_plot, s, c, a, w, d, l, r)
    plot_v(sub_plot, bin_ends, to_plot, s, c, a, w, d, r)

    return

def plot_h(sub_plot, bin_ends, to_plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Helper function to plot horizontal lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    to_plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form
        [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.hlines(to_plot,
                   bin_ends[:-1],
                   bin_ends[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=r)

    return

def plot_v(sub_plot, bin_ends, to_plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], r=False):
    """
    Helper function to plot vertical lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    to_plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form
        [(start_point, (points_on, points_off, ...))]
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.vlines(bin_ends[1:-1],
                   to_plot[:-1],
                   to_plot[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   rasterized=r)

    return
