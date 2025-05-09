#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:54:54 2025

@author: sylvainbertaina
"""

"""
Module for specialized plotting of EPR data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # For colormaps

def plot_angular_sweep_waterfall(
    field_axis,
    angle_axis,
    spectral_data_2d,
    field_unit='G',
    angle_unit='deg',
    intensity_label='Intensity (a.u.)',
    offset_factor=0.2,
    integrate=False,
    ax=None,
    title="Angular Sweep Waterfall Plot",
    cmap_name='viridis',
    line_label_prefix='Angle:'
):
    """
    Creates a waterfall plot for angular sweep data.

    Each angle's spectrum is plotted as a line, offset vertically.
    Lines are colored based on their angle value using a colormap.

    Args:
        field_axis (np.ndarray): 1D array for the primary sweep axis (e.g., magnetic field).
        angle_axis (np.ndarray): 1D array for the angular axis.
        spectral_data_2d (np.ndarray): 2D array of spectral intensities.
                                       Expected shape: (len(angle_axis), len(field_axis)).
        field_unit (str, optional): Unit for the field axis. Defaults to 'G'.
        angle_unit (str, optional): Unit for the angle axis. Defaults to 'deg'.
        intensity_label (str, optional): Label for the y-axis. Defaults to 'Intensity (a.u.)'.
        offset_factor (float, optional): Factor to determine the vertical offset
                                         between spectra. Offset is calculated as
                                         offset_factor * data_range_over_all_spectra.
                                         Defaults to 0.2.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on.
                                             If None, a new figure and axes are created.
                                             Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Angular Sweep Waterfall Plot".
        cmap_name (str, optional): Name of the matplotlib colormap to use for coloring lines
                                   by angle. Defaults to 'viridis'.
        line_label_prefix (str, optional): Prefix for individual line labels in legend if generated.
                                        Defaults to 'Angle:'. If empty, no individual line labels.
    """
    if spectral_data_2d.shape != (len(angle_axis), len(field_axis)):
        raise ValueError(
            f"Shape mismatch: spectral_data_2d shape {spectral_data_2d.shape} "
            f"does not match (len(angle_axis)={len(angle_axis)}, len(field_axis)={len(field_axis)})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    num_angles = len(angle_axis)
    cmap = cm.get_cmap(cmap_name, num_angles)

    # Calculate overall data range for sensible offsetting
    data_min = np.nanmin(spectral_data_2d)
    data_max = np.nanmax(spectral_data_2d)
    data_range = data_max - data_min
    if data_range == 0: # Handle flat data
        data_range = 1.0 # Avoid division by zero or zero offset for non-zero data
    vertical_offset_per_spectrum = offset_factor * data_range

    for i in range(num_angles):
        spectrum = spectral_data_2d[i, :]
        offset = i * vertical_offset_per_spectrum
        color = cmap(i / (num_angles -1 if num_angles > 1 else 1) ) # Normalize index for colormap

        label = None
        if line_label_prefix:
            label = f"{line_label_prefix} {angle_axis[i]:.1f} {angle_unit}"
            
        if integrate==True:
            ax.plot(field_axis, np.cumsum(spectrum) + offset, color=color, label=label)
        else:
            ax.plot(field_axis, spectrum + offset, color=color, label=label)
                

    ax.set_xlabel(f"Magnetic Field ({field_unit})")
    ax.set_ylabel(intensity_label)
    ax.set_title(title)

    # Improve y-axis ticks and labels if spectra are offset significantly
    # This makes the y-axis less cluttered by default, showing relative intensity.
    # To show actual offset values, one might need a different approach or twin axis.
    ax.set_yticks([]) # Remove y-ticks as they represent offset + intensity
    # ax.set_yticklabels([]) # This line is redundant if set_yticks([]) is used.

    # Optionally add a colorbar to indicate angles, if line labels are not preferred
    # This can be tricky for line plots without a mappable artist.
    # A separate legend for colors might be better or explicit line labels.
    # For now, rely on line_label_prefix for legend generation.
    if line_label_prefix and num_angles <= 10: # Show legend if few lines
        ax.legend(loc='best', fontsize='small')
    elif line_label_prefix and num_angles > 10:
        # For many lines, a legend is too crowded. Consider a colorbar.
        # Creating a scalar mappable for the colorbar:
        norm = plt.Normalize(vmin=angle_axis.min(), vmax=angle_axis.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # Dummy array
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=f"Angle ({angle_unit})")


    ax.set_xlim(field_axis.min(), field_axis.max())
    # Adjust y_lim to show all offset spectra
    min_y_plot = np.nanmin(spectral_data_2d[0,:]) # First spectrum baseline
    max_y_plot = np.nanmax(spectral_data_2d[-1,:]) + (num_angles -1) * vertical_offset_per_spectrum # Last spectrum top
    # Add some padding
    plot_yrange = max_y_plot - min_y_plot
    if plot_yrange == 0: plot_yrange = 1
    ax.set_ylim(min_y_plot - 0.1 * plot_yrange, max_y_plot + 0.1 * plot_yrange)

    plt.tight_layout()
    return fig, ax



def plot_2d_map(
    x_axis_data,
    y_axis_data,
    spectral_data_2d,
    x_unit='',
    y_unit='',
    intensity_label='Intensity (a.u.)',
    ax=None,
    title="2D Spectral Map",
    cmap_name='viridis',
    shading='auto'
):
    """
    Creates a 2D color map (pcolormesh) of spectral data.

    Args:
        x_axis_data (np.ndarray): 1D array for the x-axis.
        y_axis_data (np.ndarray): 1D array for the y-axis.
        spectral_data_2d (np.ndarray): 2D array of spectral intensities.
                                       Expected shape: (len(y_axis_data), len(x_axis_data)).
        x_unit (str, optional): Unit for the x-axis. Defaults to ''.
        y_unit (str, optional): Unit for the y-axis. Defaults to ''.
        intensity_label (str, optional): Label for the colorbar. Defaults to 'Intensity (a.u.)'.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on.
                                             If None, a new figure and axes are created.
                                             Defaults to None.
        title (str, optional): Title of the plot. Defaults to "2D Spectral Map".
        cmap_name (str, optional): Name of the matplotlib colormap. Defaults to 'viridis'.
        shading (str, optional): Shading option for pcolormesh ('auto', 'flat', 'gouraud').
                                 Defaults to 'auto'.
    """
    if spectral_data_2d.shape != (len(y_axis_data), len(x_axis_data)):
        raise ValueError(
            f"Shape mismatch: spectral_data_2d shape {spectral_data_2d.shape} "
            f"does not match (len(y_axis_data)={len(y_axis_data)}, len(x_axis_data)={len(x_axis_data)})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # pcolormesh typically wants X, Y to define the corners of the cells.
    # If x_axis_data and y_axis_data define cell centers, one might need to adjust.
    # For now, assume they define the coordinates for the values in spectral_data_2d.
    im = ax.pcolormesh(
        x_axis_data,
        y_axis_data,
        spectral_data_2d,
        cmap=cmap_name,
        shading=shading
    )

    cbar = fig.colorbar(im, ax=ax, label=intensity_label)

    ax.set_xlabel(f"Abscissa 1 ({x_unit})" if x_unit else "Abscissa 1")
    ax.set_ylabel(f"Abscissa 2 ({y_unit})" if y_unit else "Abscissa 2")
    ax.set_title(title)

    ax.set_xlim(x_axis_data.min(), x_axis_data.max())
    ax.set_ylim(y_axis_data.min(), y_axis_data.max())

    plt.tight_layout()
    return fig, ax