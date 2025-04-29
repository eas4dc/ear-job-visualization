# %%
import os
import time
from datetime import datetime


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# %%

# Default configuration
default_config = {
    "colormap": {
        "v_min": -1,
        "v_max": 1,
        "step": 0.1,
        "colormap": "viridis_r",
    },
    "figure": {
        "nrows": 1,
        "interactive": False,
        "title": ["Test Title 1", None],
        "ylabel": ["Test Label 1", "Test Label 2"],
    },
    "monitoring": {
        "interval": 5,
        "verbose": 0,
    },
}

def check_config(config):
    """Check if the configuration is valid"""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    # Check colormap settings
    colormap = config.get("colormap", {})
    if not isinstance(colormap, dict):
        raise ValueError("Colormap settings must be a dictionary.")
    if "v_min" not in colormap or "v_max" not in colormap:
        raise ValueError("Colormap must contain 'v_min' and 'v_max' keys.")
    if "step" not in colormap:
        raise ValueError("Colormap must contain 'step' key.")
    if "colormap" not in colormap:
        raise ValueError("Colormap must contain 'colormap' key.")

    # Check figure settings
    figure = config.get("figure", {})
    if not isinstance(figure, dict):
        raise ValueError("Figure settings must be a dictionary.")
    if "nrows" not in figure or not isinstance(figure["nrows"], int):
        raise ValueError("Figure must contain 'nrows' key.")
    if "interactive" not in figure or not isinstance(figure["interactive"], bool):
        raise ValueError("Figure must contain 'interactive' key set to true or false.")
    if "title" not in figure or not isinstance(figure["title"], list):
        raise ValueError("Figure config must contain a 'title' key. The value should be a list containing either strings or None.")
    if "ylabel" not in figure or not isinstance(figure["ylabel"], list):
        raise ValueError("Figure config must contain a 'ylabel' key. The value should be a list containing either strings or None.")

    # Check monitoring settings
    monitoring = config.get("monitoring", {})
    if not isinstance(monitoring, dict):
        raise ValueError("Monitoring settings must be a dictionary.")
    if "interval" not in monitoring:
        raise ValueError("Monitoring must contain 'interval' key.")
    if "verbose" not in monitoring:
        raise ValueError("Monitoring must contain 'verbose' key.")

def read_and_process_data(filepath):
    """Read data from file and extract x and y values"""
    data = np.loadtxt(
        filepath, skiprows=1
    )  # Replace with your actual data reading function
    return data[:, 0], data[:, 1]


def setup_color_norm(ydata, v_min=-1, v_max=1, step=0.1, colormap="viridis_r"):
    """Set up colormap and normalization for the plot"""
    bounds = np.arange(
        v_min if v_min is not None else np.nanmin(ydata),
        v_max + step if v_max is not None else np.nanmax(ydata) + step,
        step,
    )
    cmap = mpl.colormaps[colormap]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="both")
    return cmap, norm, bounds


def calculate_plot_size(nrows=1):

    aspect_ratio = (
        10  # x-axis is 10 times longer than y-axis (defined in plot_colored_bars)
    )
    height_cb = 1.5

    # Calculate the width and height of the figure
    width = 8
    height = (width / aspect_ratio + height_cb) * nrows

    return width, height


def create_figure_with_grid(nrows=1):
    """Create matplotlib figure with grid layout"""

    width, height = calculate_plot_size(nrows)
    fig = plt.figure(figsize=(width, height))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, 1),
        axes_pad=0,
        label_mode="L",
        cbar_mode="single",
        cbar_location="bottom",
        cbar_size="20%",
        cbar_pad=0.5,
    )

    return fig, grid


def format_axes(ax, title, ylabel_text):
    """Configure axes appearance and settings"""

    ax.set_aspect(1)
    ax.grid(axis="x", alpha=0.5)
    ax.set_title(title)
    ax.set_yticks([0], labels=ylabel_text)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.minorticks_on()


def plot_colored_bars(ax, ydata, cmap, norm):
    """Plot the bar chart with color-mapped bars"""
    bar = ax.bar(
        range(len(ydata)),
        len(ydata) / 10,
        width=1,
        color=[cmap(norm(x)) if not np.isnan(x) else "white" for x in ydata],
    )
    return bar


def add_colorbar_to_grid(grid, cmap, norm, label="Metric", cbar=None):
    """
    Add a colorbar to the grid layout.

    Parameters
    ----------
    grid : ImageGrid
        The grid layout to which the colorbar will be added.
    cmap : matplotlib.colors.Colormap
        The colormap to use for the colorbar.
    norm : matplotlib.colors.Normalize
        The normalization used for the colorbar.
    label : str, optional
        The label for the colorbar. Default is "Metric".
    cbar : matplotlib.colorbar.Colorbar, optional
        An existing colorbar to update. If None, a new colorbar is created.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar object.
    """
    if cbar is None:
        cbar = grid.cbar_axes[0].colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            label=label,
            format=None,
        )
    else:
        cbar.update_normal(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        cbar.set_label(label)

    return cbar


def create_metric_plot(ydata, config=None, fig=None, grid=None, cbar=None):
    """
    Create a metric plot with color-mapped bars and a colorbar.

    Parameters
    ----------
    ydata : array-like
        The data to be plotted on the y-axis.
    config : dict, optional
        A dictionary containing configuration for colormap and figure settings.
        If None, default settings are used.
    fig : matplotlib.figure.Figure, optional
        An existing figure to reuse. If None, a new figure is created.
    grid : ImageGrid, optional
        An existing grid layout to reuse. If None, a new grid is created.
    cbar : matplotlib.colorbar.Colorbar, optional
        An existing colorbar to reuse. If None, a new colorbar is created.

    Returns
    -------
    tuple
        A tuple containing the figure, grid, and colorbar objects.
    """
    # Use default configuration if none is provided
    if config is None:
        config = default_config
    
    # Check the configuration for validity
    check_config(config)

    # Extract colormap and figure settings from the config
    nrows = config["figure"]["nrows"]
    interactive = config["figure"]["interactive"]
    
    # Create or reuse the figure and grid
    if fig is None or not plt.fignum_exists(fig.number):
        fig, grid = create_figure_with_grid(nrows=nrows)
    else:
        _ = [grid[i].clear() for i in range(nrows)]  # Clear previous content

    # Set up colormap and normalization
    cm_config = config["colormap"]
    cmap, norm, _ = setup_color_norm(
        ydata,
        v_min=cm_config["v_min"],
        v_max=cm_config["v_max"],
        step=cm_config["step"],
        colormap=cm_config["colormap"],
    )

    # Plot data and format axes
    title_lst = ["Test Title", None]
    ylabel_lst = ["Test Label 1", "Test Label 2"]
    for i in range(nrows):
        plot_colored_bars(grid[i], ydata, cmap, norm)
        format_axes(grid[i], title_lst[i], [ylabel_lst[i]])

    # Add or update the colorbar
    cbar = add_colorbar_to_grid(grid, cmap, norm, cbar=cbar)

    # Handle interactive mode
    if interactive:
        fig.canvas.draw_idle()
        plt.pause(0.1)

    return fig, grid, cbar


def monitor_and_replot(file_path, config=None):
    """
    Monitor a file and replot at regular intervals.

    Parameters
    ----------
    file_path : str
        The path to the file to monitor.
    config : dict, optional
        A dictionary containing configuration for monitoring, colormap, and figure settings.
        If None, default settings are used.

    Returns
    -------
    None
    """
    # Use default configuration if none is provided
    if config is None:
        config = default_config

    # Extract monitoring settings from the config
    interval = config["monitoring"]["interval"]
    verbose = config["monitoring"]["verbose"]

    # Extract figure settings

    last_modified = 0
    last_data_length = 0
    stop_flag = {"stop": False}

    # Initialize figure and grid of axes
    fig = None
    grid = None
    cbar = None

    def on_close(event):
        # Callback to set the stop flag when the window is closed
        stop_flag["stop"] = True

    try:
        while True:
            if stop_flag["stop"]:
                print("Stopping monitoring.")
                break

            current_modified = os.path.getmtime(file_path)
            xdata, ydata = read_and_process_data(file_path)

            # Only replot if file was modified or data length changed
            if current_modified != last_modified or len(xdata) != last_data_length:
                if verbose > 0:
                    print(
                        f"File changed. Replotting at {datetime.now().strftime('%H:%M:%S')}"
                    )

                fig, grid, cbar = create_metric_plot(
                    ydata, config=config, fig=fig, grid=grid, cbar=cbar
                )

                if fig is not None:
                    fig.canvas.mpl_connect("close_event", on_close)

                last_modified = current_modified
                last_data_length = len(xdata)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


def main(file_data_path, configuration):
    
    plt.ion() # Enable interactive mode, which allows for dynamic updates
    monitor_and_replot(file_data_path, configuration)


# %%

if __name__ == "__main__":
    # Set matplotlib backend to avoid GUI issues
    # mpl.use("gtk3agg")  # or "TkAgg", "Agg", etc. depending on your environment
    
    file_data_path = "fake_metric_data.txt"  # Path to your data file

    configuration = {
        "colormap": {
            "v_min": -1,
            "v_max": 1,
            "step": 0.1,
            "colormap": "viridis_r",
        },
        "figure": {
            "nrows": 2,
            "interactive": True,
            "title": ["Test Title 1", None],
            "ylabel": ["Test Label 1", "Test Label 2"],
        },
        "monitoring": {
            "interval": 0.5, # Interval in seconds
            "verbose": 0,
        },
    }

    main(file_data_path, configuration)

# %%
