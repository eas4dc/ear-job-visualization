# Interactive Plotting and Data Generator

This repository contains two Python scripts that work together to generate and visualize data in real-time:

1. `data_generator.py` - Generates synthetic sine wave data  
2. `interactive_plotting_demo.py` - Creates real-time visualizations of the generated data

## Table of Contents

- [1. Data Generator](#1-data-generator)  
  - [Overview](#overview)  
  - [Features](#features)  
  - [How It Works](#how-it-works)  
  - [Usage](#usage)  
- [2. Interactive Plotting Demo](#2-interactive-plotting-demo)  
  - [Overview](#overview-1)  
  - [Features](#features-1)  
  - [How It Works](#how-it-works-1)  
  - [Usage](#usage-1)  
- [Using Both Scripts Together](#using-both-scripts-together)  
- [Requirements](#requirements)  
- [Notes](#notes)  
- [License](#license)  

---

## 1. Data Generator

### Overview

`data_generator.py` creates a periodic sine wave signal and continuously writes it to `fake_metric_data.txt`, simulating a real-time data stream.

### Features

- Configurable sine wave parameters  
- Continuous file updates with automatic reset  
- Adjustable output frequency  
- Verbose/quiet modes  

### How It Works

1. **Signal Generation**  
   - Creates sine wave data points based on:
     - Total points  
     - Number of cycles  
     - Time interval between points  

2. **File Management**  
   - Writes data in `t signal` format  
   - Automatically resets after reaching max points  
   - Updates at specified intervals  

### Usage

```bash
python data_generator.py
```

Configuration parameters in the script:

```python
data_points = 100            # Points before reset  
cycles = 5                   # Sine wave cycles  
interval_seconds = 0.1       # Update interval  
out_file = "fake_metric_data.txt"  # Output file  
verbosity = 1                # 1=verbose, 0=quiet  
```

---

## 2. Interactive Plotting Demo

### Overview

`interactive_plotting_demo.py` reads and visualizes the generated data in real-time using Matplotlib.

### Features

- Real-time data monitoring  
- Interactive matplotlib visualization  
- Customizable plot appearance  
- Adjustable refresh rate  

### How It Works

**File Monitoring**  
- Watches `fake_metric_data.txt` for changes  
- Updates plot when file changes  

**Visualization**  
- Creates bar chart with color mapping  
- Uses matplotlib's interactive mode  
- Supports multiple subplots stacked in one column
- Interactive visualization can be stopped with the keyboard directly on the terminal where the script is running or closing the visualization window.

### Usage

```bash
python interactive_plotting_demo.py
```

Default configuration:

```python
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
        "interval": 0.5,  # Refresh interval in seconds
        "verbose": 0,     # Verbosity level
    },
}
```

---

## Using Both Scripts Together

1. Open a terminal and run the data generator:

    ```bash
    python data_generator.py
    ```

2. Open another terminal and run the plotting script:

    ```bash
    python interactive_plotting_demo.py
    ```

3. Observe:
   - The generator continuously updates the data file  
   - The plotter automatically refreshes the visualization  

---

## Requirements

- Python 3.9+

Required packages:

```bash
pip install numpy matplotlib
```

---

## Notes

- Keep both scripts in the same directory  
- Adjust file paths in the code if needed  
- For Jupyter notebooks, use `%matplotlib qt` for interactive plots  

---

## License

MIT License
