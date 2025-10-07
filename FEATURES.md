# ear-job-visualizer Features

## Features Overview

### Loop and Application Signatures
- **Loop Signatures**: Generate visualizations for loop-level performance metrics.
- **Application Signatures**: Create application-level performance signatures for detailed analysis.
- **Multi-column Support**
- **Multiple Node support**: Visualize data from multiple nodes in a single image.

### List of metrics
- **Obtain a list of currently supported metrics**: Use the `--avail-metrics` option to see all available metrics for visualization.
  - LIST HERE

### Output Formats
- **Image Outputs**: Save visualizations as PNG, PDF, or other image formats.
- **Paraver Trace Generation**: Export runtime traces for further analysis in external tools. A BIT OF RELATED FEATURES HERE (configuration of paraver, etc.) 

### Configuration Options
- **Print configuration**: Use the `--print-config` option to display the current configuration settings.
- **Customizable Configuration**: Adjust visualization parameters such as color schemes, time ranges, metrics thresholds, and titles.

---

## Commands and Supported Options

### `ear-job-visualizer`
- **`-h, --help`**: Show help message and exit.
- **`--version`**: Show program's version number and exit.

---

### Main Options
- **`-c, --config-file CONFIG_FILE`**: Specify a custom configuration file.
- **`--format {runtime,ear2prv}`**: Build results according to the chosen format:
  - `runtime`: Generate static images.
  - `ear2prv`: Generate Paraver-compatible trace files.
- **`--print-config`**: Print the used configuration file.
- **`--avail-metrics`**: Print the available metrics provided by the configuration file.

---

### Format Common Options
- **`--loops-file LOOPS_FILE`**: Specify the loop input file(s) to read data from.
- **`--apps-file APPS_FILE`**: Specify the app input file(s) to read data from.
- **`-j, --job-id JOB_ID`**: Filter the data by the Job ID.
- **`-s, --step-id STEP_ID`**: Filter the data by the Step ID.
- **`-o, --output OUTPUT`**: Set the output file name or directory.
- **`-k, --keep-csv`**: Keep temporary CSV files.

---

### `runtime` Format Options
- **`-t, --title TITLE`**: Set the resulting figure title.
- **`-r, --manual-range`**: Use the range of values specified in the configuration file for the colormap.
- **`-m, --metrics metric [metric ...]`**: Space-separated list