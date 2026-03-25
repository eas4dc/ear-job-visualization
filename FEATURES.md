### Runtime Visualizations
- **Loop Metrics Heatmaps**: Generate per-node timelines of loop-level metrics (GFLOPS, CPI, %MPI, etc.) as PNG/PDF figures.
- **GPU Metrics Support**: Plot GPU-specific metrics and automatically hide GPUs that stay at zero utilization.
- **Multi-node Alignment**: Display all nodes on a shared timeline, letting you compare execution phases across the cluster.
- **Manual/Auto Ranges**: Pick between automatic min/max scaling or the ranges defined in the configuration with `--manual-range`.

### Paraver Trace Export
- **Complete Metric Export**: Convert loop/app CSV data to a Paraver-compatible trace containing every metric present in the input.
- **Workflow-Friendly**: Include multiple steps/apps of a job in a single trace; only `--job-id` is required for this format.
- **Node/GPU Mapping**: Map node metrics to Paraver tasks and GPU metrics to thread lanes so they are ready for Paraver configs shipped under `examples/`.

### Metric Discovery
- **Discover Available Metrics**: Run `--avail-metrics` (optionally with `-c custom_config.json`) to list every metric name that can be used in `-m`.

### Output Formats
- **Image Outputs**: Save figures as PNG/PDF (default names `runtime_<metric>[ -<suffix>]` or inside the directory passed to `-o`).
- **Paraver Trace Generation**: Emit `.prv/.pcf/.row` files when `--format ear2prv`, ready for Paraver or other BSC Tools.

### Configuration Options
- **Inspect Configuration**: `--print-config` dumps the active JSON configuration.
- **Custom Configuration**: `-c/--config-file` lets you change palettes, metric ranges, column mappings, etc.

---

## Commands and Supported Options

### `ear-job-visualizer`
- **`-h, --help`**: Show help message and exit.
- **`--version`**: Show program’s version number and exit.

---

### Main Options
- **`-c, --config-file CONFIG_FILE`**: Use a custom configuration JSON.
- **`--format {runtime,ear2prv}`**: Choose between static figures (`runtime`) and Paraver traces (`ear2prv`).
- **`--print-config`**: Print the configuration file and exit.
- **`--avail-metrics`**: List metrics defined in the configuration.

`--job-id` is required whenever `--format` is used; `--step-id` and `-m/--metrics` are required only for `runtime`.

---

### Format Common Options
- **`--loops-file LOOPS_FILE`**: Optional; read loop CSVs/dirs instead of letting the tool call `eacct -r`. Must be paired with `--apps-file`.
- **`--apps-file APPS_FILE`**: Optional; read app CSVs/dirs instead of `eacct -l`. Must be paired with `--loops-file`.
- **`-j, --job-id JOB_ID`**: Select the job to visualize (mandatory with `--format`).
- **`-s, --step-id STEP_ID`**: Filter by step; mandatory for `runtime`, optional for `ear2prv`.
- **`-o, --output OUTPUT`**: Output file/directory base name (figures or Paraver trace prefix).
- **`-k, --keep-csv`**: Keep the temporary CSV files when data was fetched through `eacct`.

---

### `runtime` Format Options
- **`-t, --title TITLE`**: Prefix each figure title with `<title>:`.
- **`-r, --manual-range`**: Use the metric ranges defined in the config (instead of auto-scaling).
- **`-m, --metrics metric [metric ...]`**: Space-separated metric names (see `--avail-metrics`).
