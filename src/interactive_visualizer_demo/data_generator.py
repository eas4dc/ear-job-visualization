import time
import numpy as np

def main(npoints, cycles, wait_seconds, out_file, verbosity=0):
    """Print data points at fixed time intervals"""

    if verbosity > 0:
        print(f"Generating periodic signal {out_file}")
        print("Each point contains: x_value y_value")
        print(f"File will reset after {npoints} points ({wait_seconds*npoints} seconds)")
        print("Press Ctrl+C to stop...")

    # Check if output file already exists and remove it
    try:
        with open(out_file, 'r') as f:
            pass
    except FileNotFoundError:
        pass
    else:
        if verbosity > 0:
            print(f"Removing existing file: {out_file}")
        import os
        os.remove(out_file)

    # Set initial value and counter
    n = 1

    t_val = np.linspace(0, wait_seconds * npoints, npoints)  # Time values
    period = wait_seconds * npoints  / cycles # Period of the sine wave
    freq = 2 * np.pi / period # Frequency of the sine wave
    signal = np.sin(freq * t_val)  # Sine wave values
    data = np.column_stack((t_val, signal))  # Combine time and signal into a single array

    try:
        while True:

            # Generate data
            np.savetxt(out_file, data[:n], fmt='%f', delimiter=' ', header='t signal', comments='')
            
            # Update counter
            n += 1

            # Reset the file if it exceeds the limit
            if n > npoints:
                n = 1
                if verbosity > 0:
                    print(f"File reset after reaching {npoints} points")

            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        if verbosity > 0:
            print("\nData generation stopped by user.")


if __name__ == "__main__":
    # Configuration
    data_points = 1000  # Number of points in each dimension
    cycles = 10 # Number of cycles for the sine wave
    interval_seconds = 0.1  # Seconds between updates
    out_file = "fake_metric_data.txt"  # Output file path
    verbosity = 1  # Default to verbose output (1=verbose, 0=quiet)
    
    main(data_points, cycles, interval_seconds, out_file, verbosity=verbosity)