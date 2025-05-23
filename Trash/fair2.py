# fair_converter.py
import sys
import os
import json
import csv
from pathlib import Path
import numpy as np
import warnings
import h5py  # Required for HDF5 output: pip install h5py
from typing import Dict, Any, Tuple, Optional, Union, List

# Assuming eprload is in the same directory or installable
try:
    from eprload import eprload
except ImportError:
    print("Error: Could not import 'eprload'. Make sure eprload.py is in the same directory or accessible in the Python path.", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions for Saving ---

def _save_to_csv_json(
    output_basename: Path,
    x: Union[np.ndarray, List[np.ndarray], None],
    y: np.ndarray,
    pars: Dict[str, Any],
    original_file_path: str
) -> None:
    """Saves data to CSV and metadata to JSON."""
    json_file = output_basename.with_suffix(".json")
    csv_file = output_basename.with_suffix(".csv")

    print(f"  Saving metadata to: {json_file}")
    # --- Save Metadata to JSON ---
    metadata_to_save = {
        "original_file": original_file_path,
        "parameters": pars,
         # Add more top-level FAIR metadata if desired, e.g.,
         # "creation_date": datetime.now().isoformat(),
         # "converter_version": "1.0",
    }
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            # Use default=str to handle potential non-serializable types gracefully
            json.dump(metadata_to_save, f, indent=4, default=str)
    except IOError as e:
        warnings.warn(f"Could not write JSON file {json_file}: {e}")
        return # Don't proceed to CSV if JSON fails? Or should we? Let's proceed.
    except TypeError as e:
        warnings.warn(f"Error serializing metadata to JSON for {json_file}: {e}. Some parameters might not be saved correctly.")
        # Continue saving potentially incomplete JSON


    print(f"  Saving data to: {csv_file}")
    # --- Save Data to CSV ---
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # --- Write Header ---
            writer.writerow(['# EPR Data Export'])
            writer.writerow(['# Original File:', original_file_path])
            # Add a few key parameters to header for quick context
            writer.writerow(['# MW_Frequency_(GHz):', pars.get('MWFQ', 'N/A')])
            writer.writerow(['# Center_Field_(G):', pars.get('HCF', pars.get('GST', 'N/A'))])
            writer.writerow(['# Sweep_Width_(G):', pars.get('HSW', pars.get('GSI', 'N/A'))])
            writer.writerow(['# Data_Shape:', str(y.shape)])
            writer.writerow(['# Data_Type:', str(y.dtype)])
            writer.writerow(['# ---'])

            # --- Prepare Data Columns ---
            header_row = []
            data_columns = []
            is_complex = np.iscomplexobj(y)
            is_2d = y.ndim == 2

            # Get Abscissa Units if available (similar logic to _plot_data)
            x_units = pars.get('_Abscissa_Units', [])
            if isinstance(x_units, str): # Ensure it's a list
                 x_units = [x_units]
            x_unit = x_units[0] if len(x_units) > 0 else 'a.u.'
            y_unit = x_units[1] if len(x_units) > 1 else 'a.u.'


            if not is_2d: # 1D Data
                n_pts = y.shape[0]
                # Abscissa Column
                if x is not None and isinstance(x, np.ndarray) and x.shape == y.shape:
                    header_row.append(f"Abscissa ({x_unit})")
                    data_columns.append(x)
                else:
                    header_row.append("Index")
                    data_columns.append(np.arange(n_pts))
                    if x is not None: warnings.warn("Provided x-axis ignored for CSV (shape mismatch or not ndarray). Using index.")

                # Intensity Column(s)
                if is_complex:
                    header_row.extend(["Intensity_Real (a.u.)", "Intensity_Imag (a.u.)"])
                    data_columns.append(np.real(y))
                    data_columns.append(np.imag(y))
                else:
                    header_row.append("Intensity (a.u.)")
                    data_columns.append(y)

            else: # 2D Data - use "long" format (X, Y, Value(s))
                ny, nx = y.shape
                x_coords_flat = np.arange(nx) # Default: Index
                y_coords_flat = np.arange(ny) # Default: Index
                header_row.extend([f"X_Index ({nx} points)", f"Y_Index ({ny} points)"])

                # Determine X and Y axes from input 'x'
                if isinstance(x, list) and len(x) >= 2:
                    x_axis, y_axis = x[0], x[1]
                    if isinstance(x_axis, np.ndarray) and x_axis.size == nx:
                        x_coords_flat = x_axis
                        header_row[0] = f"X_Axis ({x_unit})"
                    if isinstance(y_axis, np.ndarray) and y_axis.size == ny:
                        y_coords_flat = y_axis
                        header_row[1] = f"Y_Axis ({y_unit})"
                elif isinstance(x, np.ndarray) and x.ndim == 1 and x.size == nx:
                     # Only x-axis provided
                     x_coords_flat = x
                     header_row[0] = f"X_Axis ({x_unit})"
                     # Y remains index

                # Create grid and flatten
                xx, yy = np.meshgrid(x_coords_flat, y_coords_flat) # Note: meshgrid default indexing='xy' matches y[row, col] -> y[y_coord, x_coord]
                data_columns.append(xx.ravel())
                data_columns.append(yy.ravel())

                # Intensity Column(s)
                if is_complex:
                    header_row.extend(["Intensity_Real (a.u.)", "Intensity_Imag (a.u.)"])
                    data_columns.append(np.real(y).ravel())
                    data_columns.append(np.imag(y).ravel())
                else:
                    header_row.append("Intensity (a.u.)")
                    data_columns.append(y.ravel())

            # --- Write Data ---
            writer.writerow(header_row)
            # Transpose columns into rows for writing
            rows_to_write = np.stack(data_columns, axis=-1)
            writer.writerows(rows_to_write)

    except IOError as e:
        warnings.warn(f"Could not write CSV file {csv_file}: {e}")
    except Exception as e:
        warnings.warn(f"An unexpected error occurred while writing CSV {csv_file}: {e}")


def _save_to_hdf5(
    output_basename: Path,
    x: Union[np.ndarray, List[np.ndarray], None],
    y: np.ndarray,
    pars: Dict[str, Any],
    original_file_path: str
) -> None:
    """Saves data and metadata to an HDF5 file."""
    h5_file = output_basename.with_suffix(".h5")
    print(f"  Saving data and metadata to: {h5_file}")

    try:
        with h5py.File(h5_file, 'w') as f:
            # --- Store Metadata ---
            # Create a group for metadata attributes
            meta_grp = f.create_group("metadata")
            meta_grp.attrs["original_file"] = original_file_path
            meta_grp.attrs["description"] = "EPR data and parameters"

            # Store parameters as attributes - handle potential type issues
            pars_grp = meta_grp.create_group("parameters")
            for key, value in pars.items():
                try:
                    # Try direct assignment, h5py handles basic types and numpy arrays well
                    if value is None:
                         pars_grp.attrs[key] = "None" # Store None as string
                    elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float, str, np.number)):
                         # Store simple lists/tuples of basic types directly (h5py > 3 might handle this)
                         # As a safe bet, convert to numpy array or string
                         try:
                             pars_grp.attrs[key] = np.array(value)
                         except TypeError:
                             pars_grp.attrs[key] = str(value) # Fallback to string
                    elif isinstance(value, (Path, )): # Convert Path object to string
                        pars_grp.attrs[key] = str(value)
                    else:
                        pars_grp.attrs[key] = value

                except TypeError as e:
                    # If type is not directly supported by HDF5 attributes, convert to string
                    warnings.warn(f"Could not store parameter '{key}' (type: {type(value)}) directly in HDF5 attributes. Converting to string.")
                    pars_grp.attrs[key] = str(value)

            # --- Store Data ---
            data_grp = f.create_group("data")
            # Store main intensity data
            ds_y = data_grp.create_dataset("intensity", data=y)
            ds_y.attrs["units"] = "a.u." # Assume arbitrary units for intensity
            if np.iscomplexobj(y):
                 ds_y.attrs["signal_type"] = "complex"
            else:
                 ds_y.attrs["signal_type"] = "real"


            # Get Abscissa Units if available
            x_units = pars.get('_Abscissa_Units', [])
            if isinstance(x_units, str): x_units = [x_units]
            x_unit = x_units[0] if len(x_units) > 0 else 'a.u.'
            y_unit = x_units[1] if len(x_units) > 1 else 'a.u.'


            # Store abscissa(e)
            if x is None:
                # Create index axes if no abscissa provided
                if y.ndim == 1:
                    nx = y.shape[0]
                    ds_x = data_grp.create_dataset("abscissa_x", data=np.arange(nx))
                    ds_x.attrs["units"] = "points"
                    ds_x.attrs["axis_type"] = "index"
                elif y.ndim == 2:
                    ny, nx = y.shape
                    ds_x = data_grp.create_dataset("abscissa_x", data=np.arange(nx))
                    ds_x.attrs["units"] = "points"
                    ds_x.attrs["axis_type"] = "index"
                    ds_y_ax = data_grp.create_dataset("abscissa_y", data=np.arange(ny))
                    ds_y_ax.attrs["units"] = "points"
                    ds_y_ax.attrs["axis_type"] = "index"

            elif isinstance(x, np.ndarray): # 1D data with abscissa
                ds_x = data_grp.create_dataset("abscissa_x", data=x)
                ds_x.attrs["units"] = x_unit
                ds_x.attrs["axis_type"] = "independent_variable"
            elif isinstance(x, list) and len(x) >= 1: # Potentially 2D data
                if len(x) >= 1 and x[0] is not None and isinstance(x[0], np.ndarray):
                    ds_x = data_grp.create_dataset("abscissa_x", data=x[0])
                    ds_x.attrs["units"] = x_unit
                    ds_x.attrs["axis_type"] = "independent_variable_x"
                if len(x) >= 2 and x[1] is not None and isinstance(x[1], np.ndarray):
                     ds_y_ax = data_grp.create_dataset("abscissa_y", data=x[1])
                     ds_y_ax.attrs["units"] = y_unit
                     ds_y_ax.attrs["axis_type"] = "independent_variable_y"
            # Link axes to data dimensions using HDF5 Dimension Scales API (optional but good)
            # Check if dimension scales exist before trying to attach
            if 'intensity' in data_grp and 'abscissa_y' in data_grp and y.ndim >=2 :
                 data_grp["intensity"].dims[0].label = 'y'
                 data_grp["intensity"].dims[0].attach_scale(data_grp["abscissa_y"])
            if 'intensity' in data_grp and 'abscissa_x' in data_grp and y.ndim >=1:
                 dim_idx = 1 if y.ndim >= 2 else 0
                 data_grp["intensity"].dims[dim_idx].label = 'x'
                 data_grp["intensity"].dims[dim_idx].attach_scale(data_grp["abscissa_x"])


    except ImportError:
        warnings.warn("h5py library not found. Skipping HDF5 output. Install with 'pip install h5py'")
    except IOError as e:
        warnings.warn(f"Could not write HDF5 file {h5_file}: {e}")
    except Exception as e:
         warnings.warn(f"An unexpected error occurred while writing HDF5 file {h5_file}: {type(e).__name__} - {e}")
         import traceback
         traceback.print_exc()


# --- Main Conversion Function ---

def convert_bruker_to_fair(
    input_file_or_dir: Union[str, Path, None] = None,
    output_dir: Optional[Union[str, Path]] = None,
    scaling: str = "",
    output_formats: List[str] = ['csv_json', 'hdf5']
) -> None:
    """
    Loads Bruker EPR data using eprload and converts it to FAIR formats.

    Args:
        input_file_or_dir (Union[str, Path, None], optional): Path to the Bruker
            data file (.dta, .dsc, .spc, .par) or a directory. If None or
            a directory, a file dialog will open. Defaults to None.
        output_dir (Optional[Union[str, Path]], optional): Directory to save
            the converted files. If None, files are saved in the same
            directory as the input file. Defaults to None.
        scaling (str, optional): Scaling options passed to eprload
            (e.g., 'nPGT'). Defaults to "".
        output_formats (List[str], optional): List of formats to generate.
            Options: 'csv_json', 'hdf5'. Defaults to ['csv_json', 'hdf5'].

    Returns:
        None. Prints status messages and warnings.
    """
    print(f"Starting FAIR conversion process...")

    # Load data using eprload (suppress its internal plotting)
    x, y, pars, original_file_path_str = eprload(
        file_name=input_file_or_dir,
        scaling=scaling,
        plot_if_possible=False # Important: disable plotting in eprload
    )

    # --- Check if loading was successful ---
    if y is None or pars is None or original_file_path_str is None:
        print("Data loading failed or was cancelled by user. Aborting conversion.")
        return

    print(f"Successfully loaded: {original_file_path_str}")
    original_file_path = Path(original_file_path_str)

    # --- Determine Output Location and Basename ---
    if output_dir is None:
        output_path = original_file_path.parent
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Create output dir if needed

    # Use original filename without extension as base for output files
    output_basename = output_path / original_file_path.stem
    print(f"Output base name: {output_basename}")


    # --- Perform Conversions based on requested formats ---
    if not output_formats:
         warnings.warn("No output formats specified. Nothing to do.")
         return

    if 'csv_json' in output_formats:
        print("\nGenerating CSV + JSON output...")
        _save_to_csv_json(output_basename, x, y, pars, original_file_path_str)
        print("...CSV + JSON generation finished.")

    if 'hdf5' in output_formats:
         print("\nGenerating HDF5 output...")
         # Check for h5py again before calling, although _save_to_hdf5 does it too
         try:
             import h5py
             _save_to_hdf5(output_basename, x, y, pars, original_file_path_str)
             print("...HDF5 generation finished.")
         except ImportError:
              warnings.warn("h5py library not found. Skipping HDF5 output. Install with 'pip install h5py'")

    print("\nFAIR conversion process finished.")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running Bruker to FAIR Converter Example ---")
    print("This will first use eprload's dialog to select a Bruker file,")
    print("then convert it to CSV/JSON and HDF5 formats in the same directory.")
    print("-" * 50)

    # Example 1: Use file dialog to select input, save to same directory
    convert_bruker_to_fair() # output_dir=None by default

    # # Example 2: Specify an input file and output directory
    # input_bruker_file = "path/to/your/test_data.DTA" # <-- CHANGE THIS
    # output_folder = "path/to/your/output_folder"   # <-- CHANGE THIS
    # input_path = Path(input_bruker_file)
    #
    # if input_path.exists():
    #     print("\n" + "-" * 50)
    #     print(f"--- Example 2: Processing specified file: {input_bruker_file} ---")
    #     convert_bruker_to_fair(
    #         input_file_or_dir=input_bruker_file,
    #         output_dir=output_folder,
    #         scaling="nG" # Example scaling
    #     )
    # else:
    #      print(f"\nSkipping Example 2: Input file not found at '{input_bruker_file}'")

    print("-" * 50)
    print("--- Converter Example Finished ---")