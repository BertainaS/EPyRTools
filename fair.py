# convert_to_fair.py
import eprload as per # Import your library as 'per'
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
import datetime
import warnings

# --- Helper function to make metadata JSON serializable ---
def make_serializable(data):
    """Converts numpy arrays and other non-JSON types in a dict to lists/strings."""
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        # Check if array contains non-numeric types like objects or strings
        if data.dtype.kind in ('O', 'S', 'U'):
             warnings.warn(f"Converting numpy array with dtype {data.dtype} to list. Potential data alteration for complex objects.")
        return data.tolist() # Convert numpy arrays to nested lists
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        # Handle potential NaN/Inf which are not standard JSON
        if np.isnan(data): return 'NaN'
        if np.isinf(data): return 'Infinity' if data > 0 else '-Infinity'
        return float(data)
    elif isinstance(data, (np.complex_, np.complex64, np.complex128)):
        return {'real': make_serializable(data.real), 'imag': make_serializable(data.imag)}
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (Path, datetime.datetime, datetime.date)):
        return str(data) # Convert Path and datetime objects to strings
    # Basic types that are already JSON serializable
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        warnings.warn(f"Data type {type(data)} is not explicitly handled for JSON serialization. Converting to string.")
        return str(data) # Fallback: convert unknown types to string

# --- Main Conversion Function ---
def convert_bruker_to_fair(input_path, output_dir, scaling_options=""):
    """
    Loads Bruker EPR data using per.eprload and saves it as CSV + JSON.

    Args:
        input_path (str): Path to the Bruker data file (.dta, .dsc, .spc, .par)
                          or directory to browse.
        output_dir (str): Directory where the FAIR CSV and JSON files will be saved.
        scaling_options (str): Scaling string passed to per.eprload (e.g., 'nG').

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    print(f"Attempting to load: {input_path}")
    print(f"Using scaling: '{scaling_options}'")

    try:
        # Call your eprload function, suppressing its default plot
        x_data, y_data, parameters, source_file_path = per.eprload(
            input_path,
            scaling=scaling_options,
            plot_if_possible=False # We don't need the default plot here
        )

        # Check if loading failed (e.g., user cancel, file error)
        if y_data is None or source_file_path is None:
            print(f"Failed to load data from '{input_path}'. Skipping conversion.", file=sys.stderr)
            return False

        source_file = Path(source_file_path)
        print(f"Successfully loaded: {source_file.name}")

        # --- Prepare Output ---
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Create output dir if needed

        # Generate base filename for outputs
        base_output_name = source_file.stem

        # --- Save Data (CSV) ---
        data_filename = output_path / f"{base_output_name}_data.csv"
        print(f"Saving data to: {data_filename}")

        if y_data.ndim == 1:
            # 1D Data: Create a 2-column CSV (abscissa, ordinate)
            header = "Abscissa,Ordinate"
            if x_data is not None and isinstance(x_data, np.ndarray) and x_data.shape == y_data.shape:
                # Try to get units from parameters for header (example)
                x_unit = parameters.get('XUNI', 'a.u.')
                if isinstance(x_unit, list): x_unit = x_unit[0] # Take first for 1D
                y_unit = "Intensity (a.u.)" # Add Y unit if available in params?
                header = f"Abscissa ({x_unit}),Ordinate ({y_unit})"
                # Stack horizontally (column-wise)
                save_data = np.vstack((x_data, y_data)).T
            else:
                warnings.warn("Abscissa data missing or incompatible. Saving ordinate data with index.")
                # Save ordinate only, add index column
                header = "Index,Ordinate (a.u.)"
                save_data = np.vstack((np.arange(y_data.size), y_data)).T

            # Handle complex data
            if np.iscomplexobj(save_data):
                 # Save real and imaginary parts in separate columns
                 header += "_Real,Ordinate_Imag" # Adjust header if needed
                 real_part = save_data[:, 1].real
                 imag_part = save_data[:, 1].imag
                 save_data = np.vstack((save_data[:, 0].real, real_part, imag_part)).T # Keep abscissa real

            np.savetxt(data_filename, save_data, delimiter=',', header=header, comments='')

        elif y_data.ndim == 2:
            # 2D Data: Save the matrix as CSV. Store axes info in metadata.
            # Note: CSV is not ideal for large 2D data, but keeps it simple text.
            print("Saving 2D data matrix. Axes details will be in the JSON metadata.")
            header = f"2D Data Matrix ({y_data.shape[0]}x{y_data.shape[1]})"
            
            save_matrix = y_data
            # Handle complex 2D data: save real part only to CSV for simplicity
            # Or save two separate files (real/imag)? Let's save real for now.
            # Metadata will indicate if original was complex.
            if np.iscomplexobj(save_matrix):
                warnings.warn("Saving only the real part of complex 2D data to CSV.")
                header += " - Real Part Only"
                save_matrix = save_matrix.real
                
            np.savetxt(data_filename, save_matrix, delimiter=',', header=header, comments='')

        else:
            print(f"Data has {y_data.ndim} dimensions. CSV saving implemented only for 1D/2D. Skipping data save.", file=sys.stderr)
            # Optionally save as NPY binary format?
            # npy_filename = output_path / f"{base_output_name}_data.npy"
            # np.save(npy_filename, y_data)
            # print(f"Saved high-dimensional data to binary NumPy format: {npy_filename}")
            
        # --- Save Metadata (JSON) ---
        metadata_filename = output_path / f"{base_output_name}_metadata.json"
        print(f"Saving metadata to: {metadata_filename}")

        # Prepare metadata dictionary
        fair_metadata = {
            "FAIR_conversion_details": {
                "script_name": Path(__file__).name,
                "conversion_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "EPyRTools_version": getattr(per, '__version__', 'unknown'), # Add version if EPyRTools has one
                "scaling_applied_by_eprload": scaling_options,
            },
            "source_data": {
                "original_filename": source_file.name,
                "original_full_path": str(source_file.resolve()), # Full path for traceability
                "format_type_guessed_by_eprload": parameters.get('_FormatGuess', 'unknown') # Store how eprload identified it if available
            },
            "data_representation": {
                 "data_file": data_filename.name, # Relative path within the output dir
                 "dimensions": y_data.shape,
                 "is_complex": np.iscomplexobj(y_data).item(), # Store bool if data was complex
                 "axes": []
                 # Future: Add NPY file details here if used for >2D
            },
            "original_parameters": make_serializable(parameters) # Add the original params
        }

        # Add axes info to metadata
        if x_data is not None:
            if isinstance(x_data, np.ndarray) and y_data.ndim == 1:
                x_label = parameters.get("XNAM", "Abscissa")
                x_unit = parameters.get("XUNI", "a.u.")
                fair_metadata["data_representation"]["axes"].append({
                    "axis": 0,
                    "name": x_label,
                    "unit": x_unit,
                    "data_in_file": data_filename.name, # Indicates data is in the CSV
                    "column_index (1D)": 0 # 0-based index
                })
            elif isinstance(x_data, list) and y_data.ndim == 2 and len(x_data) >= 2:
                # Assume x_data = [x_axis_vec, y_axis_vec] for 2D
                 axis_names = parameters.get("_AxesNames", ["X", "Y"]) # Get from eprload/utils if available
                 axis_units = parameters.get("_Abscissa_Units", ["a.u.", "a.u."])
                 if not isinstance(axis_units, list): axis_units = [axis_units] * len(axis_names) # Ensure list

                 # X-axis (corresponds to columns, numpy axis 1)
                 if isinstance(x_data[0], np.ndarray) and x_data[0].size == y_data.shape[1]:
                     fair_metadata["data_representation"]["axes"].append({
                        "axis": 1, # Numpy axis index
                        "name": axis_names[0],
                        "unit": axis_units[0] if len(axis_units) > 0 else "a.u.",
                        "values": make_serializable(x_data[0]) # Embed axis values
                     })
                 else:
                     warnings.warn(f"X-axis data for 2D plot has unexpected shape/type ({type(x_data[0])}, size {getattr(x_data[0], 'size', 'N/A')}) vs data cols ({y_data.shape[1]}). Skipping X-axis metadata.")


                 # Y-axis (corresponds to rows, numpy axis 0)
                 if isinstance(x_data[1], np.ndarray) and x_data[1].size == y_data.shape[0]:
                    fair_metadata["data_representation"]["axes"].append({
                       "axis": 0, # Numpy axis index
                       "name": axis_names[1] if len(axis_names) > 1 else "Y",
                       "unit": axis_units[1] if len(axis_units) > 1 else "a.u.",
                       "values": make_serializable(x_data[1]) # Embed axis values
                    })
                 else:
                     warnings.warn(f"Y-axis data for 2D plot has unexpected shape/type ({type(x_data[1])}, size {getattr(x_data[1], 'size', 'N/A')}) vs data rows ({y_data.shape[0]}). Skipping Y-axis metadata.")
            else:
                warnings.warn("Format of 'x_data' is not directly usable for standard axes metadata.")
                fair_metadata["data_representation"]["axes"].append({
                     "axis_info_raw": f"Type: {type(x_data)}, consult original parameters or inspect data."
                })


        # Write JSON metadata file
        try:
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(fair_metadata, f, indent=4, ensure_ascii=False)
        except TypeError as e:
            print(f"\nError: Could not serialize metadata to JSON: {e}", file=sys.stderr)
            print("This might happen if the parameters dictionary contains complex objects", file=sys.stderr)
            print("Check the output of the 'make_serializable' function warnings.", file=sys.stderr)
            return False
        except Exception as e:
             print(f"\nError writing JSON metadata file: {e}", file=sys.stderr)
             return False

        print("Conversion successful.")
        return True

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}", file=sys.stderr)
        return False
    except ImportError:
        print("Error: Could not import the 'EPyRTools' library as 'per'.", file=sys.stderr)
        print("Please ensure EPyRTools is installed and accessible in your Python environment.", file=sys.stderr)
        print("Also check that eprload.py and its 'sub' directory are correctly placed within the EPyRTools package.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Bruker EPR data (BES3T, ESP) to FAIR format (CSV data + JSON metadata).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_path",
        help="Path to the Bruker data file (.dta, .dsc, .spc, .par) OR a directory to open a file browser."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="FAIR_output",
        help="Directory to save the converted CSV and JSON files."
    )
    parser.add_argument(
        "-s", "--scaling",
        default="",
        help="Scaling options string passed to eprload (e.g., 'nG'). See eprload documentation for details."
    )
    # Example: Add version if your package has one
    try:
       version = getattr(per, '__version__', 'unknown')
       parser.add_argument('--version', action='version', version=f'%(prog)s using EPyRTools version {version}')
    except ImportError:
        # Handle case where per couldn't be imported even for version initially
        pass
    except AttributeError:
         pass # If per doesn't have __version__

    args = parser.parse_args()

    # Use Path object for input path handling
    input_path_obj = Path(args.input_path)

    if input_path_obj.is_dir():
        # If directory, let eprload handle the file dialog via convert function
        # We expect convert_bruker_to_fair to potentially show the dialog
        print(f"Input path '{args.input_path}' is a directory. eprload should open a file dialog...")
        success = convert_bruker_to_fair(args.input_path, args.output_dir, args.scaling)
    elif input_path_obj.is_file():
        # If file, proceed directly
        success = convert_bruker_to_fair(args.input_path, args.output_dir, args.scaling)
    elif not input_path_obj.exists():
        # If doesn't exist, could be a path for the dialog *or* an error
        # We pass it to eprload to decide / raise FileNotFoundError if needed
        print(f"Input path '{args.input_path}' does not exist. Passing to eprload (may open dialog or error out)...")
        success = convert_bruker_to_fair(args.input_path, args.output_dir, args.scaling)

    if not success:
        sys.exit(1) # Exit with error code if conversion failed
    else:
         print("\nConversion process finished.")
         sys.exit(0)