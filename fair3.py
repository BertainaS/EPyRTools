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
from datetime import datetime # For timestamp

# Assuming eprload is in the same directory or installable
try:
    from eprload import eprload
except ImportError:
    print("Error: Could not import 'eprload'. Make sure eprload.py is in the same directory or accessible in the Python path.", file=sys.stderr)
    sys.exit(1)

# --- Bruker Parameter Map ---
# Define mapping from Bruker keys to FAIR representations
# Add more parameters as needed!
# Units should reflect the typical value expected after potential parsing in eprload

BRUKER_PARAM_MAP = {
    # === Microwave Bridge Parameters ===
    "MWFQ": {"fair_name": "microwave_frequency", "unit": "GHz", "description": "Microwave frequency. Usually measured by a frequency counter."},
    "MWPW": {"fair_name": "microwave_power_attenuation", "unit": "W", "description": "Microwave power *setting* on the bridge (BES3T). Often related to source power minus attenuation setting. Use MP for ESP calculated power."},
    "MP":   {"fair_name": "microwave_power_at_sample", "unit": "mW", "description": "Microwave power *estimated* at the sample (ESP/WinEPR). Often calculated based on bridge attenuation and source power. Calibration is usually required for accuracy."},
    "PowerAtten": {"fair_name": "microwave_power_attenuator", "unit": "dB", "description": "Microwave attenuator setting (dB)."},
    "PowerReading": {"fair_name": "microwave_power_reading", "unit": "mW", "description": "Direct power reading from a sensor, if available."}, # Might be custom/less common

    # === Magnetic Field Parameters ===
    "HCF":  {"fair_name": "field_center", "unit": "G", "description": "Center magnetic field for the sweep (ESP)."},
    "HSW":  {"fair_name": "field_sweep_width", "unit": "G", "description": "Magnetic field sweep width (ESP). Total range = HSW."},
    "GST":  {"fair_name": "field_sweep_start", "unit": "G", "description": "Magnetic field sweep start value (ESP)."},
    "GSI":  {"fair_name": "field_sweep_increment", "unit": "G", "description": "Magnetic field sweep increment (step size) (ESP)."},
    "HMA":  {"fair_name": "field_modulation_amplitude", "unit": "G", "description": "Field modulation amplitude (ESP). Check if peak, pk-pk, or RMS."}, # Older key, prefer MA
    "HMF":  {"fair_name": "field_modulation_frequency", "unit": "Hz", "description": "Field modulation frequency (ESP)."}, # Older key, prefer MF
    "XMIN": {"fair_name": "axis_x_min", "unit": "determined_by_XUNI", "description": "Minimum value of the primary (X) axis (BES3T). Often field."},
    "XWID": {"fair_name": "axis_x_width", "unit": "determined_by_XUNI", "description": "Width (range) of the primary (X) axis (BES3T)."},
    "FieldStart": {"fair_name": "field_scan_start", "unit": "G", "description": "Starting field for the scan (BES3T). May overlap with XMIN."},
    "FieldStop": {"fair_name": "field_scan_stop", "unit": "G", "description": "Stopping field for the scan (BES3T)."},

    # === Modulation Parameters ===
    "MF":   {"fair_name": "modulation_frequency", "unit": "Hz", "description": "Field modulation frequency setting."},
    "MA":   {"fair_name": "modulation_amplitude", "unit": "G", "description": "Field modulation amplitude setting. Check if peak, peak-to-peak, or RMS."},
    "ModPhase": {"fair_name": "modulation_phase", "unit": "deg", "description": "Modulation phase relative to detection reference (degrees)."}, # Primarily BES3T

    # === Receiver / Signal Channel Parameters ===
    "RCAG": {"fair_name": "receiver_gain_db", "unit": "dB", "description": "Receiver gain setting (BES3T, logarithmic dB scale)."},
    "RRG":  {"fair_name": "receiver_gain_counts", "unit": "", "description": "Receiver gain setting (ESP, often linear arbitrary counts)."},
    "ConvGain": {"fair_name": "conversion_gain", "unit": "", "description": "Signal channel overall conversion gain (older systems)."}, # Might overlap with RRG/RCAG
    "SPTP": {"fair_name": "dwell_time", "unit": "s", "description": "Dwell time per point (sampling period) (BES3T)."},
    "RCT":  {"fair_name": "conversion_time", "unit": "ms", "description": "ADC conversion time per point (ESP)."},
    "RTC":  {"fair_name": "receiver_time_constant", "unit": "s", "description": "Receiver time constant (filter setting) (BES3T)."},
    "TCN":  {"fair_name": "time_constant_number", "unit": "", "description": "Time constant selection number (ESP). Needs lookup table."}, # ESP, less direct
    "RPH":  {"fair_name": "receiver_phase", "unit": "deg", "description": "Receiver phase setting (signal channel phase adjust)."},
    "RMA":  {"fair_name": "receiver_mode", "unit": "", "description": "Receiver mode (e.g., Quadrature, In-Phase)."}, # Often coded, e.g., in JSS
    "HAR":  {"fair_name": "detection_harmonic", "unit": "", "description": "Harmonic of modulation frequency used for detection (1=fundamental, 2=second harmonic, etc.)."},
    "QVA":  {"fair_name": "q_value", "unit": "", "description": "Estimated Q-value of the resonator, if calculated."},

    # === Acquisition / Averaging Parameters ===
    "AVGS": {"fair_name": "num_scans_set", "unit": "", "description": "Number of scans *set* for acquisition (BES3T). Total accumulations = AVGS * NSC."},
    "NSC":  {"fair_name": "num_scans_per_trigger", "unit": "", "description": "Number of scans acquired per trigger or block (BES3T)."}, # Used with AVGS
    "JSD":  {"fair_name": "num_scans_done", "unit": "", "description": "Number of scans *actually performed* (ESP)."},
    "SctNorm": {"fair_name": "bruker_scaling_applied", "unit": "", "description": "Flag indicating if Bruker scaling (averaging, gain etc.) was applied internally by Xepr (true/false)."}, # BES3T
    "AveragingMode": {"fair_name": "averaging_mode", "unit": "", "description": "Type of averaging performed (e.g., Linear, Exponential)."}, # Primarily pulse/transient

    # === Temperature Parameters ===
    "STMP": {"fair_name": "temperature_setpoint", "unit": "K", "description": "Setpoint temperature for the controller (BES3T)."},
    "TE":   {"fair_name": "temperature_measured", "unit": "K", "description": "Measured temperature, usually at sensor location (ESP)."},
    "Temperature": {"fair_name": "temperature_controller_reading", "unit": "K", "description": "Current temperature reading from the controller (BES3T). May overlap with STMP if stable."}, # BES3T
    "TCT":  {"fair_name": "temperature_controller_type", "unit": "", "description": "Type of temperature controller used."},

    # === Data Structure / Format Parameters ===
    "XPTS": {"fair_name": "points_x", "unit": "points", "description": "Number of points along the primary (X) axis."},
    "YPTS": {"fair_name": "points_y", "unit": "points", "description": "Number of points along the secondary (Y) axis (for 2D/3D)."},
    "ZPTS": {"fair_name": "points_z", "unit": "points", "description": "Number of points along the tertiary (Z) axis (for 3D)."},
    "RES":  {"fair_name": "resolution_x_esp", "unit": "points", "description": "Number of points along the X axis (older ESP name)."},
    "REY":  {"fair_name": "resolution_y_esp", "unit": "points", "description": "Number of points along the Y axis (older ESP name)."},
    "ANZ":  {"fair_name": "total_points_esp", "unit": "points", "description": "Total number of data points in the file (ESP). Useful for consistency check."},
    "SSX":  {"fair_name": "scanned_size_x_esp", "unit": "points", "description": "Number of points acquired along X dimension (ESP, often pulse)."},
    "SSY":  {"fair_name": "scanned_size_y_esp", "unit": "points", "description": "Number of points acquired along Y dimension (ESP, often pulse)."},
    "IKKF": {"fair_name": "data_kind", "unit": "", "description": "Kind of data stored per point (e.g., REAL, CPLX, REAL,IMAG) (BES3T). List for multi-channel."},
    "BSEQ": {"fair_name": "byte_sequence", "unit": "", "description": "Byte order of data values (BIG=big-endian, LIT=little-endian) (BES3T)."},
    "IRFMT":{"fair_name": "data_real_format", "unit": "", "description": "Number format for real part data points (e.g., I=int32, F=float32, D=float64) (BES3T)."},
    "IIFMT":{"fair_name": "data_imag_format", "unit": "", "description": "Number format for imaginary part data points (if complex) (BES3T)."},
    "JSS":  {"fair_name": "status_flags_esp", "unit": "", "description": "Job and Spectrometer Status flags (bit-encoded) (ESP). Indicates complex, 2D, etc."}, # Needs decoding
    "DOS":  {"fair_name": "operating_system_esp", "unit": "", "description": "Indicates file origin (DOS/Windows vs Unix) affecting endianness (ESP)."}, # WinEPR

    # === Axis Definition Parameters (BES3T primarily, but used for calculating abscissa in general) ===
    "XUNI": {"fair_name": "axis_x_unit", "unit": "", "description": "Unit label for the X axis (e.g., 'G', 's', 'MHz')."},
    "YUNI": {"fair_name": "axis_y_unit", "unit": "", "description": "Unit label for the Y axis."},
    "ZUNI": {"fair_name": "axis_z_unit", "unit": "", "description": "Unit label for the Z axis."},
    "YMIN": {"fair_name": "axis_y_min", "unit": "determined_by_YUNI", "description": "Minimum value of the Y axis."},
    "YWID": {"fair_name": "axis_y_width", "unit": "determined_by_YUNI", "description": "Width (range) of the Y axis."},
    "ZMIN": {"fair_name": "axis_z_min", "unit": "determined_by_ZUNI", "description": "Minimum value of the Z axis."},
    "ZWID": {"fair_name": "axis_z_width", "unit": "determined_by_ZUNI", "description": "Width (range) of the Z axis."},
    "XTYP": {"fair_name": "axis_x_type", "unit": "", "description": "Type of X axis (IDX=linear/indexed, IGD=indirect/gradient list) (BES3T)."},
    "YTYP": {"fair_name": "axis_y_type", "unit": "", "description": "Type of Y axis (BES3T)."},
    "ZTYP": {"fair_name": "axis_z_type", "unit": "", "description": "Type of Z axis (BES3T)."},
    "XGF":  {"fair_name": "axis_x_gradient_file", "unit": "", "description": "Companion file suffix for non-linear X axis values (BES3T)."}, # Not a value, but file info
    "YGF":  {"fair_name": "axis_y_gradient_file", "unit": "", "description": "Companion file suffix for non-linear Y axis values (BES3T)."},
    "ZGF":  {"fair_name": "axis_z_gradient_file", "unit": "", "description": "Companion file suffix for non-linear Z axis values (BES3T)."},
    "XXLB": {"fair_name": "pulse_axis_x_start_esp", "unit": "unit_from_XXUN", "description": "Start value for X axis in pulse experiments (ESP)."},
    "XXWI": {"fair_name": "pulse_axis_x_width_esp", "unit": "unit_from_XXUN", "description": "Width/range for X axis in pulse experiments (ESP)."},
    "XYLB": {"fair_name": "pulse_axis_y_start_esp", "unit": "unit_from_XYUN", "description": "Start value for Y axis in 2D pulse experiments (ESP)."},
    "XYWI": {"fair_name": "pulse_axis_y_width_esp", "unit": "unit_from_XYUN", "description": "Width/range for Y axis in 2D pulse experiments (ESP)."},
    "XXUN": {"fair_name": "pulse_axis_x_unit_esp", "unit": "", "description": "Unit for X axis in pulse experiments (ESP)."},
    "XYUN": {"fair_name": "pulse_axis_y_unit_esp", "unit": "", "description": "Unit for Y axis in pulse experiments (ESP)."},

    # === General Metadata ===
    "TITL": {"fair_name": "title", "unit": "", "description": "User-defined title for the experiment."},
    "OPER": {"fair_name": "operator_name", "unit": "", "description": "Name of the operator who ran the experiment."},
    "DATE": {"fair_name": "acquisition_date", "unit": "", "description": "Date of data acquisition."},
    "TIME": {"fair_name": "acquisition_time", "unit": "", "description": "Time of data acquisition."},
    "CMNT": {"fair_name": "comment", "unit": "", "description": "User-provided comments about the experiment."},
    "SAMP": {"fair_name": "sample_description", "unit": "", "description": "Description of the sample."},
    "SMNM": {"fair_name": "sample_name", "unit": "", "description": "Short name or code for the sample."},
    "FSRC": {"fair_name": "file_source", "unit": "", "description": "Originating software or system (e.g., Xepr, WinEPR)."}, # May not exist, can be inferred
    "VERS": {"fair_name": "software_version", "unit": "", "description": "Version of the acquisition software."},
    "INST": {"fair_name": "instrument_id", "unit": "", "description": "Identifier for the EPR spectrometer used."},

    # === Experiment Type Indicators ===
    "EXPT": {"fair_name": "experiment_mode_bes3t", "unit": "", "description": "Experiment mode declared in BES3T (e.g., CW, Pulse, ENDOR)."}, # BES3T
    "JEX":  {"fair_name": "experiment_type_esp", "unit": "", "description": "Primary experiment type declared in ESP (e.g., 'field-sweep', 'ENDOR', 'time-sweep')."}, # ESP
    "JEY":  {"fair_name": "experiment_axis_y_esp", "unit": "", "description": "Parameter varied along the Y-axis in ESP (e.g., 'frequency-sweep', 'mw-power-sweep')."}, # ESP (for 2D)

    # === Pulse Experiment Parameters (Examples - Add more as needed!) ===
    # --- Pulse Definitions (Lengths/Amplitudes) ---
    "P1":   {"fair_name": "pulse_1_length", "unit": "ns", "description": "Length of pulse P1."},
    "P2":   {"fair_name": "pulse_2_length", "unit": "ns", "description": "Length of pulse P2."},
    "P3":   {"fair_name": "pulse_3_length", "unit": "ns", "description": "Length of pulse P3."},
    # ... up to P32 or more ...
    "A1":   {"fair_name": "pulse_1_amplitude", "unit": "dB", "description": "Amplitude/attenuation setting for pulse P1 (relative to full power)."}, # Unit might vary
    "A2":   {"fair_name": "pulse_2_amplitude", "unit": "dB", "description": "Amplitude/attenuation setting for pulse P2."},
    # ...
    # --- Delays ---
    "D1":   {"fair_name": "delay_1", "unit": "ns", "description": "Duration of delay D1."},
    "D2":   {"fair_name": "delay_2", "unit": "ns", "description": "Duration of delay D2."},
    # ... up to D32 or more ...
    # --- Acquisition Parameters ---
    "DW":   {"fair_name": "pulse_dwell_time", "unit": "ns", "description": "Dwell time (sampling interval) during pulse acquisition."}, # Often called T/td instead
    "AQWD": {"fair_name": "pulse_acquisition_window_duration", "unit": "ns", "description": "Total duration of the signal acquisition window."},
    "DEC":  {"fair_name": "pulse_decimation_factor", "unit": "", "description": "Decimation factor applied during digital filtering/acquisition."},
    "DS":   {"fair_name": "pulse_delay_start_acquisition", "unit": "ns", "description": "Delay between last pulse and start of acquisition (dead time compensation)."},
    # --- Repetition / Averaging ---
    "TR":   {"fair_name": "pulse_repetition_time", "unit": "us", "description": "Time between start of consecutive shots/sequences."}, # Unit might vary (ms, s)
    "NEX":  {"fair_name": "pulse_num_experiments", "unit": "", "description": "Number of times the entire experiment sequence (including phase cycles/loops) is repeated for averaging."},
    "PEX":  {"fair_name": "pulse_phase_cycle_length", "unit": "", "description": "Number of steps in the primary phase cycle."},
    # --- Loop Counters (Examples) ---
    "L1":   {"fair_name": "loop_1_count", "unit": "", "description": "Number of iterations for loop L1."},
    "L2":   {"fair_name": "loop_2_count", "unit": "", "description": "Number of iterations for loop L2."},
    # ...

}

# --- Helper Functions for Saving ---

def _process_parameters(pars: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Processes the raw parameters dictionary using the BRUKER_PARAM_MAP."""
    fair_metadata = {}
    unmapped_parameters = {}

    # Add explicit conversion metadata
    fair_metadata["conversion_info"] = {
        "value": { # Nest value to match other parameter structure
            "converter_script": "fair_converter.py", # Or add versioning
            "conversion_timestamp": datetime.now().isoformat(),
            "eprload_version": "unknown" # TODO: Add version to eprload if possible
        },
        "unit": "",
        "description": "Information about the conversion process to FAIR format."
    }


    for key, value in pars.items():
        # Skip internal keys added by eprload (like abscissa units)
        if key.startswith('_'):
            continue

        if key in BRUKER_PARAM_MAP:
            map_info = BRUKER_PARAM_MAP[key]
            fair_key = map_info["fair_name"]

            # Determine unit: Use map unit, but check if specific axis unit overrides it
            unit = map_info["unit"]
            if unit == "determined_by_XUNI" and "XUNI" in pars:
                unit = pars.get("XUNI", "unknown")
            elif unit == "determined_by_YUNI" and "YUNI" in pars:
                 # Handle potential list format for YUNI/ZUNI if IKKF indicates multiple traces
                 yuni_val = pars.get("YUNI", "unknown")
                 unit = yuni_val.split(',')[0].strip() if isinstance(yuni_val, str) else str(yuni_val)

            elif unit == "determined_by_ZUNI" and "ZUNI" in pars:
                 zuni_val = pars.get("ZUNI", "unknown")
                 unit = zuni_val.split(',')[0].strip() if isinstance(zuni_val, str) else str(zuni_val)


            fair_metadata[fair_key] = {
                "value": value,
                "unit": unit,
                "description": map_info["description"]
            }
        else:
            # Store unmapped parameters separately
            unmapped_parameters[key] = value
            warnings.warn(f"Parameter '{key}' not found in BRUKER_PARAM_MAP. Storing in 'unmapped_parameters'.")

    return fair_metadata, unmapped_parameters


def _save_to_csv_json(
    output_basename: Path,
    x: Union[np.ndarray, List[np.ndarray], None],
    y: np.ndarray,
    pars: Dict[str, Any],
    original_file_path: str
) -> None:
    """Saves data to CSV and structured metadata to JSON."""
    json_file = output_basename.with_suffix(".json")
    csv_file = output_basename.with_suffix(".csv")

    fair_meta, unmapped_meta = _process_parameters(pars)

    print(f"  Saving structured metadata to: {json_file}")
    # --- Save Metadata to JSON ---
    metadata_to_save = {
        "original_file": original_file_path,
        "fair_metadata": fair_meta,
    }
    if unmapped_meta: # Only add if there are unmapped parameters
        metadata_to_save["unmapped_parameters"] = unmapped_meta

    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=4, default=str) # default=str handles non-serializables
    except IOError as e:
        warnings.warn(f"Could not write JSON file {json_file}: {e}")
    except TypeError as e:
        warnings.warn(f"Error serializing metadata to JSON for {json_file}: {e}. Some parameters might not be saved correctly.")

    print(f"  Saving data to: {csv_file}")
    # --- Save Data to CSV ---
    # (CSV saving logic remains the same as before, it doesn't use the detailed map directly)
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # --- Write Header ---
            writer.writerow(['# EPR Data Export'])
            writer.writerow(['# Original File:', original_file_path])
            # Add a few key parameters derived from the FAIR map if available
            mwfq_info = fair_meta.get('microwave_frequency', {})
            field_info = fair_meta.get('field_center', fair_meta.get('field_sweep_start', {}))
            sweep_info = fair_meta.get('field_sweep_width', fair_meta.get('field_sweep_increment', {}))

            writer.writerow(['# Microwave_Frequency:', f"{mwfq_info.get('value', 'N/A')} {mwfq_info.get('unit', '')}".strip()])
            writer.writerow(['# Field_Center/Start:', f"{field_info.get('value', 'N/A')} {field_info.get('unit', '')}".strip()])
            writer.writerow(['# Field_Sweep/Increment:', f"{sweep_info.get('value', 'N/A')} {sweep_info.get('unit', '')}".strip()])
            writer.writerow(['# Data_Shape:', str(y.shape)])
            writer.writerow(['# Data_Type:', str(y.dtype)])
            writer.writerow(['# ---'])

            # --- Prepare Data Columns ---
            header_row = []
            data_columns = []
            is_complex = np.iscomplexobj(y)
            is_2d = y.ndim == 2

            # Get Abscissa Units using potentially mapped keys first
            x_unit_val = fair_meta.get('axis_x_unit', {}).get('value', 'a.u.')
            # Handle list case from IKKF in BES3T for Y/Z units if needed
            y_unit_val = fair_meta.get('axis_y_unit', {}).get('value', 'a.u.')
            if isinstance(y_unit_val, str) and ',' in y_unit_val: y_unit_val = y_unit_val.split(',')[0].strip()
            z_unit_val = fair_meta.get('axis_z_unit', {}).get('value', 'a.u.')
            if isinstance(z_unit_val, str) and ',' in z_unit_val: z_unit_val = z_unit_val.split(',')[0].strip()


            if not is_2d: # 1D Data
                n_pts = y.shape[0]
                # Abscissa Column
                if x is not None and isinstance(x, np.ndarray) and x.shape == y.shape:
                    header_row.append(f"Abscissa ({x_unit_val})")
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
                        header_row[0] = f"X_Axis ({x_unit_val})"
                    if isinstance(y_axis, np.ndarray) and y_axis.size == ny:
                        y_coords_flat = y_axis
                        header_row[1] = f"Y_Axis ({y_unit_val})" # Use Y unit here
                elif isinstance(x, np.ndarray) and x.ndim == 1 and x.size == nx:
                     x_coords_flat = x
                     header_row[0] = f"X_Axis ({x_unit_val})"

                # Create grid and flatten
                xx, yy = np.meshgrid(x_coords_flat, y_coords_flat)
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
            rows_to_write = np.stack(data_columns, axis=-1)
            writer.writerows(rows_to_write)

    except IOError as e:
        warnings.warn(f"Could not write CSV file {csv_file}: {e}")
    except Exception as e:
        warnings.warn(f"An unexpected error occurred while writing CSV {csv_file}: {e}")

def _try_set_h5_attr(h5_object, key: str, value: Any):
    """Helper to safely set HDF5 attributes, converting to string on type error."""
    try:
        if value is None:
            h5_object.attrs[key] = "None" # Store None as string
        elif isinstance(value, (list, tuple)) and all(isinstance(i, (int, float, str, np.number, bytes)) for i in value):
             # Store lists/tuples of basic types directly if h5py supports it, else convert
             try:
                  h5_object.attrs[key] = value # Let h5py handle if possible (newer versions are better)
             except TypeError:
                 # Try converting to numpy array first
                 try:
                     h5_object.attrs[key] = np.array(value)
                 except TypeError:
                      h5_object.attrs[key] = str(value) # Fallback to string
        elif isinstance(value, Path): # Convert Path object to string
            h5_object.attrs[key] = str(value)
        else:
            h5_object.attrs[key] = value # Try direct assignment

    except TypeError:
        # If type is not directly supported by HDF5 attributes, convert to string
        warnings.warn(f"Could not store attribute '{key}' (type: {type(value)}) directly in HDF5 attributes. Converting to string.")
        h5_object.attrs[key] = str(value)
    except Exception as e:
         warnings.warn(f"Unexpected error storing attribute '{key}': {type(e).__name__} - {e}. Skipping.")


def _save_to_hdf5(
    output_basename: Path,
    x: Union[np.ndarray, List[np.ndarray], None],
    y: np.ndarray,
    pars: Dict[str, Any],
    original_file_path: str
) -> None:
    """Saves data and structured metadata to an HDF5 file."""
    h5_file = output_basename.with_suffix(".h5")
    fair_meta, unmapped_meta = _process_parameters(pars)

    print(f"  Saving structured data and metadata to: {h5_file}")

    try:
        with h5py.File(h5_file, 'w') as f:
            # --- Store Global Metadata ---
            f.attrs["original_file"] = original_file_path
            f.attrs["description"] = "FAIR representation of EPR data converted from Bruker format."
            f.attrs["conversion_timestamp"] = datetime.now().isoformat()
            f.attrs["converter_script_version"] = "fair_converter_v1.1" # Example version

            # --- Store Structured FAIR Metadata ---
            param_grp = f.create_group("metadata/parameters_fair")
            param_grp.attrs["description"] = "Mapped parameters with units and descriptions."

            for fair_key, info in fair_meta.items():
                item_grp = param_grp.create_group(fair_key)
                _try_set_h5_attr(item_grp, "value", info["value"])
                _try_set_h5_attr(item_grp, "unit", info["unit"])
                _try_set_h5_attr(item_grp, "description", info["description"])

            # --- Store Unmapped Parameters ---
            if unmapped_meta:
                unmap_grp = f.create_group("metadata/parameters_original")
                unmap_grp.attrs["description"] = "Parameters from the original file not found in the FAIR mapping."
                for key, value in unmapped_meta.items():
                    _try_set_h5_attr(unmap_grp, key, value)

            # --- Store Data ---
            data_grp = f.create_group("data")
            ds_y = data_grp.create_dataset("intensity", data=y)
            ds_y.attrs["description"] = "Experimental intensity data."
            ds_y.attrs["units"] = "a.u."
            if np.iscomplexobj(y):
                 ds_y.attrs["signal_type"] = "complex"
            else:
                 ds_y.attrs["signal_type"] = "real"

            # Get Abscissa Units from FAIR map
            x_unit_val = fair_meta.get('axis_x_unit', {}).get('value', 'a.u.')
            y_unit_val = fair_meta.get('axis_y_unit', {}).get('value', 'a.u.')
            if isinstance(y_unit_val, str) and ',' in y_unit_val: y_unit_val = y_unit_val.split(',')[0].strip()
            z_unit_val = fair_meta.get('axis_z_unit', {}).get('value', 'a.u.')
            if isinstance(z_unit_val, str) and ',' in z_unit_val: z_unit_val = z_unit_val.split(',')[0].strip()

            # Store abscissa(e)
            axis_datasets = {} # Store references to axis datasets
            if x is None:
                if y.ndim >= 1:
                    nx = y.shape[-1]
                    ds_x = data_grp.create_dataset("abscissa_x", data=np.arange(nx))
                    _try_set_h5_attr(ds_x, "units", "points")
                    _try_set_h5_attr(ds_x, "description", "X axis (index)")
                    _try_set_h5_attr(ds_x, "axis_type", "index")
                    axis_datasets['x'] = ds_x
                if y.ndim >= 2:
                    ny = y.shape[-2]
                    ds_y_ax = data_grp.create_dataset("abscissa_y", data=np.arange(ny))
                    _try_set_h5_attr(ds_y_ax, "units", "points")
                    _try_set_h5_attr(ds_y_ax, "description", "Y axis (index)")
                    _try_set_h5_attr(ds_y_ax, "axis_type", "index")
                    axis_datasets['y'] = ds_y_ax
                # Add Z if needed

            elif isinstance(x, np.ndarray): # 1D data
                ds_x = data_grp.create_dataset("abscissa_x", data=x)
                _try_set_h5_attr(ds_x, "units", x_unit_val)
                _try_set_h5_attr(ds_x, "description", f"X axis ({fair_meta.get('axis_x_unit',{}).get('description','Unknown')})")
                _try_set_h5_attr(ds_x, "axis_type", "independent_variable")
                axis_datasets['x'] = ds_x
            elif isinstance(x, list): # Multi-D data
                if len(x) >= 1 and x[0] is not None and isinstance(x[0], np.ndarray):
                    ds_x = data_grp.create_dataset("abscissa_x", data=x[0])
                    _try_set_h5_attr(ds_x, "units", x_unit_val)
                    _try_set_h5_attr(ds_x, "description", f"X axis ({fair_meta.get('axis_x_unit',{}).get('description','Unknown')})")
                    _try_set_h5_attr(ds_x, "axis_type", "independent_variable_x")
                    axis_datasets['x'] = ds_x
                if len(x) >= 2 and x[1] is not None and isinstance(x[1], np.ndarray):
                     ds_y_ax = data_grp.create_dataset("abscissa_y", data=x[1])
                     _try_set_h5_attr(ds_y_ax, "units", y_unit_val)
                     _try_set_h5_attr(ds_y_ax, "description", f"Y axis ({fair_meta.get('axis_y_unit',{}).get('description','Unknown')})")
                     _try_set_h5_attr(ds_y_ax, "axis_type", "independent_variable_y")
                     axis_datasets['y'] = ds_y_ax
                if len(x) >= 3 and x[2] is not None and isinstance(x[2], np.ndarray):
                     ds_z_ax = data_grp.create_dataset("abscissa_z", data=x[2])
                     _try_set_h5_attr(ds_z_ax, "units", z_unit_val)
                     _try_set_h5_attr(ds_z_ax, "description", f"Z axis ({fair_meta.get('axis_z_unit',{}).get('description','Unknown')})")
                     _try_set_h5_attr(ds_z_ax, "axis_type", "independent_variable_z")
                     axis_datasets['z'] = ds_z_ax

            # --- Link axes to data dimensions using HDF5 Dimension Scales API ---
            # Use explicit positive indexing and add error handling
            if 'intensity' in data_grp:
                dims = ds_y.dims
                current_ndim = ds_y.ndim # Use ndim from the actual HDF5 dataset

                # Link X dimension (last dimension, index ndim-1)
                if current_ndim >= 1 and 'x' in axis_datasets:
                    x_dim_index = current_ndim - 1
                    try:
                        dims[x_dim_index].label = 'x'
                        dims[x_dim_index].attach_scale(axis_datasets['x'])
                    except OverflowError as e_ovfl:
                         warnings.warn(f"OverflowError setting label/scale for X dim (index {x_dim_index}): {e_ovfl}. Skipping dimension link.")
                    except Exception as e:
                         warnings.warn(f"Error setting label/scale for X dim (index {x_dim_index}): {type(e).__name__} - {e}. Skipping dimension link.")

                # Link Y dimension (second to last dimension, index ndim-2)
                if current_ndim >= 2 and 'y' in axis_datasets:
                    y_dim_index = current_ndim - 2
                    try:
                        dims[y_dim_index].label = 'y'
                        dims[y_dim_index].attach_scale(axis_datasets['y'])
                    except OverflowError as e_ovfl:
                         warnings.warn(f"OverflowError setting label/scale for Y dim (index {y_dim_index}): {e_ovfl}. Skipping dimension link.")
                    except Exception as e:
                         warnings.warn(f"Error setting label/scale for Y dim (index {y_dim_index}): {type(e).__name__} - {e}. Skipping dimension link.")

                # Link Z dimension (third to last dimension, index ndim-3)
                if current_ndim >= 3 and 'z' in axis_datasets:
                    z_dim_index = current_ndim - 3
                    try:
                        dims[z_dim_index].label = 'z'
                        dims[z_dim_index].attach_scale(axis_datasets['z'])
                    except OverflowError as e_ovfl:
                         warnings.warn(f"OverflowError setting label/scale for Z dim (index {z_dim_index}): {e_ovfl}. Skipping dimension link.")
                    except Exception as e:
                         warnings.warn(f"Error setting label/scale for Z dim (index {z_dim_index}): {type(e).__name__} - {e}. Skipping dimension link.")


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
    Loads Bruker EPR data using eprload and converts it to FAIR formats
    with structured metadata.

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

    output_basename = output_path / original_file_path.stem
    print(f"Output base name: {output_basename}")

    # --- Perform Conversions based on requested formats ---
    if not output_formats:
         warnings.warn("No output formats specified. Nothing to do.")
         return

    print("\nProcessing parameters and generating outputs...")
    # Note: _process_parameters is called inside each save function now

    if 'csv_json' in output_formats:
        _save_to_csv_json(output_basename, x, y, pars, original_file_path_str)
        print("...CSV + JSON generation finished.")

    if 'hdf5' in output_formats:
         _save_to_hdf5(output_basename, x, y, pars, original_file_path_str)
         print("...HDF5 generation finished.")

    print("\nFAIR conversion process finished.")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running Bruker to FAIR Converter Example (with structured metadata) ---")
    print("This will first use eprload's dialog to select a Bruker file,")
    print("then convert it to CSV/JSON and HDF5 formats in the same directory,")
    print("applying the parameter map for richer metadata.")
    print("-" * 50)

    # Example 1: Use file dialog to select input, save to same directory
    convert_bruker_to_fair()

    # # Example 2: Specify an input file and output directory
    # input_bruker_file = "path/to/your/test_data.DSC" # <-- CHANGE THIS
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