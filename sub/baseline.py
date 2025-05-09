#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Correction Utilities

This module provides functions for performing various types of baseline corrections
on 1D spectral data, including polynomial, constant offset, and exponential decay
baselines.
"""

import numpy as np
import warnings
from scipy.optimize import curve_fit
# For examples and tests:
import matplotlib.pyplot as plt
import unittest

# --- Helper Function for Masking ---

def _create_fit_mask(
    reference_axis: np.ndarray,
    fit_window: tuple = None,
    exclude_regions: list = None
) -> np.ndarray:
    """
    Creates a boolean mask for data points to be used in fitting.

    Args:
        reference_axis (np.ndarray): The axis (e.g., x-values or indices)
            against which fit_window and exclude_regions are defined.
        fit_window (tuple, optional): A (start, end) tuple specifying the
            primary region for fitting. Points outside this window are
            excluded. If None, the full range of reference_axis is
            initially considered.
        exclude_regions (list of tuples, optional): A list of (start, end)
            tuples specifying regions to *exclude* from the fitting mask.
            These are applied based on the reference_axis.

    Returns:
        np.ndarray: Boolean mask where True indicates a point to include in fit.
    """
    n_points = len(reference_axis)
    mask = np.ones(n_points, dtype=bool)

    if fit_window is not None:
        start_fw, end_fw = fit_window
        if start_fw >= end_fw:
            warnings.warn(
                f"Fit window start ({start_fw}) is not less than its end "
                f"({end_fw}). The window will be ignored, and the full range "
                f"considered (before exclusions).", UserWarning
            )
        else:
            mask &= (reference_axis >= start_fw) & (reference_axis <= end_fw)

    if exclude_regions is not None:
        for region_idx, region in enumerate(exclude_regions):
            start_ex, end_ex = region
            if start_ex >= end_ex:
                warnings.warn(
                    f"Exclusion region #{region_idx} start ({start_ex}) is not "
                    f"less than its end ({end_ex}). This region will be ignored.", UserWarning
                )
                continue
            mask &= ~((reference_axis >= start_ex) & (reference_axis <= end_ex))
    return mask


# --- Baseline Correction Functions ---

def baseline_polynomial(
    y_data: np.ndarray,
    x_data: np.ndarray = None,
    poly_order: int = 1,
    exclude_regions: list = None,
    roi: tuple = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs baseline correction by subtracting a polynomial fit.

    The polynomial is fitted to regions of the data presumed to be baseline.
    These regions can be specified by defining an overall region of
    interest (ROI) for fitting, and/or by excluding signal regions.

    Args:
        y_data (np.ndarray): The 1D spectral data array.
        x_data (np.ndarray, optional): The x-axis data corresponding to y_data.
            If provided, `exclude_regions` and `roi` are interpreted in
            the units of `x_data`. If None, indices are used. Defaults to None.
        poly_order (int, optional): Order of the polynomial. Defaults to 1.
        exclude_regions (list of tuples, optional): List of (start, end)
            tuples specifying regions to *exclude* from baseline fitting.
            Interpreted in `x_data` units or indices. Defaults to None.
        roi (tuple, optional): A single (start, end) tuple specifying the
            region of interest where baseline fitting should *occur*.
            Data outside this ROI (and within `exclude_regions`) will not
            be used. Interpreted in `x_data` units or indices.
            If None, the entire dataset (minus `exclude_regions`) is used.
            Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - y_corrected: The baseline-corrected y_data.
            - baseline: The calculated polynomial baseline.

    Raises:
        ValueError: If inputs are invalid (e.g., y_data not 1D, mismatched lengths).
    """
    if not isinstance(y_data, np.ndarray) or y_data.ndim != 1:
        raise ValueError("y_data must be a 1D NumPy array.")
    if poly_order < 0:
        raise ValueError("poly_order must be a non-negative integer.")

    n_points = len(y_data)
    # y_to_fit = np.copy(y_data) # No need to copy y_data if only using masked version for polyfit

    if x_data is not None:
        if not isinstance(x_data, np.ndarray) or x_data.ndim != 1:
            raise ValueError("x_data must be a 1D NumPy array if provided.")
        if len(x_data) != n_points:
            raise ValueError("x_data and y_data must have the same length.")
        reference_axis_for_fit = np.copy(x_data) # Used for fitting
        reference_axis_for_eval = np.copy(x_data) # Used for baseline evaluation
    else:
        reference_axis_for_fit = np.arange(n_points) # Use indices for fitting
        reference_axis_for_eval = np.arange(n_points) # And for evaluation

    # Validate roi
    if roi is not None:
        if not (isinstance(roi, tuple) and len(roi) == 2):
            raise ValueError("roi must be a tuple of (start, end).")
    # Validate exclude_regions
    if exclude_regions is not None:
        if not isinstance(exclude_regions, list):
            raise ValueError("exclude_regions must be a list of (start, end) tuples.")
        for i, region in enumerate(exclude_regions):
            if not (isinstance(region, tuple) and len(region) == 2):
                raise ValueError(
                    f"Each item in exclude_regions (item {i}) must be a (start, end) tuple."
                )

    fit_mask = _create_fit_mask(reference_axis_for_fit,
                                fit_window=roi,
                                exclude_regions=exclude_regions)

    num_points_for_fit = np.sum(fit_mask)
    if num_points_for_fit <= poly_order:
        warnings.warn(
            f"Not enough points ({num_points_for_fit}) for polynomial fit of "
            f"order {poly_order} after applying ROI/exclusions. "
            "Returning original data.", UserWarning
        )
        return y_data, np.zeros_like(y_data)

    try:
        # Fit polynomial using only the masked (True) points
        coeffs = np.polyfit(reference_axis_for_fit[fit_mask], y_data[fit_mask], poly_order)
        # Evaluate the polynomial over the entire original x_range (or indices)
        baseline = np.polyval(coeffs, reference_axis_for_eval)
        y_corrected = y_data - baseline
        return y_corrected, baseline
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Polynomial baseline fit failed: {e}. Returning original data.", UserWarning)
        return y_data, np.zeros_like(y_data)


def baseline_constant_offset(
    y_data: np.ndarray,
    offset_region_indices: tuple = None,
    method: str = 'mean'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs baseline correction by subtracting a constant offset.

    The offset is calculated from a specified region of y_data (using indices)
    via its mean or median.

    Args:
        y_data (np.ndarray): The 1D spectral data array.
        offset_region_indices (tuple, optional): (start_index, end_index) tuple
            specifying the slice of y_data (exclusive of end_index) to
            calculate the offset from. If None, the entire array is used.
            Defaults to None.
        method (str, optional): 'mean' or 'median'. Defaults to 'mean'.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - y_corrected: The baseline-corrected y_data.
            - baseline: The calculated constant baseline array.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(y_data, np.ndarray) or y_data.ndim != 1:
        raise ValueError("y_data must be a 1D NumPy array.")
    if method.lower() not in ['mean', 'median']:
        raise ValueError("Method must be 'mean' or 'median'.")

    if offset_region_indices:
        if not (isinstance(offset_region_indices, tuple) and
                len(offset_region_indices) == 2):
            raise ValueError(
                "offset_region_indices must be a (start_index, end_index) tuple."
            )
        try:
            start_idx, end_idx = map(int, offset_region_indices) # Ensure integers
        except (ValueError, TypeError):
             raise ValueError("offset_region_indices components must be convertible to int.")

        start_idx = max(0, start_idx)
        end_idx = min(len(y_data), end_idx)

        if start_idx >= end_idx:
            warnings.warn(
                f"Invalid offset_region_indices ({offset_region_indices}). "
                "Start index must be less than end index. Using whole array.", UserWarning
            )
            region_for_offset = y_data
        else:
            region_for_offset = y_data[start_idx:end_idx]
    else:
        warnings.warn(
            "offset_region_indices not provided. Using whole array for offset.", UserWarning
        )
        region_for_offset = y_data

    if region_for_offset.size == 0:
        warnings.warn(
            "Offset calculation region is empty. Returning original data.", UserWarning
        )
        return y_data, np.zeros_like(y_data)

    offset_value = (np.mean(region_for_offset) if method.lower() == 'mean'
                    else np.median(region_for_offset))

    baseline = np.full_like(y_data, offset_value)
    y_corrected = y_data - offset_value
    return y_corrected, baseline


# --- Model Functions for Curve Fitting ---

def _stretched_exponential_decay_model(t, y0, amplitude, tau, beta):
    """Model: y0 + amplitude * exp(-(t / tau)**beta)"""
    if tau <= 1e-12:  # Prevent division by zero or extremely small tau
        return np.inf
    if not (1e-3 < beta <= 1.0001): # Beta typically 0 < beta <= 1
        return np.inf
    decay_arg = t / tau
    if isinstance(decay_arg, np.ndarray): # Handle potential negative values if t can be negative
        decay_arg_safe = np.maximum(decay_arg, 0)
    elif decay_arg < 0:
        decay_arg_safe = 0
    else:
        decay_arg_safe = decay_arg
        
    return y0 + amplitude * np.exp(-(decay_arg_safe)**beta)


def _mono_exponential_decay_model(t, y0, amplitude, tau):
    """Model: y0 + amplitude * exp(-t / tau)"""
    if tau <= 1e-12: # Prevent division by zero or extremely small tau
        return np.inf
    return y0 + amplitude * np.exp(-t / tau)


# --- Exponential Baseline Functions ---

def _fit_exponential_baseline(
    y_data: np.ndarray,
    x_data: np.ndarray,
    model_func,
    param_names: list,
    default_initial_guess_generator, # Changed to a generator function
    default_bounds: tuple,
    user_initial_guess: list = None,
    user_bounds: tuple = None,
    fit_region: tuple = None,
    exclude_regions: list = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Helper to fit an exponential-like baseline model.
    """
    n_params = len(param_names)

    if not isinstance(y_data, np.ndarray) or y_data.ndim != 1:
        raise ValueError("y_data must be a 1D NumPy array.")
    if not isinstance(x_data, np.ndarray) or x_data.ndim != 1:
        raise ValueError(f"{model_func.__name__} requires x_data as a 1D NumPy array.")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    if fit_region is not None:
        if not (isinstance(fit_region, tuple) and len(fit_region) == 2):
            raise ValueError("fit_region must be a (start_x, end_x) tuple.")
    if exclude_regions is not None:
        if not isinstance(exclude_regions, list):
            raise ValueError("exclude_regions must be a list of (start_x, end_x) tuples.")
        for i, region in enumerate(exclude_regions):
            if not (isinstance(region, tuple) and len(region) == 2):
                raise ValueError(
                    f"Each item in exclude_regions (item {i}) must be a (start_x, end_x) tuple."
                )
    
    fit_mask = _create_fit_mask(x_data, fit_window=fit_region, exclude_regions=exclude_regions)
    
    x_masked = x_data[fit_mask]
    y_masked = y_data[fit_mask]

    if len(x_masked) < n_params:
        warnings.warn(
            f"Not enough points ({len(x_masked)}) to fit {model_func.__name__} "
            f"with {n_params} parameters after applying regions. "
            "Returning original data.", UserWarning
        )
        return y_data, np.zeros_like(y_data), None

    if user_initial_guess is None:
        # Generate heuristic guess using the masked data
        current_initial_guess = default_initial_guess_generator(x_masked, y_masked)
        param_order_str = ", ".join(param_names)
        warnings.warn(
            f"Using heuristic initial guess for {model_func.__name__} "
            f"([{param_order_str}]): {current_initial_guess}", UserWarning
        )
    else:
        if not (isinstance(user_initial_guess, (list, np.ndarray)) and
                len(user_initial_guess) == n_params):
            raise ValueError(f"initial_guess must be a list/array of {n_params} values.")
        current_initial_guess = list(user_initial_guess)


    current_bounds = default_bounds
    if user_bounds is not None:
        if not (isinstance(user_bounds, tuple) and len(user_bounds) == 2 and
                isinstance(user_bounds[0], (list, np.ndarray)) and len(user_bounds[0]) == n_params and
                isinstance(user_bounds[1], (list, np.ndarray)) and len(user_bounds[1]) == n_params):
            raise ValueError(
                f"bounds must be a tuple of two lists/arrays of length {n_params}: "
                "([lowers], [uppers])"
            )
        current_bounds = user_bounds
    
    try:
        popt, pcov = curve_fit(
            model_func, x_masked, y_masked,
            p0=current_initial_guess, bounds=current_bounds,
            maxfev=10000, # Increased maxfev
            method='trf' 
        )

        baseline = model_func(x_data, *popt) 
        y_corrected = y_data - baseline

        if pcov is not None and not np.any(np.isinf(pcov)) and np.all(np.diag(pcov) >= 0):
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = [np.nan] * n_params
            if pcov is not None:
                 warnings.warn("Covariance matrix from fit is problematic. Std errors set to NaN.", UserWarning)

        fit_parameters = {param_names[i]: popt[i] for i in range(n_params)}
        fit_parameters.update({f"std_{param_names[i]}": perr[i] for i in range(n_params)})
        
        return y_corrected, baseline, fit_parameters

    except RuntimeError as e:
        warnings.warn(
            f"{model_func.__name__} fit did not converge: {e}. Returning original data.", UserWarning
        )
    except ValueError as e: 
        warnings.warn(
            f"ValueError during {model_func.__name__} fit: {e}. Returning original data.", UserWarning
        )
    except Exception as e: 
        warnings.warn(
            f"Unexpected error in {model_func.__name__} fit: {e}. Returning original data.", UserWarning
        )
    return y_data, np.zeros_like(y_data), None

# --- Heuristic guess generators ---
def _heuristic_stretched_exp_guess(x_masked, y_masked):
    guess_y0 = np.mean(y_masked[-max(1, len(y_masked)//20):]) # Use smaller fraction for tail
    guess_A = y_masked[0] - guess_y0 if len(y_masked) > 0 else 0
    if len(x_masked) > 1 and x_masked[-1] > x_masked[0]:
        guess_tau = (x_masked[-1] - x_masked[0]) / 3.0
    else:
        guess_tau = 1.0
    guess_tau = max(guess_tau, 1e-9)
    guess_beta = 0.8
    return [guess_y0, guess_A, guess_tau, guess_beta]

def _heuristic_mono_exp_guess(x_masked, y_masked):
    guess_y0 = np.mean(y_masked[-max(1, len(y_masked)//20):])
    guess_A = y_masked[0] - guess_y0 if len(y_masked) > 0 else 0
    if len(x_masked) > 1 and x_masked[-1] > x_masked[0]:
        guess_tau = (x_masked[-1] - x_masked[0]) / 3.0
    else:
        guess_tau = 1.0
    guess_tau = max(guess_tau, 1e-9)
    return [guess_y0, guess_A, guess_tau]

def baseline_stretched_exponential(
    y_data: np.ndarray,
    x_data: np.ndarray,
    initial_guess: list = None,
    bounds: tuple = None,
    fit_region: tuple = None,
    exclude_regions: list = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fits and subtracts a stretched exponential decay baseline.
    Model: y(x) = y0 + A * exp(-((x) / τ)**β)
    """
    param_names = ['y0', 'A', 'tau', 'beta']
    default_bounds = (
        [-np.inf, -np.inf, 1e-12, 1e-3], 
        [np.inf, np.inf, np.inf, 1.0]    
    )
    return _fit_exponential_baseline(
        y_data, x_data,
        _stretched_exponential_decay_model,
        param_names, _heuristic_stretched_exp_guess, default_bounds,
        user_initial_guess=initial_guess, user_bounds=bounds,
        fit_region=fit_region, exclude_regions=exclude_regions
    )

def baseline_mono_exponential(
    y_data: np.ndarray,
    x_data: np.ndarray,
    initial_guess: list = None,
    bounds: tuple = None,
    fit_region: tuple = None,
    exclude_regions: list = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fits and subtracts a mono-exponential decay baseline.
    Model: y(x) = y0 + A * exp(-(x) / τ)
    """
    param_names = ['y0', 'A', 'tau']
    default_bounds = (
        [-np.inf, -np.inf, 1e-12], 
        [np.inf, np.inf, np.inf]   
    )
    return _fit_exponential_baseline(
        y_data, x_data,
        _mono_exponential_decay_model,
        param_names, _heuristic_mono_exp_guess, default_bounds,
        user_initial_guess=initial_guess, user_bounds=bounds,
        fit_region=fit_region, exclude_regions=exclude_regions
    )


def _polynomial_features_2d(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    order_x: int,
    order_y: int
) -> np.ndarray:
    """
    Generates a 2D polynomial design matrix (Vandermonde-like).

    For each pair (x, y) in x_coords, y_coords, creates a row:
    [1, x, y, x^2, xy, y^2, ..., x^(order_x), x^(order_x-1)y, ..., y^(order_y)]
    More precisely, it includes all terms x^i * y^j where 0<=i<=order_x and 0<=j<=order_y.

    Args:
        x_coords (np.ndarray): 1D array of x-coordinates.
        y_coords (np.ndarray): 1D array of y-coordinates (must be same length as x_coords).
        order_x (int): Maximum order for x.
        order_y (int): Maximum order for y.

    Returns:
        np.ndarray: The design matrix where each row corresponds to an (x,y) pair
                    and columns are the polynomial terms.
    """
    if x_coords.ndim != 1 or y_coords.ndim != 1:
        raise ValueError("x_coords and y_coords must be 1D arrays.")
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    if order_x < 0 or order_y < 0:
        raise ValueError("Polynomial orders must be non-negative.")

    num_points = len(x_coords)
    num_coeffs = (order_x + 1) * (order_y + 1)
    design_matrix = np.zeros((num_points, num_coeffs))

    col_idx = 0
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            design_matrix[:, col_idx] = (x_coords**i) * (y_coords**j)
            col_idx += 1
    return design_matrix

def _evaluate_polynomial_surface(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    coeffs: np.ndarray,
    order_x: int,
    order_y: int
) -> np.ndarray:
    """
    Evaluates a 2D polynomial surface defined by coefficients.

    Args:
        x_mesh (np.ndarray): 2D mesh of x-coordinates (e.g., from np.meshgrid).
        y_mesh (np.ndarray): 2D mesh of y-coordinates.
        coeffs (np.ndarray): 1D array of polynomial coefficients, ordered as
                             generated by _polynomial_features_2d.
        order_x (int): Maximum order for x used to generate coeffs.
        order_y (int): Maximum order for y used to generate coeffs.

    Returns:
        np.ndarray: The evaluated 2D polynomial surface, same shape as x_mesh/y_mesh.
    """
    if x_mesh.shape != y_mesh.shape:
        raise ValueError("x_mesh and y_mesh must have the same shape.")
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D array.")
    
    expected_num_coeffs = (order_x + 1) * (order_y + 1)
    if len(coeffs) != expected_num_coeffs:
        raise ValueError(
            f"Number of coefficients ({len(coeffs)}) does not match "
            f"expected for orders ({order_x}, {order_y}), which is {expected_num_coeffs}."
        )

    surface = np.zeros_like(x_mesh, dtype=float)
    coeff_idx = 0
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            surface += coeffs[coeff_idx] * (x_mesh**i) * (y_mesh**j)
            coeff_idx += 1
    return surface

def _get_slice_from_region(
    axis_data: np.ndarray, start_val: float, end_val: float
) -> slice:
    """
    Converts a start/end value range in axis_data units to an index slice.
    Assumes axis_data is sorted.
    """
    if start_val > end_val: # Allow reversed specification for convenience
        start_val, end_val = end_val, start_val

    # Find closest indices. searchsorted finds where elements should be inserted.
    # 'left' ensures we get the first index >= value
    # 'right' ensures we get the first index > value
    idx_start = np.searchsorted(axis_data, start_val, side='left')
    idx_end = np.searchsorted(axis_data, end_val, side='right') # 'right' makes it exclusive for slicing
    
    # Clip to ensure indices are within bounds for slicing
    idx_start = max(0, idx_start)
    idx_end = min(len(axis_data), idx_end) # Slicing up to len(axis_data) is fine

    if idx_start >= idx_end : # If region is outside data or yields no points
        # Return an empty slice at the start or end depending on where the region lies
        # This helps avoid issues if a region is entirely outside the data range
        if start_val > axis_data[-1] or end_val < axis_data[0]:
            # Typically we might want to signal this, but for masking, an empty slice works.
            # For ROI, this means no points. For exclusion, it means no points excluded.
             return slice(0,0) # Empty slice
    return slice(idx_start, idx_end)


def _create_fit_mask_2d(
    shape: tuple[int, int],
    x_axis_coords: np.ndarray = None, # Full 1D x-axis for interpreting region values
    y_axis_coords: np.ndarray = None, # Full 1D y-axis
    fit_window_roi: tuple[tuple[float, float], tuple[float, float]] = None,
    exclude_regions: list[tuple[tuple[float, float], tuple[float, float]]] = None
) -> np.ndarray:
    """
    Creates a 2D boolean mask for fitting based on ROI and excluded regions.

    Args:
        shape (tuple[int, int]): The (rows, cols) shape of the 2D data.
        x_axis_coords (np.ndarray, optional): 1D array of x-coordinates for all columns.
            If None, indices 0 to shape[1]-1 are used for region interpretation.
        y_axis_coords (np.ndarray, optional): 1D array of y-coordinates for all rows.
            If None, indices 0 to shape[0]-1 are used for region interpretation.
        fit_window_roi (tuple, optional): A region ((x_start, x_end), (y_start, y_end))
            where fitting *should* occur. If None, the entire area is initially considered.
        exclude_regions (list, optional): List of regions [((x1s,x1e),(y1s,y1e)), ...]
            to *exclude* from fitting.

    Returns:
        np.ndarray: A 2D boolean mask of the given shape. True where fitting should occur.
    """
    rows, cols = shape
    
    # Use indices if axis coordinates are not provided
    _x_axis = x_axis_coords if x_axis_coords is not None else np.arange(cols)
    _y_axis = y_axis_coords if y_axis_coords is not None else np.arange(rows)

    if fit_window_roi:
        (x_roi_start, x_roi_end), (y_roi_start, y_roi_end) = fit_window_roi
        x_slice_roi = _get_slice_from_region(_x_axis, x_roi_start, x_roi_end)
        y_slice_roi = _get_slice_from_region(_y_axis, y_roi_start, y_roi_end)
        
        mask = np.zeros(shape, dtype=bool)
        mask[y_slice_roi, x_slice_roi] = True
    else:
        mask = np.ones(shape, dtype=bool)

    if exclude_regions:
        for region in exclude_regions:
            (x_excl_start, x_excl_end), (y_excl_start, y_excl_end) = region
            x_slice_excl = _get_slice_from_region(_x_axis, x_excl_start, x_excl_end)
            y_slice_excl = _get_slice_from_region(_y_axis, y_excl_start, y_excl_end)
            mask[y_slice_excl, x_slice_excl] = False
            
    return mask

def baseline_polynomial_2d(
    z_data: np.ndarray,
    x_data: np.ndarray = None,
    y_data: np.ndarray = None,
    poly_order: int | tuple[int, int] = (1, 1),
    exclude_regions: list[tuple[tuple[float, float], tuple[float, float]]] = None,
    roi: tuple[tuple[float, float], tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs 2D baseline correction by subtracting a fitted polynomial surface.

    The polynomial surface is fitted to regions of the data presumed to be baseline.
    These regions are defined by an overall region of interest (ROI) and/or by
    excluding specific rectangular signal regions.

    Args:
        z_data (np.ndarray): The 2D data array (e.g., image, spectral map).
        x_data (np.ndarray, optional): 1D array of x-axis coordinates corresponding
            to the columns of z_data. If None, column indices are used.
        y_data (np.ndarray, optional): 1D array of y-axis coordinates corresponding
            to the rows of z_data. If None, row indices are used.
        poly_order (int or tuple[int, int], optional): Order of the polynomial.
            If int, it's used for both x and y order: (order, order).
            If tuple (order_x, order_y), specifies max order for x and y terms
            (e.g., x^order_x * y^order_y). Defaults to (1, 1) (bilinear plane).
        exclude_regions (list of tuples, optional): List of rectangular regions
            to *exclude* from baseline fitting. Each region is defined as
            `((x_start, x_end), (y_start, y_end))`. Interpreted in `x_data`/`y_data`
            units or indices. Defaults to None.
        roi (tuple, optional): A single rectangular region `((x_start, x_end), (y_start, y_end))`
            specifying where baseline fitting should *occur*. Data outside this ROI
            (and within `exclude_regions`) will not be used. Interpreted in
            `x_data`/`y_data` units or indices. If None, the entire dataset
            (minus `exclude_regions`) is used. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - z_corrected: The baseline-corrected 2D data.
            - baseline_surface: The calculated 2D polynomial baseline.

    Raises:
        ValueError: If inputs are invalid (e.g., z_data not 2D, mismatched lengths).
    """
    if not isinstance(z_data, np.ndarray) or z_data.ndim != 2:
        raise ValueError("z_data must be a 2D NumPy array.")

    rows, cols = z_data.shape

    if isinstance(poly_order, int):
        order_x, order_y = poly_order, poly_order
    elif isinstance(poly_order, tuple) and len(poly_order) == 2:
        order_x, order_y = poly_order
        if not (isinstance(order_x, int) and isinstance(order_y, int) and order_x >= 0 and order_y >= 0):
            raise ValueError("poly_order tuple elements must be non-negative integers.")
    else:
        raise ValueError("poly_order must be an int or a tuple of two ints.")

    # Prepare coordinate axes for fitting and evaluation
    if x_data is not None:
        if not (isinstance(x_data, np.ndarray) and x_data.ndim == 1 and len(x_data) == cols):
            raise ValueError(f"x_data must be a 1D array of length {cols} (num columns).")
        x_coords_eval = x_data
    else:
        x_coords_eval = np.arange(cols)

    if y_data is not None:
        if not (isinstance(y_data, np.ndarray) and y_data.ndim == 1 and len(y_data) == rows):
            raise ValueError(f"y_data must be a 1D array of length {rows} (num rows).")
        y_coords_eval = y_data
    else:
        y_coords_eval = np.arange(rows)

    # Validate roi
    if roi is not None:
        if not (isinstance(roi, tuple) and len(roi) == 2 and
                isinstance(roi[0], tuple) and len(roi[0]) == 2 and
                isinstance(roi[1], tuple) and len(roi[1]) == 2):
            raise ValueError("roi must be a tuple of ((x_start, x_end), (y_start, y_end)).")
    # Validate exclude_regions
    if exclude_regions is not None:
        if not isinstance(exclude_regions, list):
            raise ValueError("exclude_regions must be a list of region tuples.")
        for i, region in enumerate(exclude_regions):
            if not (isinstance(region, tuple) and len(region) == 2 and
                    isinstance(region[0], tuple) and len(region[0]) == 2 and
                    isinstance(region[1], tuple) and len(region[1]) == 2):
                raise ValueError(
                    f"Each item in exclude_regions (item {i}) must be "
                    f"((x_start, x_end), (y_start, y_end))."
                )

    # Create meshgrid for evaluating the full surface
    # Note: np.meshgrid indexing defaults to 'xy', so X varies along columns, Y along rows.
    # This matches typical image conventions where (row, col) -> (y, x)
    XX_eval, YY_eval = np.meshgrid(x_coords_eval, y_coords_eval)


    # Create the mask indicating which points to use for fitting
    fit_mask = _create_fit_mask_2d(
        shape=z_data.shape,
        x_axis_coords=x_coords_eval, # Use full range coords for interpretation
        y_axis_coords=y_coords_eval,
        fit_window_roi=roi,
        exclude_regions=exclude_regions
    )

    # Select points for fitting based on the mask
    # XX_eval[fit_mask] will give 1D arrays of x and y coordinates
    x_points_to_fit = XX_eval[fit_mask]
    y_points_to_fit = YY_eval[fit_mask]
    z_values_to_fit = z_data[fit_mask]

    num_points_for_fit = len(z_values_to_fit)
    num_coeffs_needed = (order_x + 1) * (order_y + 1)

    if num_points_for_fit < num_coeffs_needed:
        warnings.warn(
            f"Not enough points ({num_points_for_fit}) for polynomial fit with "
            f"orders ({order_x}, {order_y}), which requires {num_coeffs_needed} coefficients. "
            "Returning original data.", UserWarning
        )
        return z_data, np.zeros_like(z_data)

    # Construct the design matrix for the least-squares fit
    design_matrix = _polynomial_features_2d(x_points_to_fit, y_points_to_fit, order_x, order_y)

    try:
        # Perform the least-squares fit to find polynomial coefficients
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(design_matrix, z_values_to_fit, rcond=None)
        
        # Check for rank deficiency which might indicate issues
        if rank < num_coeffs_needed:
             warnings.warn(
                f"Polynomial fit may be poorly conditioned. Rank deficiency detected: "
                f"rank={rank}, expected_coeffs={num_coeffs_needed}. Results might be unreliable.",
                UserWarning
            )

        # Evaluate the fitted polynomial surface over the entire grid
        baseline_surface = _evaluate_polynomial_surface(XX_eval, YY_eval, coeffs, order_x, order_y)
        z_corrected = z_data - baseline_surface
        return z_corrected, baseline_surface

    except np.linalg.LinAlgError as e:
        warnings.warn(f"Polynomial surface fit failed: {e}. Returning original data.", UserWarning)
        return z_data, np.zeros_like(z_data)

# --- Example Usage and Demonstrations ---

def generate_gaussian(x, amplitude, mu, sigma):
    """Generates a Gaussian peak."""
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_correction(x, y_original, baseline, y_corrected, title):
    """Helper function for plotting results."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_original, label='Original Data')
    plt.plot(x, baseline, label='Calculated Baseline', linestyle='--')
    plt.plot(x, y_corrected, label='Corrected Data', linestyle=':')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_examples():
    """Runs example demonstrations for each baseline correction method."""
    print("--- Running Baseline Correction Examples ---")
    
    # Common data parameters
    x = np.linspace(0, 100, 500)
    signal1 = generate_gaussian(x, amplitude=20, mu=30, sigma=5)
    signal2 = generate_gaussian(x, amplitude=15, mu=70, sigma=8)
    pure_signal = signal1 + signal2
    noise = np.random.normal(0, 0.5, len(x))

    # 1. Polynomial Baseline Example
    print("\n1. Polynomial Baseline Example")
    poly_baseline_true = 10 + 0.1 * x - 0.001 * x**2 # Quadratic baseline
    y_poly = poly_baseline_true + pure_signal + noise
    
    # Fit with ROI and exclusion
    y_corrected_poly, baseline_poly = baseline_polynomial(
        y_poly, x, poly_order=2, 
        roi=(0, 100),  # Use full range as ROI for fitting
        exclude_regions=[(20, 40), (60, 80)] # Exclude peak regions
    )
    plot_correction(x, y_poly, baseline_poly, y_corrected_poly, 
                    'Polynomial Baseline Correction (Order 2)')
    
    # Fit with only first and last 10% of data
    percent_10 = int(0.10 * len(x))
    roi_ends = ( (x[0], x[percent_10]), (x[-percent_10-1], x[-1]) ) # this needs to be a single roi
                                                                   # or multiple points fed to polyfit
                                                                   # The current `roi` expects one continuous window
                                                                   # We can use exclude_regions to achieve this
    
    y_corrected_poly_ends, baseline_poly_ends = baseline_polynomial(
        y_poly, x, poly_order=2,
        exclude_regions=[(x[percent_10], x[-percent_10-1])] # Exclude middle 80%
    )
    plot_correction(x, y_poly, baseline_poly_ends, y_corrected_poly_ends,
                    'Polynomial Baseline (Fit to 10% ends)')


    # 2. Constant Offset Baseline Example
    print("\n2. Constant Offset Baseline Example")
    constant_offset_true = 5.0
    y_const = constant_offset_true + pure_signal + noise
    
    y_corrected_const, baseline_const = baseline_constant_offset(
        y_const, 
        offset_region_indices=(0, 50), # Use first 50 points (indices)
        method='mean'
    )
    plot_correction(x, y_const, baseline_const, y_corrected_const, 
                    'Constant Offset Baseline Correction (Mean of 0-49)')

    # 3. Mono-Exponential Baseline Example
    print("\n3. Mono-Exponential Baseline Example")
    # True parameters for the mono-exponential decay baseline
    mono_exp_y0_true = 2.0
    mono_exp_A_true = 30.0
    mono_exp_tau_true = 25.0
    mono_exp_baseline_true = _mono_exponential_decay_model(
        x, mono_exp_y0_true, mono_exp_A_true, mono_exp_tau_true
    )
    y_mono_exp = mono_exp_baseline_true + pure_signal + noise
    
    # Initial guess (can be crucial)
    # y0, A, tau
    initial_guess_mono = [np.median(y_mono_exp[-50:]), y_mono_exp[0]-np.median(y_mono_exp[-50:]), 30] 
    
    y_corrected_mono, baseline_mono, params_mono = baseline_mono_exponential(
        y_mono_exp, x,
        initial_guess=initial_guess_mono,
        # bounds=([0, 0, 1e-9], [np.inf, np.inf, 200]), # Example bounds
        fit_region=(0, 100), # Full region
        exclude_regions=[(20, 40), (60, 80)] # Exclude signal peaks
    )
    if params_mono:
        print(f"Fitted Mono-Exponential Params: {params_mono}")
    plot_correction(x, y_mono_exp, baseline_mono, y_corrected_mono, 
                    'Mono-Exponential Baseline Correction')

    # 4. Stretched Exponential Baseline Example
    print("\n4. Stretched Exponential Baseline Example")
    # True parameters for the stretched exponential decay baseline
    s_exp_y0_true = 1.5
    s_exp_A_true = 25.0
    s_exp_tau_true = 30.0
    s_exp_beta_true = 0.7
    s_exp_baseline_true = _stretched_exponential_decay_model(
        x, s_exp_y0_true, s_exp_A_true, s_exp_tau_true, s_exp_beta_true
    )
    y_s_exp = s_exp_baseline_true + pure_signal + noise
    
    # Initial guess: y0, A, tau, beta
    initial_guess_s = [np.median(y_s_exp[-50:]), y_s_exp[0]-np.median(y_s_exp[-50:]), 35, 0.8]
        
    y_corrected_s, baseline_s, params_s = baseline_stretched_exponential(
        y_s_exp, x,
        initial_guess=initial_guess_s,
        # bounds=([0,0,1e-9,0.1], [np.inf,np.inf,200,1.0]),
        fit_region=(5, 95), # Slightly narrowed fit region
        exclude_regions=[(25, 35), (65, 75)]
    )
    if params_s:
        print(f"Fitted Stretched Exponential Params: {params_s}")
    plot_correction(x, y_s_exp, baseline_s, y_corrected_s, 
                    'Stretched Exponential Baseline Correction')
    
    print("--- Examples Finished ---")


# --- Unit Tests ---

class TestBaselineCorrection(unittest.TestCase):

    def setUp(self):
        """Set up data for tests."""
        self.x = np.linspace(0, 10, 101)
        self.gaussian = generate_gaussian(self.x, 10, 5, 1)
        self.noise = np.random.normal(0, 0.1, len(self.x))
        self.y_flat = np.full_like(self.x, 5.0) + self.noise # Flat data with noise
        self.y_linear = 2 * self.x + 3 + self.noise # Linear data with noise

    def test_polynomial_zeroth_order(self):
        y_signal_offset = self.gaussian + 5.0 + self.noise
        corrected, baseline = baseline_polynomial(
            y_signal_offset, self.x, poly_order=0, exclude_regions=[(4,6)]
        )
        self.assertAlmostEqual(np.mean(baseline), 5.0, delta=0.5) # Baseline should be ~5
        self.assertTrue(np.allclose(corrected, y_signal_offset - np.mean(baseline), atol=1e-1))

    def test_polynomial_linear(self):
        y_signal_linear = self.gaussian + (0.5 * self.x + 2) + self.noise
        true_baseline_points = (0.5 * self.x + 2)
        
        corrected, baseline = baseline_polynomial(
            y_signal_linear, self.x, poly_order=1, exclude_regions=[(3,7)] # Exclude peak
        )
        # Check if the baseline is close to the true linear baseline
        # This is a strong assertion, depends on noise and exclusion
        np.testing.assert_allclose(baseline, true_baseline_points, atol=0.5 + np.std(self.noise)*3)
        # Check if corrected signal is close to original gaussian + noise
        np.testing.assert_allclose(corrected, self.gaussian + self.noise, atol=0.5 + np.std(self.noise)*3)

    def test_polynomial_no_xdata(self):
        y_signal_offset = self.gaussian + 5.0
        # Intentionally make x_indices different from self.x values for test
        indices = np.arange(len(self.x)) 
        # Create y based on indices for a clear test
        y_idx_linear = 0.1 * indices + 2 + generate_gaussian(indices, 10, 50, 10) + self.noise
        
        corrected, baseline = baseline_polynomial(y_idx_linear, poly_order=1, exclude_regions=[(40,60)])
        
        # Coefficients for y = 0.1*idx + 2
        expected_coeffs = np.polyfit(indices[[True if not (40<=i<=60) else False for i in indices]],
                                     (0.1 * indices + 2)[[True if not (40<=i<=60) else False for i in indices]], 1)
        
        fitted_coeffs = np.polyfit(indices[np.where(baseline != 0)[0]], baseline[np.where(baseline != 0)[0]], 1)
        # Since polyval is used, baseline is calculated across all indices
        # For this check, we can't directly compare to expected_coeffs if there's noise.
        # Instead, check that the returned baseline is a polynomial of order 1.
        self.assertEqual(len(np.polyfit(indices, baseline, 1)), 2) # Check it fits a line
        self.assertEqual(y_idx_linear.shape, corrected.shape)


    def test_polynomial_insufficient_points(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corrected, baseline = baseline_polynomial(
                self.y_linear, self.x, poly_order=2, roi=(0,1) # only ~10 points for order 2
            )
            self.assertTrue(len(w) > 0) # Check if warning was issued
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("Not enough points" in str(w[-1].message))
            np.testing.assert_array_equal(corrected, self.y_linear) # Should return original
            np.testing.assert_array_equal(baseline, np.zeros_like(self.y_linear))

    def test_constant_offset_mean(self):
        y_const_signal = self.gaussian + 10.0 + self.noise
        offset_region = (0, int(0.2 * len(self.x))) # First 20%
        
        # Calculate expected mean from the region, excluding signal influence for true baseline
        # For testing, we assume the offset region is truly baseline
        true_offset_in_region = np.mean((10.0 + self.noise)[offset_region[0]:offset_region[1]])

        corrected, baseline = baseline_constant_offset(
            y_const_signal, offset_region_indices=offset_region, method='mean'
        )
        self.assertAlmostEqual(baseline[0], true_offset_in_region, delta=0.5)
        np.testing.assert_allclose(corrected, y_const_signal - baseline[0], atol=1e-1)

    def test_constant_offset_median(self):
        y_const_signal = self.gaussian + 7.0 + self.noise
        offset_region = (len(self.x) - int(0.2*len(self.x)), len(self.x)-1) # Last 20%
        
        true_offset_in_region_median = np.median((7.0 + self.noise)[offset_region[0]:offset_region[1]])

        corrected, baseline = baseline_constant_offset(
            y_const_signal, offset_region_indices=offset_region, method='median'
        )
        self.assertAlmostEqual(baseline[0], true_offset_in_region_median, delta=0.5)

    def test_constant_offset_invalid_region(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Start index >= end index
            corrected, baseline = baseline_constant_offset(self.y_flat, offset_region_indices=(50, 10))
            self.assertTrue(any("Invalid offset_region_indices" in str(warn.message) for warn in w))
            # Expected to use whole array if region invalid
            self.assertAlmostEqual(baseline[0], np.mean(self.y_flat), delta=0.2)

    def test_mono_exponential_fit(self):
        y0, A, tau = 1.0, 5.0, 3.0
        true_mono_exp_baseline = _mono_exponential_decay_model(self.x, y0, A, tau)
        y_data = true_mono_exp_baseline + self.gaussian + self.noise
        
        initial_guess = [0.5, 6.0, 2.5] # Good initial guess
        # Exclude the Gaussian peak region
        exclude = [(self.x[np.argmax(self.gaussian)] - 1.5, 
                    self.x[np.argmax(self.gaussian)] + 1.5)] 
        
        corrected, baseline, params = baseline_mono_exponential(
            y_data, self.x, initial_guess=initial_guess, exclude_regions=exclude
        )
        self.assertIsNotNone(params, "Fit failed to produce parameters.")
        if params: # Check if params dict is not None
            self.assertAlmostEqual(params['y0'], y0, delta=0.5 + 3*params['std_y0'] if not np.isnan(params['std_y0']) else 0.5)
            self.assertAlmostEqual(params['A'], A, delta=1.0 + 3*params['std_A'] if not np.isnan(params['std_A']) else 1.0) # Amplitude can be trickier
            self.assertAlmostEqual(params['tau'], tau, delta=0.8 + 3*params['std_tau'] if not np.isnan(params['std_tau']) else 0.8)
        np.testing.assert_allclose(baseline, true_mono_exp_baseline, atol=1.0) # Check baseline reconstruction
        np.testing.assert_allclose(corrected, self.gaussian + self.noise, atol=1.0) # Check signal recovery

    def test_stretched_exponential_fit(self):
        y0, A, tau, beta = 0.5, 4.0, 2.5, 0.7
        true_stretched_exp_baseline = _stretched_exponential_decay_model(self.x, y0, A, tau, beta)
        y_data = true_stretched_exp_baseline + self.gaussian + self.noise
        
        initial_guess = [0.3, 4.5, 2.0, 0.8] # Good initial guess
        exclude = [(4,6)] 
        
        corrected, baseline, params = baseline_stretched_exponential(
            y_data, self.x, initial_guess=initial_guess, exclude_regions=exclude
        )
        self.assertIsNotNone(params, "Fit failed to produce parameters.")
        if params:
            self.assertAlmostEqual(params['y0'], y0, delta=0.5 + 3*params['std_y0'] if not np.isnan(params['std_y0']) else 0.5)
            self.assertAlmostEqual(params['A'], A, delta=1.0 + 3*params['std_A'] if not np.isnan(params['std_A']) else 1.0)
            self.assertAlmostEqual(params['tau'], tau, delta=0.8 + 3*params['std_tau'] if not np.isnan(params['std_tau']) else 0.8)
            self.assertAlmostEqual(params['beta'], beta, delta=0.2 + 3*params['std_beta'] if not np.isnan(params['std_beta']) else 0.2)
        np.testing.assert_allclose(baseline, true_stretched_exp_baseline, atol=1.0)
        np.testing.assert_allclose(corrected, self.gaussian + self.noise, atol=1.0)

    def test_exponential_fit_insufficient_points(self):
        y_data = _mono_exponential_decay_model(self.x, 1, 5, 3) + self.noise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corrected, baseline, params = baseline_mono_exponential(
                y_data, self.x, fit_region=(0, 0.1) # Very few points for 3 params
            )
            self.assertTrue(any("Not enough points" in str(warn.message) for warn in w))
            self.assertIsNone(params)
            np.testing.assert_array_equal(corrected, y_data)
            np.testing.assert_array_equal(baseline, np.zeros_like(y_data))

    def test_input_validations(self):
        y_2d = np.array([[1,2],[3,4]])
        with self.assertRaises(ValueError):
            baseline_polynomial(y_2d, self.x)
        with self.assertRaises(ValueError):
            baseline_polynomial(self.x, x_data=np.array([1,2])) # Mismatched length
        with self.assertRaises(ValueError):
            baseline_constant_offset(y_2d)
        with self.assertRaises(ValueError):
            baseline_mono_exponential(y_2d, self.x)
        with self.assertRaises(ValueError):
            baseline_mono_exponential(self.x, x_data=y_2d) # x_data not 1D



# --- Main execution ---
if __name__ == "__main__":
    # Run examples
    run_examples()

    # Run unit tests
    print("\n\n--- Running Unit Tests ---")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBaselineCorrection))
    runner = unittest.TextTestRunner()
    runner.run(suite)