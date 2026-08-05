"""
CSV-only MAVEN/IUVS aurora dataset builder.

What this does:
- Loops over a large orbit range, e.g. 2500--15500.
- Finds L1C and L1B files inside 100-orbit block folders.
- Processes every scan without making quicklook plots or PDFs.
- Writes one scan-level row for every usable scan, including red/non-detection scans.
- Appends to one master CSV so long runs can be resumed.

The output is scan-level, similar to the old short_csv() output, but now includes:
- detection_flag: 1 for green/yellow, 0 for red
- color_flag_muv: green/yellow/red
- n_pixels/count: number of selected pixels in that scan
- data_quality_note: explains whether selected pixels existed

Adjust DATA_ROOT, OUT_CSV, ORBIT_START, and ORBIT_END below before running.
"""

import glob
import os
import time
from datetime import datetime
from warnings import filterwarnings
import sys

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import spiceypy as spice
from astropy.io import fits
from scipy.io import readsav
from scipy.stats import pearsonr
from PyUVS.data import calculate_calibration_curve
from PyUVS.spice import load_iuvs_spice

# =========================
# User settings
# =========================

ORBIT_START = 2000
ORBIT_END = 15500         # inclusive; orbit 15500 will be attempted
VERSION = "ml_v1"

DATA_ROOT = "D:/nightglow"
L1C_ROOT = f"{DATA_ROOT}/c"
L1B_ROOT = f"{DATA_ROOT}/b"
OUT_CSV = f"{DATA_ROOT}_tables/iuvs_ml_scan_dataset.csv"

SPICE_DIR = "D:/periapse_quicklooks/spice"
TEMPLATE_DIR = "PyUVS/ancillary"

# Detection thresholds copied from the plotting script logic.
SZA_PREFILTER_MIN = 102
SZA_SELECTED_MIN = 105
CO_SNR_MIN = 4
CO2P_SNR_MIN = 3
YELLOW_R_MIN = 0.60
GREEN_R_MIN = 0.55
GREEN_R2_MIN = 0.25

# If True, skip scans already present in OUT_CSV by nick_number.
RESUME = False


# =========================
# Small helpers
# =========================

def orbit_block_folder(orbit: int) -> str:
    """Return the 100-orbit block folder name, e.g. 2500 -> '02500', 2599 -> '02500'."""
    block_start = (orbit // 100) * 100
    return f"{block_start:05d}"


def get_fits(orbit: int, channel: str = "muv"):
    """Find L1C and L1B files for one orbit in the correct 100-orbit block folder."""
    orbit_str = f"{orbit:05d}"
    block = orbit_block_folder(orbit)

    l1c_pattern = f"{L1C_ROOT}/{block}/*periapse*orbit{orbit_str}*.gz"
    l1b_pattern = f"{L1B_ROOT}/{block}/*periapse*orbit{orbit_str}-{channel}*.gz"

    l1c_files = sorted(glob.glob(l1c_pattern))
    l1b_files = sorted(glob.glob(l1b_pattern))
    return l1c_files, l1b_files


def get_dims(l1c_hdul):
    dims = l1c_hdul["spectrum_noaltbin"].data["muv"].shape
    return dims[0], dims[1], dims[2], dims[3]


def rearrange_dimensions(array):
    array = np.swapaxes(array, 1, 2)
    array = np.swapaxes(array, 0, 1)
    return array


def get_feature_indices(l1c_hdul):
    features = l1c_hdul["emission_features"].data["emission"]
    return {
        "solar": np.where(features == "Solar cont. muv")[0][0],
        "co": np.where(features == "CO Cameron")[0][0],
        "co2p": np.where(features == "CO2pUVD")[0][0],
        "o2972": np.where(features == "O2972")[0][0],
        "no": np.where(features == "NO band")[0][0],
        "o1304": np.where(features == "OI 130.4")[0][0],
        "o1356": np.where(features == "OI 135.6")[0][0],
    }


def avg_unc(unc_arr, count):
    if count <= 0 or np.all(np.isnan(unc_arr)):
        return np.nan
    return float(np.sqrt(np.nansum(unc_arr ** 2)) / count)


def safe_nanmean(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def safe_nanstd(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanstd(arr))


def parse_date_time(utc_string):
    try:
        dt = datetime.strptime(utc_string, "%Y/%j %b %d %H:%M:%S.%fUTC")
        return datetime.strftime(dt, "%Y-%b-%d"), datetime.strftime(dt, "%H:%M:%S")
    except Exception:
        return np.nan, np.nan


def local_time_from_spice(et_value, longitude_deg):
    try:
        hr, mn, sc, _, _ = spice.et2lst(
            float(et_value),
            499,  # Mars body code
            np.radians(float(longitude_deg)),
            "planetocentric",
            timlen=256,
            ampmlen=256,
        )
        return float(hr + mn / 60 + sc / 3600)
    except Exception:
        return np.nan


# =========================
# Detection logic, no plots
# =========================

def cosmic_ray_filter(const, chisq, sza):
    """Initial scan mask from baseline, chi-square, and SZA."""
    threshold_const = np.nanmean(const) + np.nanstd(const)
    threshold_chisq = np.nanmedian(chisq) + 2 * np.nanstd(chisq)

    flags = np.ones_like(sza, dtype=int)
    bad = (np.abs(const) > threshold_const) | (np.abs(chisq) > threshold_chisq) | (sza < SZA_PREFILTER_MIN)

    # This matches the original behavior: if one spatial bin in an integration is bad,
    # the whole integration is rejected.
    bad_integrations = np.unique(np.where(bad)[0])
    flags[bad_integrations, :] = 0
    return flags


def selected_pixel_mask(flags, co_rad, co_unc, co2p_rad, co2p_unc, sza):
    """Final selected-pixel mask used for averages and model features."""
    co_snr = co_rad / co_unc
    co2p_snr = co2p_rad / co2p_unc
    selected = (((co_snr > CO_SNR_MIN) | (co2p_snr > CO2P_SNR_MIN)) &
                (sza > SZA_SELECTED_MIN) &
                (flags == 1))
    return selected.astype(int)


def load_muv_templates(n_spec):
    load_spec = 256 if n_spec == 219 else n_spec
    templates = np.load(f"{TEMPLATE_DIR}/templates_{load_spec}.npy", allow_pickle=True).item()

    template_wavelength = templates.get("wavelength")
    template_solar = templates.get("solar_continuum")
    template_co = templates.get("co_cameron")
    template_co2p = templates.get("co2p_uvd")
    template_no = templates.get("no_nightglow")

    if n_spec == 219:
        start = 6
        template_wavelength = template_wavelength[start:start + n_spec]
        template_solar = template_solar[start:start + n_spec]
        template_co = template_co[start:start + n_spec]
        template_co2p = template_co2p[start:start + n_spec]
        template_no = template_no[start:start + n_spec]

    return template_wavelength, template_solar, template_co, template_co2p, template_no


def get_coeff(mlr, template_dn, dlambda, cal):
    return mlr / (np.sum(template_dn / cal) * dlambda)


def average_muv_spectrum(selected, spectra, wavelength, calibration_curve,
                         solar_rad, co_rad, co2p_rad, no_rad, const,
                         template_wavelength, template_solar, template_co, template_co2p, template_no,
                         dwavelength):
    """Return average wavelength/background/aurora fit for selected pixels.

    If there are no selected pixels, return None values. The scan row will still be
    written, but spectrum-fit metrics will be NaN.
    """
    good_int, good_spa = np.where(selected == 1)
    if len(good_int) == 0:
        return None, None, None

    solar_coeff = get_coeff(solar_rad[good_int, good_spa], template_solar, dwavelength, calibration_curve[3, :])
    co_coeff = get_coeff(co_rad[good_int, good_spa], template_co, dwavelength, calibration_curve[3, :])
    co2p_coeff = get_coeff(co2p_rad[good_int, good_spa], template_co2p, dwavelength, calibration_curve[3, :])
    no_coeff = get_coeff(no_rad[good_int, good_spa], template_no, dwavelength, calibration_curve[3, :])

    solar_component = template_solar * np.nanmean(solar_coeff)
    co_component = template_co * np.nanmean(co_coeff)
    co2p_component = template_co2p * np.nanmean(co2p_coeff)
    no_component = template_no * np.nanmean(no_coeff)
    constant = np.nanmean(const[good_int, good_spa])

    plot_wavelength = np.nanmean(wavelength[good_spa], axis=0)
    background = np.nanmean(spectra[good_int, good_spa], axis=0) - solar_component - no_component - constant
    aurora_fit = co_component + co2p_component
    return plot_wavelength, background, aurora_fit


def calculate_correlation(plot_wavelength, background, aurora_fit):
    if plot_wavelength is None or background is None or aurora_fit is None:
        return np.nan, np.nan
    try:
        minwave_ind = np.argmin(np.abs(plot_wavelength - 190))
        maxwave_ind = np.argmin(np.abs(plot_wavelength - 246))
        y_pred = aurora_fit[minwave_ind:maxwave_ind]
        y_true = background[minwave_ind:maxwave_ind]
        if len(y_true) < 2 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
            return np.nan, np.nan
        r2 = metrics.r2_score(y_true, y_pred)
        r = pearsonr(y_true, y_pred)[0]
        return float(r2), float(r)
    except Exception:
        return np.nan, np.nan


def classify_scan(n_pixels, r_square, pearson_r_value):
    """Reproduce the original red/yellow/green logic."""
    if np.isnan(pearson_r_value):
        return "red", 0
    if (pearson_r_value >= YELLOW_R_MIN) and (n_pixels <= 5) and (n_pixels >= 3):
        return "yellow", 1
    if ((r_square > GREEN_R2_MIN) or (pearson_r_value > GREEN_R_MIN)) and (n_pixels > 5):
        return "green", 1
    return "red", 0

def max_to_median(arr):
    try:
        arr = np.asarray(arr, dtype=float)

        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan

        med = np.nanmedian(arr)
        mx = np.nanmax(arr)

        if not np.isfinite(med) or med == 0 or not np.isfinite(mx):
            return np.nan

        return float(mx / med)

    except Exception:
        return np.nan

def selected_spectra_max_to_median(spectra_2d):
    try:
        arr = np.asarray(spectra_2d, dtype=float)

        if arr.size == 0:
            return np.nan, np.nan, np.nan

        if arr.ndim == 1:
            arr = arr[None, :]

        if np.all(np.isnan(arr)):
            return np.nan, np.nan, np.nan

        med = np.nanmedian(arr, axis=1)
        mx = np.nanmax(arr, axis=1)

        ratio = np.full_like(mx, np.nan, dtype=float)
        good = np.isfinite(mx) & np.isfinite(med) & (med != 0)

        ratio[good] = mx[good] / med[good]

        if np.all(np.isnan(ratio)):
            return np.nan, np.nan, np.nan

        return (
            float(np.nanmax(ratio)),
            float(np.nanmean(ratio)),
            float(np.nanmedian(ratio)),
        )

    except Exception:
        return np.nan, np.nan, np.nan

def longest_true_run(mask):
    """Return longest consecutive run of True values."""
    max_run = 0
    current = 0

    for val in mask:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return max_run


def cosmic_ray_features_1d(spectrum, threshold1=10, threshold2=20):
    """
    Cosmic-ray-like features for one spectrum.
    Uses median-relative brightness.
    """
    try:
        arr = np.asarray(spectrum, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return {
                "cr_n_bins_above_10x": np.nan,
                "cr_n_bins_above_20x": np.nan,
                "cr_max_consecutive_10x": np.nan,
                "cr_p99_to_median": np.nan,
                "cr_p95_to_median": np.nan,
                "cr_max_to_p95": np.nan,
                "cr_score": np.nan,
            }

        med = np.nanmedian(arr)

        if not np.isfinite(med) or med <= 0:
            return {
                "cr_n_bins_above_10x": np.nan,
                "cr_n_bins_above_20x": np.nan,
                "cr_max_consecutive_10x": np.nan,
                "cr_p99_to_median": np.nan,
                "cr_p95_to_median": np.nan,
                "cr_max_to_p95": np.nan,
                "cr_score": np.nan,
            }

        p95 = np.nanpercentile(arr, 95)
        p99 = np.nanpercentile(arr, 99)
        mx = np.nanmax(arr)

        high10 = arr > threshold1 * med
        high20 = arr > threshold2 * med

        n10 = int(np.sum(high10))
        n20 = int(np.sum(high20))
        longest10 = int(longest_true_run(high10))

        p99_to_median = float(p99 / med)
        p95_to_median = float(p95 / med)

        if np.isfinite(p95) and p95 > 0:
            max_to_p95 = float(mx / p95)
        else:
            max_to_p95 = np.nan

        # Simple soft score. You can tune this later.
        cr_score = (
            0.4 * min(n10 / 10, 1) +
            0.3 * min(longest10 / 5, 1) +
            0.3 * min(p99_to_median / 20, 1)
        )

        return {
            "cr_n_bins_above_10x": n10,
            "cr_n_bins_above_20x": n20,
            "cr_max_consecutive_10x": longest10,
            "cr_p99_to_median": p99_to_median,
            "cr_p95_to_median": p95_to_median,
            "cr_max_to_p95": max_to_p95,
            "cr_score": float(cr_score),
        }

    except Exception:
        return {
            "cr_n_bins_above_10x": np.nan,
            "cr_n_bins_above_20x": np.nan,
            "cr_max_consecutive_10x": np.nan,
            "cr_p99_to_median": np.nan,
            "cr_p95_to_median": np.nan,
            "cr_max_to_p95": np.nan,
            "cr_score": np.nan,
        }


def selected_spectra_cosmic_ray_features(spectra_2d):
    """
    Compute cosmic-ray features for selected pixel spectra, then summarize.
    """
    try:
        arr = np.asarray(spectra_2d, dtype=float)

        if arr.size == 0:
            return {
                "cr_n_bins_above_10x_max": np.nan,
                "cr_n_bins_above_20x_max": np.nan,
                "cr_max_consecutive_10x_max": np.nan,
                "cr_p99_to_median_max": np.nan,
                "cr_p95_to_median_max": np.nan,
                "cr_max_to_p95_max": np.nan,
                "cr_score_max": np.nan,
                "cr_score_mean": np.nan,
            }

        if arr.ndim == 1:
            arr = arr[None, :]

        features = [cosmic_ray_features_1d(spec) for spec in arr]

        def collect(key):
            vals = np.array([f[key] for f in features], dtype=float)
            if np.all(np.isnan(vals)):
                return np.nan
            return vals

        cr_score_vals = collect("cr_score")

        return {
            "cr_n_bins_above_10x_max": float(np.nanmax(collect("cr_n_bins_above_10x"))),
            "cr_n_bins_above_20x_max": float(np.nanmax(collect("cr_n_bins_above_20x"))),
            "cr_max_consecutive_10x_max": float(np.nanmax(collect("cr_max_consecutive_10x"))),
            "cr_p99_to_median_max": float(np.nanmax(collect("cr_p99_to_median"))),
            "cr_p95_to_median_max": float(np.nanmax(collect("cr_p95_to_median"))),
            "cr_max_to_p95_max": float(np.nanmax(collect("cr_max_to_p95"))),
            "cr_score_max": float(np.nanmax(cr_score_vals)),
            "cr_score_mean": float(np.nanmean(cr_score_vals)),
        }

    except Exception:
        return {
            "cr_n_bins_above_10x_max": np.nan,
            "cr_n_bins_above_20x_max": np.nan,
            "cr_max_consecutive_10x_max": np.nan,
            "cr_p99_to_median_max": np.nan,
            "cr_p95_to_median_max": np.nan,
            "cr_max_to_p95_max": np.nan,
            "cr_score_max": np.nan,
            "cr_score_mean": np.nan,
        }


# =========================
# Row creation
# =========================

def summarize_scan(orbit_number, scan_index, l1c, l1b, feature_indices, n_spec, possible_diffuse):
    """Process one scan and return a one-row dictionary for the master CSV."""
    scan = scan_index

    # Radiances and uncertainties.
    profile = l1c["radiance_noaltbin"].data["profile"]
    rand_unc = l1c["radiance_noaltbin"].data["random_unc"]

    solar_rad = profile[scan, :, :, feature_indices["solar"]]
    co_rad = profile[scan, :, :, feature_indices["co"]]
    co2p_rad = profile[scan, :, :, feature_indices["co2p"]]
    o2972_rad = profile[scan, :, :, feature_indices["o2972"]]
    o1304_rad = profile[scan, :, :, feature_indices["o1304"]]
    o1356_rad = profile[scan, :, :, feature_indices["o1356"]]
    no_rad = profile[scan, :, :, feature_indices["no"]]

    co_unc = rand_unc[scan, :, :, feature_indices["co"]]
    co2p_unc = rand_unc[scan, :, :, feature_indices["co2p"]]
    o2972_unc = rand_unc[scan, :, :, feature_indices["o2972"]]
    o1304_unc = rand_unc[scan, :, :, feature_indices["o1304"]]
    o1356_unc = rand_unc[scan, :, :, feature_indices["o1356"]]

    const = l1c["radiance_noaltbin"].data["baseline_muv"][scan]
    chisq = l1c["radiance_noaltbin"].data["chi_square_muv"][scan]

    latitude = rearrange_dimensions(l1c["pixelgeometry"].data["pixel_corner_lat"][scan])
    longitude = rearrange_dimensions(l1c["pixelgeometry"].data["pixel_corner_lon"][scan])
    altitude = rearrange_dimensions(l1c["pixelgeometry"].data["pixel_corner_mrh_alt"][scan])
    sza = l1c["pixelgeometry"].data["pixel_solar_zenith_angle"][scan].T
    et = l1c["integration"].data["et"][scan]
    et_reshaped = np.repeat(et[:, None], sza.shape[1], axis=1)

    sc_lat = l1c["spacecraftgeometry"].data["sub_spacecraft_lat"][scan]
    sc_lon = l1c["spacecraftgeometry"].data["sub_spacecraft_lon"][scan]
    sc_alt = l1c["spacecraftgeometry"].data["spacecraft_alt"][scan]
    l_s = l1c["observation"].data["SOLAR_LONGITUDE"][0]

    wavelength = np.squeeze(l1b["observation"].data["wavelength"])
    spectra = l1b["detector_dark_subtracted"].data
    calibration_curve = calculate_calibration_curve(l1b)

    flags = cosmic_ray_filter(const, chisq, sza)
    selected = selected_pixel_mask(flags, co_rad, co_unc, co2p_rad, co2p_unc, sza)
    good_int, good_spa = np.where(selected == 1)
    selected_spectra = spectra[good_int, good_spa]

    cr_features = selected_spectra_cosmic_ray_features(selected_spectra)

    muv_pixel_max_to_median_max, \
    muv_pixel_max_to_median_mean, \
    muv_pixel_max_to_median_median = selected_spectra_max_to_median(selected_spectra)
    n_pixels = int(len(good_int))

    template_wavelength, template_solar, template_co, template_co2p, template_no = load_muv_templates(n_spec)
    dwavelength = np.diff(template_wavelength)[0]
    avg_wavelength, avg_background, avg_aurora = average_muv_spectrum(
        selected, spectra, wavelength, calibration_curve,
        solar_rad, co_rad, co2p_rad, no_rad, const,
        template_wavelength, template_solar, template_co, template_co2p, template_no,
        dwavelength,
    )
    r_square, pearson_r_value = calculate_correlation(avg_wavelength, avg_background, avg_aurora)
    muv_max_to_median = max_to_median(avg_background)
    color_flag, detection_flag = classify_scan(n_pixels, r_square, pearson_r_value)

    co_cameron_scale = 1.0
    if n_spec == 200:
        co_cameron_scale = 1.2010
    if n_spec == 219:
        co_cameron_scale = 1.0029

    # Use selected pixels for the standard scan summary when possible.
    # If no pixels are selected, still write a row with scan-wide geometry and NaN feature averages.
    if n_pixels > 0:
        idx_int = good_int
        idx_spa = good_spa
        representative_int = int(np.median(idx_int))
        data_quality_note = "selected_pixels"
    else:
        idx_int = np.array([], dtype=int)
        idx_spa = np.array([], dtype=int)
        representative_int = int(len(et) // 2)
        data_quality_note = "no_selected_pixels"

    nick_number = f"{orbit_number + (scan + 1) / 100:.2f}"
    utc_string = l1c["integration"].data["utc"][scan, representative_int]
    date, utc_time = parse_date_time(utc_string)

    solar_mean = np.nan
    solar_max = np.nan
    solar_to_co = np.nan
    solar_to_co2p = np.nan
    solar_to_aurora = np.nan

    if n_pixels > 0:
        lon_values = longitude[idx_int, idx_spa, 4].astype(float)
        indbig = np.where(lon_values > 300)[0]
        indsmall = np.where(lon_values < 60)[0]
        if (len(indbig) != 0) and (len(indsmall) != 0):
            lon_values = lon_values.copy()
            lon_values[np.where(lon_values < 180)] += 360
            mean_lon = safe_nanmean(lon_values)
            if mean_lon >= 360:
                mean_lon -= 360
            if mean_lon < 0:
                mean_lon += 360
        else:
            mean_lon = safe_nanmean(lon_values)

        mean_lat = safe_nanmean(latitude[idx_int, idx_spa, 4])
        mean_alt = safe_nanmean(altitude[idx_int, idx_spa, 4])
        std_alt = safe_nanstd(altitude[idx_int, idx_spa, 4])
        std_spa = safe_nanstd(idx_spa)
        mean_et = safe_nanmean(et_reshaped[idx_int, idx_spa])
        mean_sza = safe_nanmean(sza[idx_int, idx_spa])
        mean_sc_lat = safe_nanmean(sc_lat[idx_int])
        mean_sc_lon = safe_nanmean(sc_lon[idx_int])
        mean_sc_alt = safe_nanmean(sc_alt[idx_int])
        cf_region = int((mean_lon >= 150) and (mean_lon < 210) and (mean_lat >= -60) and (mean_lat < -30))
        local_time = local_time_from_spice(mean_et, mean_lon)

        co_mean = safe_nanmean(co_rad[idx_int, idx_spa]) * co_cameron_scale
        co_unc_mean = avg_unc(co_unc[idx_int, idx_spa], n_pixels) * co_cameron_scale
        co2p_mean = safe_nanmean(co2p_rad[idx_int, idx_spa])
        co2p_unc_mean = avg_unc(co2p_unc[idx_int, idx_spa], n_pixels)
        o1304_mean = safe_nanmean(o1304_rad[idx_int, idx_spa])
        o1304_unc_mean = avg_unc(o1304_unc[idx_int, idx_spa], n_pixels)
        o1356_mean = safe_nanmean(o1356_rad[idx_int, idx_spa])
        o1356_unc_mean = avg_unc(o1356_unc[idx_int, idx_spa], n_pixels)
        o2972_mean = safe_nanmean(o2972_rad[idx_int, idx_spa])
        o2972_unc_mean = avg_unc(o2972_unc[idx_int, idx_spa], n_pixels)
        solar_mean = safe_nanmean(solar_rad[idx_int, idx_spa])
        solar_max = np.nanmax(solar_rad[idx_int, idx_spa])
        solar_to_co = solar_mean / co_mean if np.isfinite(co_mean) and co_mean != 0 else np.nan
        solar_to_co2p = solar_mean / co2p_mean if np.isfinite(co2p_mean) and co2p_mean != 0 else np.nan
        solar_to_aurora = (
            solar_mean / (co_mean + co2p_mean)
            if np.isfinite(co_mean)
            and np.isfinite(co2p_mean)
            and (co_mean + co2p_mean) != 0
            else np.nan
        )
    else:
        mean_lon = safe_nanmean(longitude[:, :, 4])
        mean_lat = safe_nanmean(latitude[:, :, 4])
        mean_alt = safe_nanmean(altitude[:, :, 4])
        std_alt = safe_nanstd(altitude[:, :, 4])
        std_spa = np.nan
        mean_et = safe_nanmean(et_reshaped)
        mean_sza = safe_nanmean(sza)
        mean_sc_lat = safe_nanmean(sc_lat)
        mean_sc_lon = safe_nanmean(sc_lon)
        mean_sc_alt = safe_nanmean(sc_alt)
        cf_region = int((mean_lon >= 150) and (mean_lon < 210) and (mean_lat >= -60) and (mean_lat < -30))
        local_time = local_time_from_spice(mean_et, mean_lon)

        co_mean = co_unc_mean = co2p_mean = co2p_unc_mean = np.nan
        o1304_mean = o1304_unc_mean = o1356_mean = o1356_unc_mean = np.nan
        o2972_mean = o2972_unc_mean = np.nan

    def snr(rad, unc):
        if np.isnan(rad) or np.isnan(unc) or unc == 0:
            return np.nan
        return float(rad / unc)

    co2p_safe = co2p_mean if not np.isnan(co2p_mean) and co2p_mean != 0 else np.nan
    discrete = 0 if orbit_number in possible_diffuse else 1

    return {
        "orbit_number": orbit_number,
        "date": date,
        "utc_time": utc_time,
        "scan": scan + 1,
        "nick_number": nick_number,
        "latitude": mean_lat,
        "longitude": mean_lon,
        "cf_region": cf_region,
        "altitude": mean_alt,
        "altitude_std": std_alt,
        "spatial_bin_std": std_spa,
        "local_time": local_time,
        "et": mean_et,
        "sza": mean_sza,
        "sc_latitude": mean_sc_lat,
        "sc_longitude": mean_sc_lon,
        "sc_altitude": mean_sc_alt,
        "solar_longitude": float(l_s),
        "O1304_radiance": o1304_mean,
        "O1304_uncertainty": o1304_unc_mean,
        "O1304_snr": snr(o1304_mean, o1304_unc_mean),
        "O1356_radiance": o1356_mean,
        "O1356_uncertainty": o1356_unc_mean,
        "O1356_snr": snr(o1356_mean, o1356_unc_mean),
        "CO_radiance": co_mean,
        "CO_uncertainty": co_unc_mean,
        "CO_snr": snr(co_mean, co_unc_mean),
        "CO2p_radiance": co2p_mean,
        "CO2p_uncertainty": co2p_unc_mean,
        "CO2p_snr": snr(co2p_mean, co2p_unc_mean),
        "O2972_radiance": o2972_mean,
        "O2972_uncertainty": o2972_unc_mean,
        "O2972_snr": snr(o2972_mean, o2972_unc_mean),
        "CO/CO2p": float(co_mean / co2p_safe) if not np.isnan(co_mean) and not np.isnan(co2p_safe) else np.nan,
        "max_to_median": muv_max_to_median,
        "pixel_max_to_median_max": muv_pixel_max_to_median_max,
        "pixel_max_to_median_mean": muv_pixel_max_to_median_mean,
        "pixel_max_to_median_median": muv_pixel_max_to_median_median,
        "count": n_pixels,
        "n_pixels": n_pixels,
        "r_square_muv": r_square,
        "pearson_r_muv": pearson_r_value,
        "pearson_r_fuv": np.nan,
        "color_flag_muv": color_flag,
        "color_flag_fuv": "red",
        "detection_flag": detection_flag,
        "cosmic_ray_check_muv": int(np.isfinite(cr_features["cr_score_max"]) and cr_features["cr_score_max"] > 0.7),
        "cr_n_bins_above_10x_max": cr_features["cr_n_bins_above_10x_max"],
        "cr_n_bins_above_20x_max": cr_features["cr_n_bins_above_20x_max"],
        "cr_max_consecutive_10x_max": cr_features["cr_max_consecutive_10x_max"],
        "cr_p99_to_median_max": cr_features["cr_p99_to_median_max"],
        "cr_p95_to_median_max": cr_features["cr_p95_to_median_max"],
        "cr_max_to_p95_max": cr_features["cr_max_to_p95_max"],
        "cr_score_max": cr_features["cr_score_max"],
        "cr_score_mean": cr_features["cr_score_mean"],
        "solar_mean": solar_mean,
        "solar_max": solar_max,
        "solar_to_CO": solar_to_co,
        "solar_to_CO2p": solar_to_co2p,
        "solar_to_aurora": solar_to_aurora,
        "discrete": discrete,
        "data_quality_note": data_quality_note,
    }


# =========================
# Main loop
# =========================

def append_rows(rows, out_csv):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=write_header, index=False)


def load_completed_nick_numbers(out_csv):
    if not RESUME or not os.path.exists(out_csv):
        return set()
    try:
        existing = pd.read_csv(out_csv, usecols=["nick_number"], dtype={"nick_number": str})
        return set(existing["nick_number"].astype(str))
    except Exception:
        return set()


def main():
    filterwarnings("ignore")
    load_iuvs_spice(spice_directory=SPICE_DIR)

    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        print(f"Deleted existing file: {OUT_CSV}")

    december2014 = np.arange(382, 453 + 1, 1)
    march2015 = [806, 810, 824, 826, 842, 844, 846, 848, 850, 852]
    september2017 = np.arange(5700, 5800 + 1, 1)
    diffuse_aurora = np.concatenate((december2014, march2015, september2017))
    possible_diffuse = np.concatenate((
        np.arange(2508, 2517 + 1),
        np.arange(3338, 3546 + 1),
        np.arange(4556, 4567 + 1),
        [6753, 4736, 3828],
        diffuse_aurora,
    ))

    completed = load_completed_nick_numbers(OUT_CSV)
    t0 = time.time()
    buffer = []

    total_orbits = ORBIT_END - ORBIT_START + 1
    bar_length = 40

    for i, orbit_number in enumerate(range(ORBIT_START, ORBIT_END + 1)):

        progress = (i + 1) / total_orbits
        elapsed = time.time() - t0
        eta = elapsed / progress - elapsed

        filled = int(bar_length * progress)
        bar = "█" * filled + "-" * (bar_length - filled)

        sys.stdout.write(
            f"\r[{bar}] {progress*100:5.1f}% "
            f"Orbit {orbit_number} | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"ETA: {eta/60:.1f} min"
        )
        sys.stdout.flush()
        
        l1c_files, l1b_files = get_fits(orbit_number, "muv")
        if not l1c_files or not l1b_files:
            continue

        try:
            with fits.open(l1c_files[-1], ignore_missing_simple=True, ignore_missing_end=True) as l1c:
                # Fast skip for fully dayside orbits, copied from the original logic.
                orbit_sza = l1c["pixelgeometry"].data["pixel_solar_zenith_angle"]
                if np.nanmax(orbit_sza) < 112:
                    continue

                n_scans, n_int, n_spa, n_spec = get_dims(l1c)
                if len(l1b_files) < n_scans:
                    print(f"\nOrbit {orbit_number}: only {len(l1b_files)} L1B files for {n_scans} scans; skipping incomplete orbit")
                    continue

                feature_indices = get_feature_indices(l1c)

                for scan_index in range(n_scans):
                    nick = f"{orbit_number + (scan_index + 1) / 100:.2f}"
                    if nick in completed:
                        continue

                    try:
                        with fits.open(l1b_files[scan_index], ignore_missing_simple=True, ignore_missing_end=True) as l1b:
                            row = summarize_scan(
                                orbit_number,
                                scan_index,
                                l1c,
                                l1b,
                                feature_indices,
                                n_spec,
                                possible_diffuse,
                            )
                            buffer.append(row)
                            completed.add(nick)
                    except Exception as exc:
                        print(f"\nOrbit {orbit_number}, scan {scan_index + 1}: skipped because {exc}")
                        continue

                    if len(buffer) >= 200:
                        append_rows(buffer, OUT_CSV)
                        buffer.clear()
        except Exception as exc:
            print(f"\nOrbit {orbit_number}: skipped because {exc}")
            continue

    append_rows(buffer, OUT_CSV)
    print(f"\nDone. Master CSV saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
