import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits

MAIN_DIR = "D:/nightglow/c/"

def rearrange_dimensions(array):
    array = np.swapaxes(array, 1, 2)
    array = np.swapaxes(array, 0, 1)
    return array

def circular_mean_deg(lon_deg):
    lon_deg = np.asarray(lon_deg)
    good = np.isfinite(lon_deg)
    if not np.any(good):
        return np.nan

    lon_rad = np.radians(lon_deg[good])
    x = np.mean(np.cos(lon_rad))
    y = np.mean(np.sin(lon_rad))
    ang = np.degrees(np.arctan2(y, x)) % 360
    return ang

def get_orbit_number_from_filename(filepath):
    name = os.path.basename(filepath)
    # expects something like: mvn_iuv_l1c_periapse-orbit02512_...
    part = name.split("orbit")[1][:5]
    return int(part)

rows = []

# find all l1c periapse files in subfolders like 02500, 02600, etc.
file_list = sorted(glob.glob(os.path.join(MAIN_DIR, "*", "*periapse*orbit*.fits.gz")))
total_files = len(file_list)

for i, filepath in enumerate(file_list):
    try:
        hdul = fits.open(filepath)

        orbit_number = get_orbit_number_from_filename(filepath)
        percent = (i + 1) / total_files * 100
        print(f"\rProcessing orbit {orbit_number} ({i+1}/{total_files}) [{percent:.1f}%]", end="")

        dims = hdul["spectrum_noaltbin"].data["muv"].shape
        n_scans = dims[0]
        n_int = dims[1]
        n_spa = dims[2]
        mid_spa = n_spa // 2
        median_int = n_int // 2

        for scan in range(n_scans):
            # geometry
            latitude = rearrange_dimensions(
                hdul["pixelgeometry"].data["pixel_corner_lat"][scan]
            )
            longitude = rearrange_dimensions(
                hdul["pixelgeometry"].data["pixel_corner_lon"][scan]
            )
            altitude = rearrange_dimensions(
                hdul["pixelgeometry"].data["pixel_corner_mrh_alt"][scan]
            )

            sza = hdul["pixelgeometry"].data["pixel_solar_zenith_angle"][scan].T
            local_time = hdul["pixelgeometry"].data["local_time"][scan].T

            # spacecraft geometry
            sc_lat = hdul["spacecraftgeometry"].data["sub_spacecraft_lat"][scan]
            sc_lon = hdul["spacecraftgeometry"].data["sub_spacecraft_lon"][scan]
            sc_alt = hdul["spacecraftgeometry"].data["spacecraft_alt"][scan]

            # timing
            utc = hdul["integration"].data["utc"][scan]
            et = hdul["integration"].data["et"][scan]

            # optional observation-level quantity
            solar_longitude = hdul["observation"].data["SOLAR_LONGITUDE"][0]

            row = {
                "orbit_number": orbit_number,
                "scan": scan + 1,
                "utc": utc[median_int],
                "et_mean": float(np.nanmean(et)),
                "latitude": float(np.nanmean(latitude[:, mid_spa, 4])),
                "longitude": float(circular_mean_deg(longitude[:, mid_spa, 4])),
                "altitude": float(np.nanmean(altitude[:, mid_spa, 4])),
                "sza": float(np.nanmean(sza[:, mid_spa])),
                "local_time": float(np.nanmean(local_time[:, mid_spa])),
                "sc_latitude": float(np.nanmean(sc_lat)),
                "sc_longitude": float(circular_mean_deg(sc_lon)),
                "sc_altitude": float(np.nanmean(sc_alt)),
                "solar_longitude": float(solar_longitude),
                "min_altitude_midspa": float(np.nanmin(altitude[:, mid_spa, 4])),
                "max_altitude_midspa": float(np.nanmax(altitude[:, mid_spa, 4])),
            }

            rows.append(row)

        hdul.close()

    except Exception as e:
        print(f"Skipping {filepath}: {e}")

df = pd.DataFrame(rows)
df.to_csv("all_scans.csv", index=False)
print("Saved all_scans.csv")