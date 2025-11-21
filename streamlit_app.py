# streamlit_app.py
"""
Streamlit app — Full NDVI crop damage pipeline (Option A)
- Upload a GeoPackage (.gpkg)
- Authenticate Earth Engine (interactive)
- Run full pipeline (CSP mask, addBands paired sampling, MAD/IQR severity)
- Precompute NDVI charts (fortnightly)
- Build Folium map (single polygon FeatureGroup) + LayerControl
- Search CaseID -> zoom to polygon
- Download CSV results
Notes:
 - This is the heavy pipeline (same logic as your Colab). Runs may take time depending on polygon count.
 - For production, consider service-account EE auth to avoid interactive login.
"""

import os
import tempfile
import shutil
import time
import math
from io import BytesIO
import base64

import streamlit as st
from streamlit import caching

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import mapping

import folium
from streamlit_folium import st_folium

import ee
import geemap.foliumap as geemap_tile
import matplotlib.pyplot as plt

# ---------------------------
# TOP: user-configurable parameters (edit here)
# ---------------------------
st.set_page_config(layout="wide", page_title="Crop Damage NDVI — Full Pipeline")

# IO
DEFAULT_SAMPLE_LIMIT = 3000
CHART_DIR_BASE = "/tmp/ndvi_charts"   # will be created per-run
EXPORT_CSV_NAME = "gee_damage_results_FINAL_RS.csv"
EXPORT_MAP_NAME = "FINAL_CROP_DAMAGE_MAP.html"

# Date windows
PRE_START_DEFAULT = '2025-08-01'
PRE_END_DEFAULT   = '2025-11-20'
POST_OFFSET_START_DAYS_DEFAULT = 1
POST_OFFSET_END_DAYS_DEFAULT   = 10

# Phenology
CHART_START_DEFAULT = '2025-05-15'
CHART_END_DEFAULT   = '2025-11-20'
FREQ_DAYS_DEFAULT = 15

# Algorithm thresholds (keep your established logic)
CLEAR_THRESHOLD_DEFAULT = 0.8
DIFF_THRESHOLD_DEFAULT = 0.16
K_MAD = 2
IQR_FACTOR = 1.5
SLOPE_FACTOR = 150

# Spatial
SCALE_DEFAULT = 10
MAXPIX = int(1e7)
BUFFER_KM_DEFAULT = 20

# Performance
GETINFO_RETRIES = 3
GETINFO_WAIT = 1.2

# ---------------------------
# UI: sidebar config
# ---------------------------
st.sidebar.title("Pipeline settings")
st.sidebar.markdown("Adjust these if needed (defaults are tuned for Sentinel-2).")

PRE_START = st.sidebar.text_input("Pre-event start (YYYY-MM-DD)", PRE_START_DEFAULT)
PRE_END   = st.sidebar.text_input("Pre-event end (YYYY-MM-DD)", PRE_END_DEFAULT)
POST_OFFSET_START_DAYS = st.sidebar.number_input("Post offset start days", value=POST_OFFSET_START_DAYS_DEFAULT, step=1)
POST_OFFSET_END_DAYS   = st.sidebar.number_input("Post offset end days", value=POST_OFFSET_END_DAYS_DEFAULT, step=1)

CHART_START = st.sidebar.text_input("Chart start (YYYY-MM-DD)", CHART_START_DEFAULT)
CHART_END   = st.sidebar.text_input("Chart end (YYYY-MM-DD)", CHART_END_DEFAULT)
FREQ_DAYS = st.sidebar.number_input("Chart period (days)", value=FREQ_DAYS_DEFAULT, step=1)

CLEAR_THRESHOLD = st.sidebar.slider("Cloud Score Plus (CSP) clear threshold", 0.0, 1.0, CLEAR_THRESHOLD_DEFAULT, 0.05)
DIFF_THRESHOLD = st.sidebar.number_input("Fixed NDVI difference threshold", value=DIFF_THRESHOLD_DEFAULT, step=0.01, format="%.3f")
SAMPLE_LIMIT = st.sidebar.number_input("Sample limit (pixels per polygon)", value=DEFAULT_SAMPLE_LIMIT, step=500)

SCALE = st.sidebar.number_input("Sampling scale (m)", value=SCALE_DEFAULT, step=1)
BUFFER_KM = st.sidebar.number_input("ROI buffer (km)", value=BUFFER_KM_DEFAULT, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:** This runs a heavy Earth Engine workflow — expect minutes for many polygons.")

# ---------------------------
# Helpers & EE wrappers
# ---------------------------
def safe_get_info(obj, retries=GETINFO_RETRIES, wait=GETINFO_WAIT):
    """Retry wrapper for getInfo()"""
    for attempt in range(1, retries+1):
        try:
            return obj.getInfo()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(wait * attempt)

def png_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def convert_3d_to_2d_geojson(geojson_obj):
    if geojson_obj is None or not isinstance(geojson_obj, dict):
        return geojson_obj
    t = geojson_obj.get('type')
    if t == 'Polygon':
        geojson_obj['coordinates'] = [[list(coord[:2]) for coord in ring] for ring in geojson_obj['coordinates']]
    elif t == 'MultiPolygon':
        geojson_obj['coordinates'] = [[[list(coord[:2]) for coord in ring] for ring in poly] for poly in geojson_obj['coordinates']]
    elif t == 'LineString':
        geojson_obj['coordinates'] = [list(coord[:2]) for coord in geojson_obj['coordinates']]
    elif t == 'MultiLineString':
        geojson_obj['coordinates'] = [[list(coord[:2]) for coord in line] for line in geojson_obj['coordinates']]
    elif t == 'Point':
        geojson_obj['coordinates'] = list(geojson_obj['coordinates'][:2])
    elif t == 'MultiPoint':
        geojson_obj['coordinates'] = [list(coord[:2]) for coord in geojson_obj['coordinates']]
    return geojson_obj

# ---------------------------
# EE auth block
# ---------------------------
st.header("Earth Engine authentication")
ee_auth_col, ee_info_col = st.columns([2,3])

with ee_auth_col:
    if st.button("Authenticate Earth Engine"):
        try:
            ee.Authenticate()   # interactive flow
            ee.Initialize()
            st.success("Earth Engine authenticated and initialized.")
        except Exception as e:
            st.error(f"EE authenticate failed: {e}")

# Show status
try:
    ee.Initialize()
    ee_user = ee.data.getAssetRoots()  # quick call to ensure initialized
    ee_info_col.success("Earth Engine: initialized")
except Exception as e:
    ee_info_col.warning("Earth Engine not initialized. Click Authenticate and follow the OAuth flow.")

st.markdown("---")

# ---------------------------
# File upload: geopackage
# ---------------------------
st.header("Upload GeoPackage (.gpkg)")
uploaded = st.file_uploader("Upload your polygons GeoPackage (.gpkg)", type=["gpkg"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload a .gpkg to start the analysis (or authenticate EE first).")
    st.stop()

# Save uploaded file to a temp path
tmp_dir = tempfile.mkdtemp(prefix="cropdamage_")
uploaded_path = os.path.join(tmp_dir, uploaded.name)
with open(uploaded_path, "wb") as f:
    f.write(uploaded.getbuffer())

st.success(f"Saved uploaded file to {uploaded_path}")
st.write("Reading GeoPackage...")

# ---------------------------
# Phase B: read geopackage -> gdf_wgs
# ---------------------------
try:
    gdf = gpd.read_file(uploaded_path)
except Exception as e:
    st.error(f"Failed to read uploaded gpkg: {e}")
    st.stop()

# Ensure id column present
POLYGON_ID_COL = POLYGON_ID_COL  # from top-of-file default; user can change in code
if POLYGON_ID_COL not in gdf.columns:
    st.warning(f"Column '{POLYGON_ID_COL}' not found in uploaded gpkg. Using first column as ID.")
    POLYGON_ID_COL = gdf.columns[0]

# Ensure loss date column presence; if not, create empty
LOSS_DATE_COL_INPUT = st.sidebar.text_input("Loss date column name (if present)", "Date of Lo")
if LOSS_DATE_COL_INPUT in gdf.columns:
    gdf[LOSS_DATE_COL_INPUT] = pd.to_datetime(gdf[LOSS_DATE_COL_INPUT], errors='coerce')
else:
    # create empty column to avoid KeyError later
    gdf[LOSS_DATE_COL_INPUT] = pd.NaT

gdf_wgs = gdf.to_crs(epsg=4326)
st.write(f"Loaded {len(gdf_wgs)} polygons (WGS84). Preview:")
st.dataframe(gdf_wgs.head())

# ---------------------------
# Phase C: Build S2 collections & pre_max_ndvi (server-side)
# ---------------------------
st.header("Building Sentinel-2 collections (server-side) — Phase C")
try:
    ee.Initialize()
except Exception:
    # try authenticate instruction
    st.error("Earth Engine not initialized. Click 'Authenticate Earth Engine' and sign in.")
    st.stop()

buffer_deg = BUFFER_KM / 111.32
minx, miny, maxx, maxy = gdf_wgs.total_bounds
roi_rect = ee.Geometry.Rectangle([minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg])

# Build S2 & CSP collections for the analysis period (use PRE window + a bit)
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate(PRE_START, PRE_END)
      .filterBounds(roi_rect))

csPlus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
          .filterDate(PRE_START, PRE_END)
          .filterBounds(roi_rect))

# join
join = ee.Join.saveFirst('cs')
filter_eq = ee.Filter.equals(leftField='system:index', rightField='system:index')
s2_with_cs = ee.ImageCollection(join.apply(s2, csPlus, filter_eq))

def mask_clouds_with_csp(img, use_csp=True, csp_threshold=CLEAR_THRESHOLD):
    img = ee.Image(img)
    bn = img.bandNames()
    scl_mask = ee.Algorithms.If(
        bn.contains('SCL'),
        img.select('SCL').neq(3).And(img.select('SCL').neq(8)).And(img.select('SCL').neq(9)).And(img.select('SCL').neq(10)),
        ee.Image(1)
    )
    qa_mask = ee.Algorithms.If(
        bn.contains('QA60'),
        img.select('QA60').bitwiseAnd(1 << 10).eq(0).And(img.select('QA60').bitwiseAnd(1 << 11).eq(0)),
        ee.Image(1)
    )
    combined_mask = ee.Image(scl_mask).And(ee.Image(qa_mask))

    if use_csp:
        cs_img = ee.Algorithms.If(img.get('cs'), ee.Image(img.get('cs')), None)
        cs_mask = ee.Algorithms.If(
            ee.Algorithms.IsEqual(cs_img, None),
            ee.Image(1),
            ee.Algorithms.If(
                ee.Image(cs_img).bandNames().contains('cs_cdf'),
                ee.Image(cs_img).select('cs_cdf').gte(csp_threshold),
                ee.Image(1)
            )
        )
        combined_mask = combined_mask.And(ee.Image(cs_mask))

    return img.updateMask(combined_mask).copyProperties(img, img.propertyNames())

def add_ndvi(img):
    img = ee.Image(img)
    bn = img.bandNames()
    ndvi_img = ee.Algorithms.If(
        bn.contains('B8'),
        ee.Algorithms.If(
            bn.contains('B4'),
            img.normalizedDifference(['B8', 'B4']).rename('ndvi'),
            ee.Image.constant(0).rename('ndvi').updateMask(ee.Image.constant(0))
        ),
        ee.Image.constant(0).rename('ndvi').updateMask(ee.Image.constant(0))
    )
    return img.addBands(ee.Image(ndvi_img))

# Apply masks & NDVI
s2_all = s2_with_cs.map(lambda img: mask_clouds_with_csp(img)).map(add_ndvi)

# Pre-event max NDVI
pre_collection = s2_all.filterDate(PRE_START, PRE_END)
pre_max_ndvi = pre_collection.select('ndvi').max().rename('pre_max_ndvi')

# quick info
try:
    total_images = safe_get_info(s2_all.size())
except Exception:
    total_images = "unknown"

st.write(f"Built S2 collection (approx clean images = {total_images})")

# ---------------------------
# Phase D: paired-sampling helper (server-side addBands+sample)
# ---------------------------
def sample_paired_pixels(pre_img, post_img, ee_geom, sample_limit=SAMPLE_LIMIT, scale=SCALE):
    """
    Returns (pre_vals, post_vals) — lists matched by pixel order.
    Uses ee.Image.sample on a stacked image to preserve pixel alignment.
    """
    # handle empty post (no bands) -> sample pre only
    try:
        post_bands = safe_get_info(ee.Image(post_img).bandNames())
    except Exception:
        post_bands = []
    if not post_bands:
        samp = ee.Image(pre_img).rename('pre').sample(region=ee_geom, scale=scale, numPixels=sample_limit, seed=42)
        try:
            pre_arr = safe_get_info(samp.aggregate_array('pre')) or []
        except Exception:
            pre_arr = []
        pre_vals = [float(v) for v in pre_arr if v is not None and isinstance(v, (int, float))]
        return pre_vals, []

    # create paired image
    paired = ee.Image.cat(ee.Image(pre_img).rename('pre'), ee.Image(post_img).rename('post'))
    fc = paired.sample(region=ee_geom, scale=scale, numPixels=sample_limit, seed=42)

    try:
        pre_arr = safe_get_info(fc.aggregate_array('pre')) or []
        post_arr = safe_get_info(fc.aggregate_array('post')) or []
    except Exception:
        # fallback sampling separately
        pre_arr = safe_get_info(ee.Image(pre_img).sample(region=ee_geom, scale=scale, numPixels=min(sample_limit,1000), seed=43).aggregate_array(pre_img.bandNames().get(0))) or []
        post_arr = safe_get_info(ee.Image(post_img).sample(region=ee_geom, scale=scale, numPixels=min(sample_limit,1000), seed=44).aggregate_array(post_img.bandNames().get(0))) or []

    pre_vals = [float(v) for v in pre_arr if v is not None and isinstance(v, (int, float))]
    post_vals = [float(v) for v in post_arr if v is not None and isinstance(v, (int, float))]
    n = min(len(pre_vals), len(post_vals))
    return pre_vals[:n], post_vals[:n]

# ---------------------------
# Phase E: severity metrics (same logic)
# ---------------------------
def compute_severity_and_metrics(pre_vec, post_vec, scale=SCALE):
    pre = np.array(pre_vec, dtype=float)
    post = np.array(post_vec, dtype=float)
    n = min(pre.size, post.size)
    if n == 0:
        return {
            "fixed_ha": 0.0, "mad_ha": 0.0, "iqr_ha": 0.0,
            "RS_AffectedArea": 0.0, "LogicFlag": "No matched pixels",
            "RS_Severity": None, "RS_Severity_Affected": None,
            "mean_relative_severity_pct": None, "std_relative_severity_pct": None,
            "mean_normalized_severity_pct": None, "std_normalized_severity_pct": None,
            "mean_scaled_severity_pct": None, "std_scaled_severity_pct": None,
            "mean_severity_affected_relative": None, "std_severity_affected_relative": None
        }

    pre_m = pre[:n]
    post_m = post[:n]

    median_pre = float(np.nanmedian(pre_m)) if np.isfinite(np.nanmedian(pre_m)) else 0.0
    mad = float(np.nanmedian(np.abs(pre_m - median_pre))) if np.isfinite(np.nanmedian(pre_m - median_pre)) else 0.0
    q1 = float(np.nanpercentile(pre_m, 25)) if pre_m.size > 0 else 0.0
    q3 = float(np.nanpercentile(pre_m, 75)) if pre_m.size > 0 else 0.0
    iqr = q3 - q1

    spread = 1.2 * (mad * 1.4826)
    thr_mad = median_pre - K_MAD * spread if np.isfinite(median_pre) else np.nan
    thr_iqr = q1 - IQR_FACTOR * iqr if np.isfinite(q1) else np.nan

    fixed_mask = (pre_m - post_m) >= DIFF_THRESHOLD
    mad_mask = np.full_like(post_m, False, dtype=bool) if np.isnan(thr_mad) else (post_m <= thr_mad)
    iqr_mask = np.full_like(post_m, False, dtype=bool) if np.isnan(thr_iqr) else (post_m <= thr_iqr)

    fixed_count = int(np.nansum(fixed_mask))
    mad_count = int(np.nansum(mad_mask))
    iqr_count = int(np.nansum(iqr_mask))

    ha_per_pixel = (scale * scale) / 10000.0
    fixed_ha = round(fixed_count * ha_per_pixel, 4)
    mad_ha = round(mad_count * ha_per_pixel, 4)
    iqr_ha = round(iqr_count * ha_per_pixel, 4)
    total_ha = round(n * ha_per_pixel, 4)

    with np.errstate(divide='ignore', invalid='ignore'):
        sev_rel = np.clip((pre_m - post_m) / pre_m, 0, None)
    sev_rel = np.nan_to_num(sev_rel, nan=0.0, posinf=0.0)
    sev_rel_pct = np.clip(sev_rel * 100, 0, 100)

    sev_norm_med = np.clip((pre_m - post_m) / median_pre, 0, None) if median_pre != 0 else np.zeros_like(pre_m)
    sev_norm_med_pct = np.clip(sev_norm_med * 100, 0, 100)

    sev_scaled = SLOPE_FACTOR * np.clip(pre_m - post_m, 0, None)
    sev_scaled_pct = np.clip(sev_scaled, 0, 100)

    mean_rel_all = float(np.nanmean(sev_rel_pct))
    std_rel_all = float(np.nanstd(sev_rel * 100))
    mean_norm_all = float(np.nanmean(sev_norm_med_pct))
    std_norm_all = float(np.nanstd(sev_norm_med * 100))
    mean_scaled_all = float(np.nanmean(sev_scaled_pct))
    std_scaled_all = float(np.nanstd(sev_scaled))

    affected_mask = fixed_mask | mad_mask | iqr_mask
    if np.sum(affected_mask) > 0:
        mean_rel_aff = float(np.nanmean(sev_rel_pct[affected_mask]))
        std_rel_aff = float(np.nanstd(sev_rel[affected_mask] * 100))
        mean_norm_aff = float(np.nanmean(sev_norm_med_pct[affected_mask]))
        std_norm_aff = float(np.nanstd(sev_norm_med[affected_mask] * 100))
        mean_scaled_aff = float(np.nanmean(sev_scaled_pct[affected_mask]))
        std_scaled_aff = float(np.nanstd(sev_scaled[affected_mask]))
    else:
        mean_rel_aff = std_rel_aff = mean_norm_aff = std_norm_aff = mean_scaled_aff = std_scaled_aff = None

    def format_sev(mean, std):
        if mean is None or np.isnan(mean):
            return None
        if mean >= 100:
            return "100"
        else:
            return f"{int(round(mean))} [+/- {int(round(std))}]"

    RS_Severity_all = format_sev(mean_rel_all, std_rel_all)
    RS_Severity_affected = format_sev(mean_rel_aff, std_rel_aff) if mean_rel_aff is not None else None

    fixed_ha_val = fixed_ha if not np.isnan(fixed_ha) else 0
    mad_ha_val = mad_ha if not np.isnan(mad_ha) else 0
    iqr_ha_val = iqr_ha if not np.isnan(iqr_ha) else 0

    if total_ha == 0:
        final_aff_area = 0.0
        logic_flag = "No matched pixels"
    elif fixed_ha_val >= 0.85 * total_ha:
        final_aff_area = fixed_ha_val
        logic_flag = "Fixed_HA Dominant"
    elif all(x < 0.1 * total_ha for x in [fixed_ha_val, mad_ha_val, iqr_ha_val]):
        final_aff_area = max(fixed_ha_val, mad_ha_val, iqr_ha_val)
        logic_flag = "All Low (<10%)"
    elif mad_ha_val >= 0.9 * total_ha and iqr_ha_val >= 0.9 * total_ha:
        final_aff_area = round((0.9 * total_ha + 0.9 * total_ha) / 2, 4)
        logic_flag = "MAD & IQR High (90%)"
    elif abs(mad_ha_val - iqr_ha_val) > 0.5 * total_ha:
        final_aff_area = max(fixed_ha_val, mad_ha_val, iqr_ha_val)
        logic_flag = "Diverse Results"
    elif 0.3 * total_ha <= mad_ha_val <= 0.9 * total_ha and 0.3 * total_ha <= iqr_ha_val <= 0.9 * total_ha:
        final_aff_area = round((mad_ha_val + iqr_ha_val) / 2, 4)
        logic_flag = "MAD & IQR Moderate"
    elif mad_ha_val < 0.3 * total_ha and iqr_ha_val < 0.3 * total_ha:
        final_aff_area = max(fixed_ha_val, mad_ha_val, iqr_ha_val)
        logic_flag = "MAD & IQR Low (<30%)"
    else:
        final_aff_area = max(fixed_ha_val, mad_ha_val, iqr_ha_val)
        logic_flag = "Fallback Max"

    return {
        "fixed_ha": fixed_ha, "mad_ha": mad_ha, "iqr_ha": iqr_ha,
        "RS_AffectedArea": final_aff_area, "LogicFlag": logic_flag,
        "RS_Severity": RS_Severity_all, "RS_Severity_Affected": RS_Severity_affected,
        "mean_relative_severity_pct": round(mean_rel_all, 2),
        "std_relative_severity_pct": round(std_rel_all, 2),
        "mean_normalized_severity_pct": round(mean_norm_all, 2),
        "std_normalized_severity_pct": round(std_norm_all, 2),
        "mean_scaled_severity_pct": round(mean_scaled_all, 2),
        "std_scaled_severity_pct": round(std_scaled_all, 2),
        "mean_severity_affected_relative": round(mean_rel_aff, 2) if mean_rel_aff is not None else None,
        "std_severity_affected_relative": round(std_rel_aff, 2) if std_rel_aff is not None else None
    }

# ---------------------------
# Phase F: Main loop (paired sampling per polygon)
# ---------------------------
st.header("Run full pixel-paired analysis (Phase F)")
run_btn = st.button("Run Analysis (heavy)")

if not run_btn:
    st.info("Click 'Run Analysis (heavy)' to start the full pipeline.")
    st.stop()

# Ensure chart dir clean & available
CHART_DIR = CHART_DIR_BASE
if os.path.exists(CHART_DIR):
    shutil.rmtree(CHART_DIR)
os.makedirs(CHART_DIR, exist_ok=True)

# iterate polygons
results = []
gdf_iter = gdf_wgs.reset_index(drop=True)
total = len(gdf_iter)
progress = st.progress(0)
status_text = st.empty()

for idx, row in gdf_iter.iterrows():
    case_id = row.get(POLYGON_ID_COL)
    loss_date = None
    if LOSS_DATE_COL_INPUT in row.index:
        try:
            loss_date = pd.to_datetime(row.get(LOSS_DATE_COL_INPUT, None), errors='coerce')
        except Exception:
            loss_date = None

    geom = row.geometry.buffer(0)
    if geom is None or geom.is_empty:
        results.append({"case_ID": case_id, "LogicFlag": "Invalid Geometry", "RS_AffectedArea": 0})
        status_text.text(f"{idx+1}/{total}: {case_id} invalid geometry")
        progress.progress((idx+1)/total)
        continue

    gj = mapping(geom)
    gj2d = convert_3d_to_2d_geojson(dict(gj))
    ee_geom = ee.Geometry(gj2d)

    # post image window
    if not (loss_date is None or pd.isna(loss_date)):
        start = (loss_date + pd.Timedelta(days=int(POST_OFFSET_START_DAYS))).strftime('%Y-%m-%d')
        end = (loss_date + pd.Timedelta(days=int(POST_OFFSET_END_DAYS))).strftime('%Y-%m-%d')
        post_ic = s2_all.filterDate(start, end)
        try:
            n_post_images = int(safe_get_info(post_ic.size()))
        except Exception:
            n_post_images = 0
        post_img = post_ic.median().select('ndvi') if n_post_images > 0 else ee.Image([])
        post_range = f"{start}..{end}" if n_post_images > 0 else None
    else:
        n_post_images = 0
        post_img = ee.Image([])
        post_range = None

    # sample paired
    pre_img = pre_max_ndvi.select('pre_max_ndvi').rename('pre') if 'pre_max_ndvi' in safe_get_info(pre_max_ndvi.bandNames()) else pre_max_ndvi
    try:
        pre_vals, post_vals = sample_paired_pixels(pre_img, post_img, ee_geom, sample_limit=int(SAMPLE_LIMIT), scale=int(SCALE))
    except Exception as e:
        st.warning(f"Sampling failed for {case_id}: {e}")
        pre_vals, post_vals = [], []

    pre_pixel_count = len(pre_vals)
    post_pixel_count = len(post_vals)
    used_pixels = min(pre_pixel_count, post_pixel_count)
    Surveyed_Area = round(pre_pixel_count * 0.01, 4)

    metrics = compute_severity_and_metrics(pre_vals, post_vals, scale=int(SCALE))

    row_out = {
        "case_ID": case_id,
        "Loss_Date": loss_date.date() if (loss_date is not None and not pd.isna(loss_date)) else None,
        "Post_Date_Range": post_range,
        "n_post_images": n_post_images,
        "post_cloud_pct": None,
        "pre_AvgNDVI": round(float(np.nanmean(pre_vals)), 3) if pre_vals else None,
        "post_AvgNDVI": round(float(np.nanmean(post_vals)), 3) if post_vals else None,
        "preImage Pixels": pre_pixel_count,
        "postImage Pixels": post_pixel_count,
        "usedPixels": int(used_pixels),
        "Surveyed_Area": Surveyed_Area
    }
    row_out.update(metrics)
    results.append(row_out)

    # precompute fortnightly chart for this polygon and save PNG (Phase G precompute)
    try:
        fortnights = pd.date_range(CHART_START, CHART_END, freq=f"{int(FREQ_DAYS)}D")
        ts = []
        for i in range(len(fortnights)-1):
            s = fortnights[i].strftime("%Y-%m-%d")
            e = fortnights[i+1].strftime("%Y-%m-%d")
            ic = s2_all.filterDate(s, e).map(add_ndvi).select('ndvi')
            try:
                if safe_get_info(ic.size()) == 0:
                    val = np.nan
                else:
                    val = safe_get_info(ic.max().reduceRegion(ee.Reducer.mean(), ee_geom, SCALE, maxPixels=MAXPIX).get('ndvi'))
            except Exception:
                val = np.nan
            ts.append({"date": s, "maxNDVI": val})
        df_poly = pd.DataFrame(ts)
        df_poly["date"] = pd.to_datetime(df_poly["date"])
        df_poly["maxNDVI"] = df_poly["maxNDVI"].fillna(0)

        # save PNG
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.plot(df_poly['date'], df_poly['maxNDVI'], marker='o', linestyle='-', color='green')
        ax.set_ylim(0, 1)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("NDVI", fontsize=9)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(True, linewidth=0.3)
        plt.tight_layout()

        chart_path = os.path.join(CHART_DIR, f"{case_id}.png")
        fig.savefig(chart_path, dpi=110)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Chart generation failed for {case_id}: {e}")

    status_text.text(f"Processed {idx+1}/{total} | {case_id} | RS_AffectedArea={metrics.get('RS_AffectedArea',0)}")
    progress.progress((idx+1)/total)

# Save CSV (Phase F output)
df = pd.DataFrame(results)
desired_cols = [
    "case_ID","Loss_Date","Post_Date_Range","n_post_images","post_cloud_pct",
    "pre_AvgNDVI","post_AvgNDVI",
    "preImage Pixels","postImage Pixels","usedPixels","Surveyed_Area",
    "fixed_ha","mad_ha","iqr_ha",
    "RS_AffectedArea","LogicFlag","RS_Severity","RS_Severity_Affected",
    "mean_relative_severity_pct","std_relative_severity_pct",
    "mean_normalized_severity_pct","std_normalized_severity_pct",
    "mean_scaled_severity_pct","std_scaled_severity_pct",
    "mean_severity_affected_relative","std_severity_affected_relative"
]
for c in desired_cols:
    if c not in df.columns:
        df[c] = None
df = df[desired_cols]

csv_path = os.path.join(tmp_dir, EXPORT_CSV_NAME)
df.to_csv(csv_path, index=False, float_format='%.4f')
st.success(f"Analysis complete. CSV saved: {csv_path}")

# Provide download button
with open(csv_path, "rb") as f:
    st.download_button("Download results CSV", f, file_name=EXPORT_CSV_NAME, mime="text/csv")

# ---------------------------
# PHASE G→J: Build fast folium map using precomputed charts
# ---------------------------
st.header("Interactive map (precomputed charts + EE layers)")

# Build folium map (center on ROI centroid)
centroid = geom.centroid  # last polygon centroid used as example
center = [centroid.y, centroid.x]
m = folium.Map(location=center, zoom_start=11, tiles="Esri.WorldImagery")

# EE tile layers
rgb_vis = {"min":0, "max":3000, "bands":['B4','B3','B2']}
ndvi_vis = {"min":0.0, "max":0.9, "palette":["red","yellow","green"]}
try:
    pre_rgb = s2_all.filterDate(PRE_START, PRE_END).median().select(['B4','B3','B2'])
    m.add_child(geemap_tile.ee_tile_layer(pre_rgb, vis_params=rgb_vis, name="Pre-event RGB"))
    m.add_child(geemap_tile.ee_tile_layer(pre_max_ndvi, vis_params=ndvi_vis, name="Pre-event Max NDVI"))
except Exception:
    st.warning("Could not add EE tile layers (server-side error) — map will still show polygons.")

# Add post-event RGB layers for unique loss windows (to keep legend small we add unique windows only)
unique_loss_dates = pd.to_datetime(gdf_wgs[LOSS_DATE_COL_INPUT].dropna()).unique()
for ld in unique_loss_dates:
    ld = pd.to_datetime(ld)
    start = (ld + pd.Timedelta(days=int(POST_OFFSET_START_DAYS))).strftime('%Y-%m-%d')
    end = (ld + pd.Timedelta(days=int(POST_OFFSET_END_DAYS))).strftime('%Y-%m-%d')
    post_ic = s2_all.filterDate(start, end)
    try:
        if safe_get_info(post_ic.size()) > 0:
            post_rgb = post_ic.median().select(['B4','B3','B2'])
            m.add_child(geemap_tile.ee_tile_layer(post_rgb, vis_params=rgb_vis, name=f"Post-event RGB {start}..{end}"))
    except Exception:
        pass

# Single FeatureGroup for polygons
poly_group = folium.FeatureGroup(name="CLS Polygons", show=True)
m.add_child(poly_group)

# Build display gdf merged with df results (gdf_wgs may have many columns; we use safe get)
gdf_display = gdf_wgs.merge(df, left_on=POLYGON_ID_COL, right_on="case_ID", how="left")

# Convert datetime columns to strings to avoid JSON serialization errors
for col in gdf_display.columns:
    if pd.api.types.is_datetime64_any_dtype(gdf_display[col]):
        gdf_display[col] = gdf_display[col].astype(str)

# Add polygons to single feature group, popup uses precomputed chart file (embedded base64)
for _, row in gdf_display.iterrows():
    case_id = str(row.get(POLYGON_ID_COL, ""))
    geom_json = mapping(row.geometry)
    geom_json_2d = convert_3d_to_2d_geojson(dict(geom_json))

    chart_file = os.path.join(CHART_DIR, f"{case_id}.png")
    if os.path.exists(chart_file):
        chart_b64 = png_to_base64(chart_file)
        chart_img_tag = f'<img src="data:image/png;base64,{chart_b64}" width="300px">'
    else:
        chart_img_tag = "<i>No chart available</i>"

    html = f"""
    <div style="width:360px; font-size:13px; line-height:1.25;">
      <b>Case ID:</b> {case_id}<br>
      <b>Loss Date:</b> {row.get('Loss_Date','N/A')}<br>
      <b>Surveyed Area (ha):</b> {row.get('Surveyed_Area','N/A')}<br>
      <b>Used Pixels:</b> {row.get('usedPixels','N/A')}<br>
      <b>Affected Area (RS, ha):</b> {row.get('RS_AffectedArea','N/A')}<br>
      <b>Crop Loss (RS, %):</b> {row.get('RS_Severity','N/A')}<br><br>
      <b>NDVI Phenology</b><br>
      {chart_img_tag}
    </div>
    """

    iframe = folium.IFrame(html=html, width=370, height=420)
    popup = folium.Popup(iframe, max_width=400)

    folium.GeoJson(
        geom_json_2d,
        popup=popup,
        tooltip=f"{case_id}",
        style_function=lambda feat: {'color': 'orange', 'weight': 2, 'fillOpacity': 0.12}
    ).add_to(poly_group)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Display folium map in Streamlit
st.subheader("Map — click polygons to open popup")
st.write("Use the search box (below) to zoom to a polygon by CaseID.")
st_data = st_folium(m, width=1100, height=700)

# ---------------------------
# Search box: zoom to polygon (recenter map)
# ---------------------------
st.subheader("Search CaseID")
search_id = st.text_input("Enter CaseID to search & zoom")

if st.button("Zoom to CaseID"):
    if search_id.strip() == "":
        st.warning("Enter a CaseID")
    else:
        matched = gdf_display[gdf_display[POLYGON_ID_COL].astype(str) == str(search_id)]
        if matched.empty:
            st.error("CaseID not found")
        else:
            geom = matched.geometry.iloc[0]
            cent = geom.centroid
            # create a new map centered on the polygon (with the same layers)
            m2 = folium.Map(location=[cent.y, cent.x], zoom_start=15, tiles="Esri.WorldImagery")
            try:
                m2.add_child(geemap_tile.ee_tile_layer(pre_rgb, vis_params=rgb_vis, name="Pre-event RGB"))
                m2.add_child(geemap_tile.ee_tile_layer(pre_max_ndvi, vis_params=ndvi_vis, name="Pre-event Max NDVI"))
            except Exception:
                pass
            # add the matched polygon highlighted
            geom_json_2d = convert_3d_to_2d_geojson(dict(mapping(geom)))
            folium.GeoJson(geom_json_2d, style_function=lambda feat: {'color': 'red', 'weight': 3, 'fillOpacity': 0.25}).add_to(m2)
            st_folium(m2, width=900, height=600)

st.markdown("---")
st.info("Done. You can re-run the app with a different gpkg. For large polygon sets consider running in Colab or using a service account for EE.")

# Cleanup temp directory on exit? (optional)
# shutil.rmtree(tmp_dir)  # disabled to allow user to download outputs
