# streamlit_app.py
"""
Streamlit app to display precomputed Crop Damage results (Option B).
Features:
 - Upload or use default geopackage (.gpkg)
 - Upload or use precomputed CSV (results.csv)
 - Precomputed PNG charts (chart_dir/<case_id>.png) embedded in popup
 - Search CaseID -> zoom & highlight
 - Single "CLS Polygons" layer in the legend (FeatureGroup)
 - Download results CSV
 - Important: This app does NOT run Earth Engine. EE processing should be done offline (Colab/local).
"""

import os
import base64
import json
from io import BytesIO

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping

# --------------------------
# User-editable defaults
# --------------------------
# If you used Colab earlier, your gpkg was at: /content/colab_CLS2.gpkg
# We'll try this as a local default (useful for local testing). On Streamlit Cloud
# you probably will upload the files to repo / data/ folder or use upload UI.
DEFAULT_GPKG_PATH = "./colab_CLS2.gpkg"   # <-- replace with "./data/colab_CLS2.gpkg" in repo if you prefer
DEFAULT_CSV_PATH  = "./results.csv"
DEFAULT_CHART_DIR = "./charts"            # expects chart files named <case_id>.png

# --------------------------
# Helpers
# --------------------------
def ensure_gdf_datetime_to_str(gdf):
    """Convert any datetime columns to string to avoid JSON serialization issues."""
    for c in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[c]):
            gdf[c] = gdf[c].astype(str)
    return gdf

def png_file_to_data_uri(path):
    """Return data URI for PNG at path (or None if missing)."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def geom_to_geojson_feature(geom):
    """Return geojson feature dict from shapely geom (2D coords)."""
    gj = mapping(geom)
    # strip Z/M if present (robust)
    def strip(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], (float, int)):
                return [obj[0], obj[1]]
            return [strip(x) for x in obj]
        return obj
    if "coordinates" in gj:
        gj["coordinates"] = strip(gj["coordinates"])
    return gj

# --------------------------
# App UI
# --------------------------
st.set_page_config(layout="wide", page_title="Crop Damage — Viewer (precomputed)", initial_sidebar_state="expanded")
st.title("Crop Damage Viewer — precomputed results (Option B)")

st.sidebar.header("Files / Inputs")
uploaded_gpkg = st.sidebar.file_uploader("Upload GeoPackage (.gpkg) (optional)", type=["gpkg", "zip", "geojson"])
uploaded_csv  = st.sidebar.file_uploader("Upload results CSV (optional)", type=["csv"])
uploaded_chart_zip = st.sidebar.file_uploader("Upload charts.zip (optional, optional way to upload many PNGs)", type=["zip"])

use_defaults = st.sidebar.checkbox("Use default file paths (if present)", value=True)

# default path hints
st.sidebar.markdown("**Default paths (for testing)**")
st.sidebar.text(DEFAULT_GPKG_PATH)
st.sidebar.text(DEFAULT_CSV_PATH)
st.sidebar.text(DEFAULT_CHART_DIR)

# --------------------------
# Prepare data (GeoPackage)
# --------------------------
gdf = None
if uploaded_gpkg is not None:
    # when user uploads, save to a temp file and read
    tmp_gpkg = os.path.join("uploaded_files", uploaded_gpkg.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(tmp_gpkg, "wb") as f:
        f.write(uploaded_gpkg.getbuffer())
    try:
        gdf = gpd.read_file(tmp_gpkg)
    except Exception as e:
        st.error(f"Failed reading uploaded gpkg: {e}")
else:
    # try default path
    if use_defaults and os.path.exists(DEFAULT_GPKG_PATH):
        try:
            gdf = gpd.read_file(DEFAULT_GPKG_PATH)
        except Exception as e:
            st.warning(f"Couldn't read default .gpkg at {DEFAULT_GPKG_PATH}: {e}")

if gdf is None:
    st.info("No GeoPackage loaded yet. Upload a .gpkg or enable default path with a file present.")
    st.stop()

# unify CRS to EPSG:4326 for folium
try:
    gdf = gdf.to_crs(epsg=4326)
except Exception:
    # if already 4326 or fails, proceed
    pass

# --------------------------
# Prepare results CSV
# --------------------------
df_results = None
if uploaded_csv is not None:
    # read uploaded csv
    df_results = pd.read_csv(uploaded_csv)
else:
    if use_defaults and os.path.exists(DEFAULT_CSV_PATH):
        df_results = pd.read_csv(DEFAULT_CSV_PATH)

# If results exist, merge with gdf (left join on case_ID)
POLY_ID = "case_ID"  # adjust if different
if df_results is not None and POLY_ID in df_results.columns:
    # convert types to allow safe merge
    gdf[POLY_ID] = gdf[POLY_ID].astype(str)
    df_results[POLY_ID] = df_results[POLY_ID].astype(str)
    gdf_display = gdf.merge(df_results, left_on=POLY_ID, right_on=POLY_ID, how="left")
else:
    gdf_display = gdf.copy()

# convert datetimes -> strings to avoid folium JSON error
gdf_display = ensure_gdf_datetime_to_str(gdf_display)

# --------------------------
# Prepare charts folder: uploaded ZIP or default
# --------------------------
chart_dir = DEFAULT_CHART_DIR
# if user uploaded chart zip → extract
if uploaded_chart_zip is not None:
    import zipfile
    tmp_zip = os.path.join("uploaded_files", uploaded_chart_zip.name)
    with open(tmp_zip, "wb") as f:
        f.write(uploaded_chart_zip.getbuffer())
    extract_dir = os.path.join("uploaded_files", "charts")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall(extract_dir)
    chart_dir = extract_dir
else:
    # if default exists in repo, use it
    if not os.path.exists(chart_dir):
        chart_dir = None

# helper map creation function (we rebuild map after user actions)
def build_map(center=None, zoom=11, highlight_case=None):
    # center: (lat, lon)
    if center is None:
        # compute centroid of all features as default
        c = gdf_display.unary_union.centroid
        center = (c.y, c.x)
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="CartoDB positron")

    # Add precomputed raster layers placeholders (these do not call EE)
    # If you want to add raster tile URLs (e.g., hosted XYZ tiles), add here.
    # Example:
    # folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", name="OSM").add_to(m)

    # Add polygons in one FeatureGroup (single legend entry)
    poly_group = folium.FeatureGroup(name="CLS Polygons", show=True)
    for _, row in gdf_display.iterrows():
        geom = row.geometry
        # safe GeoJSON (2D)
        geojson_feature = geom_to_geojson_feature(geom)
        props = {}
        # add a small selection of properties for popup; include all columns but keep short
        for c in gdf_display.columns:
            if c == "geometry":
                continue
            # stringify values safely
            v = row.get(c)
            try:
                props[c] = "" if pd.isna(v) else str(v)
            except Exception:
                props[c] = "NA"

        # build popup HTML (include chart if present)
        caseid = str(row.get(POLY_ID, ""))
        chart_file = None
        if chart_dir:
            candidate = os.path.join(chart_dir, f"{caseid}.png")
            if os.path.exists(candidate):
                chart_file = candidate

        # create simple HTML with key fields and embed chart (base64) if file exists
        chart_html = ""
        if chart_file:
            with open(chart_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            chart_html = f'<br><img src="data:image/png;base64,{b64}" width="320px">'

        # prepare content: show a handful of selected fields (or all)
        content_lines = []
        # prefer nice names if available (user can change)
        wanted = ["case_ID", "Sowing Dat", "Date of Lo", "Cause of e", "Area affec", "Loss %",
                  "pre_AvgNDVI", "post_AvgNDVI", "usedPixels", "Surveyed_Area", "RS_AffectedArea", "RS_Severity"]
        # fallback to all columns if none of the wanted exist
        if any([c in gdf_display.columns for c in wanted]):
            for c in wanted:
                if c in props:
                    content_lines.append(f"<b>{c}:</b> {props.get(c,'')}")
        else:
            # show first 10 properties
            shown = 0
            for k, v in props.items():
                if shown >= 10:
                    break
                content_lines.append(f"<b>{k}:</b> {v}")
                shown += 1

        html = "<div style='max-width:360px; font-size:13px; line-height:1.25;'>" + "<br>".join(content_lines) + chart_html + "</div>"
        iframe = folium.IFrame(html=html, width=360, height=420)
        popup = folium.Popup(iframe, max_width=400)

        gj = folium.GeoJson(
            geojson_feature,
            popup=popup,
            tooltip=str(caseid),
            style_function=lambda feat: {'color': 'orange', 'weight': 2, 'fillOpacity': 0.12}
        )
        gj.add_to(poly_group)

    poly_group.add_to(m)

    # If highlight_case requested, add a thick layer over that geometry and zoom
    if highlight_case is not None:
        sel = gdf_display[gdf_display[POLY_ID].astype(str) == str(highlight_case)]
        if len(sel) > 0:
            geom = sel.iloc[0].geometry
            # add highlight
            folium.GeoJson(mapping(geom), style_function=lambda f: {'color': 'red', 'weight':4, 'fillOpacity':0.05}).add_to(m)
            # zoom to bounds of geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# --------------------------
# Search & Map display
# --------------------------
st.sidebar.header("Search by CaseID")
search_id = st.sidebar.text_input("Enter CaseID to zoom (exact match)", value="")
if st.sidebar.button("Zoom to CaseID"):
    if search_id.strip() == "":
        st.sidebar.warning("Enter a case_ID first.")
    else:
        if search_id.strip() not in gdf_display[POLY_ID].astype(str).values:
            st.sidebar.error("CaseID not found in loaded GeoPackage / results.")
        else:
            map_obj = build_map(highlight_case=search_id.strip())
            st_map = st_folium(map_obj, width=1000, height=650)
            st.success(f"Zoomed to CaseID: {search_id.strip()}")
            # stop here to avoid duplicate map rendering below
            st.stop()

# default map rendering (no highlight)
map_obj = build_map()
st_map = st_folium(map_obj, width=1000, height=650)

# --------------------------
# Download results CSV
# --------------------------
st.sidebar.header("Download")
if df_results is not None:
    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download results CSV", data=csv_bytes, file_name="results.csv", mime="text/csv")
else:
    st.sidebar.info("No results CSV loaded (upload or place results.csv in app folder).")

st.sidebar.markdown("---")
st.sidebar.caption("App expects precomputed results & charts. Earth Engine processing should be done in Colab/local.")
