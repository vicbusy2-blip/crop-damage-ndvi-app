import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import tempfile
import os
import json
from shapely.geometry import mapping, shape

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="NDVI-based Crop Damage App", layout="wide")

st.title("🌾 NDVI-based Crop Damage App")
st.write("Upload polygon file (GeoPackage) and compute NDVI-based crop loss.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_gpkg = st.file_uploader(
    "Upload GeoPackage (.gpkg)",
    type=["gpkg"]
)

if uploaded_gpkg:
    # Write uploaded .gpkg into a temporary file
    tmpdir = tempfile.mkdtemp()
    gpkg_path = os.path.join(tmpdir, uploaded_gpkg.name)

    with open(gpkg_path, "wb") as f:
        f.write(uploaded_gpkg.getvalue())

    # Read GPKG
    try:
        gdf = gpd.read_file(gpkg_path)

        # Fix geometry: convert 3D → 2D
        def fix_geom(g):
            if g is None:
                return None
            try:
                # If geometry has Z, strip to XY
                if hasattr(g, "has_z") and g.has_z:
                    coords = [(x, y) for x, y, *_ in g.coords]
                    return type(g)(coords)
                else:
                    return g
            except:
                # For polygons/multipolygons
                return shape(mapping(g))

        gdf["geometry"] = gdf["geometry"].apply(fix_geom)

        # Convert CRS → WGS84
        if gdf.crs is None:
            st.warning("No CRS found → assuming EPSG:4326")
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)

    except Exception as e:
        st.error(f"Failed to read GPKG: {e}")
        st.stop()

    st.success(f"Loaded {len(gdf)} polygons!")

    # Convert GeoDataFrame → clean GeoJSON
    geojson = json.loads(gdf.to_json())

    # Build map
    m = folium.Map(
        location=[
            gdf.geometry.centroid.y.mean(),
            gdf.geometry.centroid.x.mean()
        ],
        zoom_start=12
    )

    folium.GeoJson(
        geojson,
        name="Uploaded GPKG",
        tooltip=folium.GeoJsonTooltip(fields=[gdf.columns[0]])
    ).add_to(m)

    st_map = st_folium(m, height=600, width=700)
