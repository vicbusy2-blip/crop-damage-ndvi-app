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
    import tempfile
    import os
    import json
    from shapely.geometry import mapping, shape

    tmpdir = tempfile.mkdtemp()
    gpkg_path = os.path.join(tmpdir, uploaded_gpkg.name)

    with open(gpkg_path, "wb") as f:
        f.write(uploaded_gpkg.getvalue())

    # Load GeoPackage
    gdf = gpd.read_file(gpkg_path)

    # Fix 3D → 2D
    def fix_geom(g):
        if g is None:
            return None
        try:
            if hasattr(g, "has_z") and g.has_z:
                coords = [(x, y) for x, y, *_ in g.coords]
                return type(g)(coords)
            return g
        except:
            return shape(mapping(g))

    gdf["geometry"] = gdf["geometry"].apply(fix_geom)

    # Reproject to WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    # FIX: Convert all timestamps to strings
    gdf = gdf.applymap(
        lambda x: x.isoformat() if hasattr(x, "isoformat") else x
    )

    # Convert to valid GeoJSON
    geojson = json.loads(gdf.to_json())

    # Create map
    centroid = gdf.geometry.centroid
    m = folium.Map(
        location=[centroid.y.mean(), centroid.x.mean()], 
        zoom_start=12
    )

    folium.GeoJson(
        geojson,
        name="Uploaded Polygons",
        tooltip=folium.GeoJsonTooltip(fields=[gdf.columns[0]])
    ).add_to(m)

    st_folium(m, height=600, width=700)
