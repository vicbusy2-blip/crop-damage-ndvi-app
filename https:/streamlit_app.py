import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import ee
import os

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Crop Damage NDVI App", layout="wide")

st.title("🌾 Crop Damage NDVI Assessment App")
st.write("Upload polygon file (GeoPackage) and compute NDVI-based crop loss.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload .gpkg file", type=["gpkg"])

if uploaded_file:
    gdf = gpd.read_file(uploaded_file)
    st.success(f"Loaded {len(gdf)} polygons")

    st.write("### Preview of Attributes")
    st.dataframe(gdf.head())

    # Extract centroid for map
    centroid = gdf.to_crs(4326).geometry.centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

    # Add polygons to map
    folium.GeoJson(
        gdf.to_crs(4326),
        name="Polygons",
        tooltip=folium.GeoJsonTooltip(fields=[gdf.columns[0]])
    ).add_to(m)

    st.write("### Polygon Map")
    st_folium(m, width=700, height=500)

    st.info("⚙️ NDVI processing, chart creation, and EE integration will be added next.")

else:
    st.warning("Please upload a GeoPackage to begin.")
