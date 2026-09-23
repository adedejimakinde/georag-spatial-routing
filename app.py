import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from sklearn.cluster import KMeans
import random

st.set_page_config(page_title="GeoRAG Spatial Routing", layout="wide")

st.markdown("<h1 style='color: #007acc;'>GeoRAG: Spatial Context Routing</h1>", unsafe_allow_html=True)
st.markdown("**Interactive Demo:** Visualizing the Scikit-Learn K-Means algorithm grouping raw coordinates into logical geographic contexts for LLM consumption.")

# 1. Generate Mock Retrieval Data (Simulating the Pass-1 Spatial Filter)
@st.cache_data
def generate_mock_data(center_lat, center_lon, num_points):
    points = []
    for _ in range(num_points):
        # Scatter points randomly within a small radius
        lat = center_lat + random.uniform(-0.05, 0.05)
        lon = center_lon + random.uniform(-0.05, 0.05)
        points.append([lat, lon])
    return np.array(points)

# Lagos coordinates
mock_coords = generate_mock_data(6.5244, 3.3792, 15)

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Algorithm Controls")
    max_clusters = st.slider("Max Geographic Regions", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.markdown("### LLM Context Output")
    st.markdown("Notice how the raw coordinates are synthesized into distinct regional centers before being passed to the AI.")

# 2. Execute the Routing Engine
kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(mock_coords)
centroids = kmeans.cluster_centers_

# 3. Render the Map
with col2:
    # Initialize Map over Lagos
    m = folium.Map(
        location=[6.5244, 3.3792], 
        zoom_start=11, 
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
    )

    # Initialize Map over Lagos
    m = folium.Map(
        location=[6.5244, 3.3792], 
        zoom_start=11, 
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
    )
    
    # Add the text labels over the base map
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Reference",
        overlay=True,
        control=False,
        name="Labels"
    ).add_to(m)
    
    colors = ['#007acc', '#ff4b4b', '#00cc96', '#ab63fa', '#ffa15a']
    
    # Plot the raw data points
    for idx, coord in enumerate(mock_coords):
        cluster_id = labels[idx]
        folium.CircleMarker(
            location=[coord[0], coord[1]],
            radius=6,
            color=colors[cluster_id],
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Document Chunk (Region {cluster_id + 1})"
        ).add_to(m)
        
    # Plot the calculated centroids
    for idx, centroid in enumerate(centroids):
        folium.Marker(
            location=[centroid[0], centroid[1]],
            icon=folium.Icon(color="white", icon_color=colors[idx], icon="info-sign"),
            tooltip=f"Region {idx + 1} Centroid"
        ).add_to(m)
        
        with col1:
            st.code(f"Region_{idx + 1}: \nApprox. Center: {centroid[1]:.4f} Lon, {centroid[0]:.4f} Lat\nContained Excerpts: {list(labels).count(idx)}", language="text")

    st_folium(m, width=800, height=500)