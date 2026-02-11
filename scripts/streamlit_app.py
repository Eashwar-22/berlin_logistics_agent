import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from agent import run_agent_logic

# Page Config
st.set_page_config(layout="wide", page_title="Berlin Logistics Agent")

st.title("Berlin Logistics Agent")

# --- Session State for Clicks ---
if 'start_point' not in st.session_state:
    st.session_state.start_point = (52.5200, 13.4000) # Default A
if 'end_point' not in st.session_state:
    st.session_state.end_point = (52.5000, 13.3500)   # Default B
if 'click_step' not in st.session_state:
    st.session_state.click_step = 0 # 0 = waiting for A, 1 = waiting for B

# --- Sidebar Inputs ---
st.sidebar.header("Logistics Params")
weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Snow"])
traffic = st.sidebar.select_slider("Traffic", ["Low", "Medium", "High"], value="Medium")
vehicle = st.sidebar.radio("Vehicle", ["Bike", "Scooter", "Van"])
driver_exp = st.sidebar.radio("Experience", ["Junior", "Senior", "Expert"], index=1)

# --- Main Map Area ---
st.write("**Click on the map to set points:** First click = **Pickup (Green)**, Second click = **Dropoff (Red)**.")

# Create Map
m = folium.Map(location=[52.51, 13.38], zoom_start=12)

# Add Markers
folium.Marker(
    st.session_state.start_point, 
    popup="Pickup", 
    icon=folium.Icon(color="green", icon="play")
).add_to(m)

folium.Marker(
    st.session_state.end_point, 
    popup="Dropoff", 
    icon=folium.Icon(color="red", icon="stop")
).add_to(m)

# Capture Clicks
output = st_folium(m, width=800, height=500)

# Logic to update points on click
if output['last_clicked']:
    lat = output['last_clicked']['lat']
    lng = output['last_clicked']['lng']
    
    # Simple toggle logic
    if st.session_state.click_step == 0:
        st.session_state.start_point = (lat, lng)
        st.session_state.click_step = 1
        st.rerun()
    else:
        st.session_state.end_point = (lat, lng)
        st.session_state.click_step = 0
        st.rerun()

# --- Agent Section ---
st.divider()

col1, col2 = st.columns([2, 1]) # Left is wider for text

with col1:
    st.subheader("Agent Output")
    st.info(f"📍 **Pickup:** {st.session_state.start_point}")
    st.info(f"🏁 **Dropoff:** {st.session_state.end_point}")

    if st.button("Optimize Route"):
        with st.spinner("Analyzing parameters..."):
            start = st.session_state.start_point
            end = st.session_state.end_point
            
            query = (
                f"Calculate distance between ({start[0]}, {start[1]}) and ({end[0]}, {end[1]}). "
                f"Between which two areas/streets in Berlin are these coordinates? "
                f"Then predict delivery time for a {vehicle} in {traffic} traffic with a {driver_exp} driver "
                f"in {weather} weather. Explain the prediction."
            )
            
            try:
                response = run_agent_logic(query)
                st.markdown("### 📋 Agent Analysis")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")

# --- Agent Section ---
st.divider()

st.subheader("Agent Output")
col1, col2 = st.columns(2)
with col1:
    st.info(f"📍 **Pickup:** {st.session_state.start_point}")
with col2:
    st.info(f"🏁 **Dropoff:** {st.session_state.end_point}")

if st.button("Optimize Route"):
    with st.spinner("Analyzing parameters..."):
        start = st.session_state.start_point
        end = st.session_state.end_point
        
        query = (
            f"Calculate distance between ({start[0]}, {start[1]}) and ({end[0]}, {end[1]}). "
            f"Between which two areas/streets in Berlin are these coordinates? "
            f"Then predict delivery time for a {vehicle} in {traffic} traffic with a {driver_exp} driver "
            f"in {weather} weather. Explain the prediction."
        )
        
        try:
            response = run_agent_logic(query)
            st.markdown("### 📋 Agent Analysis")
            st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")
