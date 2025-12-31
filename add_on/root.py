# launcher.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="App Launcher", layout="wide")

st.title("Digital Nova - Research Assistant") ## content sourcing and analysis

choice = st.radio("Open app:", ("OCR", "Clusstering", "Bootstrap Evaluation", "Auto-Labelling"), index=0, horizontal=True)

# map each choice to the running app's URL (adjust ports/paths)
url_map = {
    "OCR": "http://localhost:8502",   # app1 running here
    "Clusstering": "http://localhost:8502",   # app2
    "Bootstrap Evaluation": "http://localhost:8504",   # app3
    "Auto-Labelling": "http://localhost:8503",   # app4
}

target = url_map[choice]

# st.markdown(f"**Showing:** {choice} — `{target}`")

# iframe height can be adjusted
iframe_height = 1000

# embed
components.iframe(target, width="100%", height=iframe_height, scrolling=True)