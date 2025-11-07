import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.evaluation.dual_model_pipeline import DualModelPipeline

st.title("🧠 Fetal Hydrocephalus Detection System")
st.sidebar.header("Upload Ultrasound")

uploaded_file = st.file_uploader("Choose an ultrasound image", type=['png', 'jpg'])

if uploaded_file:
    # Display original
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, caption="Original Ultrasound")
    
    # Run inference
    with st.spinner("Analyzing..."):
        pipeline = DualModelPipeline()
        results = pipeline.process_image(image)
    
    with col2:
        # Show segmentation overlay
        st.image(results['overlay'], caption="Detected Regions")
    
    with col3:
        # Show measurements
        st.metric("VHR", f"{results['vhr']:.1%}")
        st.metric("Classification", results['severity'])
        
        # Color code severity
        severity_colors = {
            'Normal': 'green',
            'Mild': 'yellow', 
            'Moderate': 'orange',
            'Severe': 'red'
        }
        st.markdown(f"<h3 style='color:{severity_colors[results['severity']]}'>{results['diagnosis']}</h3>", unsafe_allow_html=True)
    
    # Detailed metrics
    with st.expander("Detailed Measurements"):
        st.json(results['measurements'])