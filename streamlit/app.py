import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from core.histogram_processor import HistogramProcessor
from core.image_handler import ImageHandler
from utils.constants import *


st.set_page_config(
    page_title="Histogram Spesifikasi",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none }
</style>
""", unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("<h3 style='text-align: center;'>Histogram Spesifikasi</h3>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    input_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png", "bmp"], key="input")
with col2:
    ref_file = st.file_uploader("Upload referensi", type=["jpg", "jpeg", "png", "bmp"], key="ref")

st.divider()

col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    process_btn = st.button("Proses", type="primary", use_container_width=True)

st.divider()

if process_btn:
    if input_file is None:
        st.error("Upload gambar input!")
        st.stop()
    if ref_file is None:
        st.error("Upload gambar referensi!")
        st.stop()
    
    progress = st.progress(0)
    
    try:
        progress.progress(20)
        img_input = Image.open(input_file)
        img_input = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
        
        img_reference = Image.open(ref_file)
        img_reference = cv2.cvtColor(np.array(img_reference), cv2.COLOR_RGB2BGR)
        
        progress.progress(40)
        image_handler = ImageHandler()
        img_input = image_handler.resize_image(img_input, MAX_IMAGE_SIZE)
        img_reference = image_handler.resize_image(img_reference, MAX_IMAGE_SIZE)
        
        progress.progress(60)
        histogram_processor = HistogramProcessor()
        img_result_bgr, gray_result, hist_inputs, hist_outputs = \
            histogram_processor.histogram_spesifikasi(img_input, img_reference, use_rgb=False)
        
        progress.progress(100)
        st.session_state.results = {
            'img_input': img_input,
            'img_reference': img_reference,
            'img_result_bgr': img_result_bgr,
            'hist_inputs': hist_inputs,
            'hist_outputs': hist_outputs
        }
        progress.empty()
        st.rerun()
        
    except Exception as e:
        progress.empty()
        st.error(f"Error: {str(e)}")
        st.stop()

if st.session_state.results:
    r = st.session_state.results
    
    st.markdown("## Hasil")
    
    col_in, col_ref, col_out = st.columns(3)
    
    with col_in:
        st.markdown("**Input**")
        st.image(cv2.cvtColor(r['img_input'], cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col_ref:
        st.markdown("**Referensi**")
        st.image(cv2.cvtColor(r['img_reference'], cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col_out:
        st.markdown("**Output**")
        img_rgb = cv2.cvtColor(r['img_result_bgr'], cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
        
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download", buf, file_name="result.png", mime="image/png", use_container_width=True)
    
    st.divider()
    
    st.markdown("## Histogram (Bar Chart)")
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    x = np.arange(0, 256, 4)
    
    hist_in = r['hist_inputs'][0].flatten()
    ax1.bar(x, hist_in[::4], width=3, color='blue', alpha=0.7)
    ax1.set_title('Histogram Input', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Nilai Pixel')
    ax1.set_ylabel('Jumlah')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim([0, 256])
    
    gray_ref = cv2.cvtColor(r['img_reference'], cv2.COLOR_BGR2GRAY)
    hist_ref = cv2.calcHist([gray_ref], [0], None, [256], [0, 256])
    hist_ref = hist_ref / hist_ref.sum()
    hist_ref = hist_ref.flatten()
    
    ax2.bar(x, hist_ref[::4], width=3, color='orange', alpha=0.7)
    ax2.set_title('Histogram Referensi', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Nilai Pixel')
    ax2.set_ylabel('Jumlah')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, 256])
    
    hist_out = r['hist_outputs'][0].flatten()
    ax3.bar(x, hist_out[::4], width=3, color='green', alpha=0.7)
    ax3.set_title('Histogram Output', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Nilai Pixel')
    ax3.set_ylabel('Jumlah')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim([0, 256])
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

