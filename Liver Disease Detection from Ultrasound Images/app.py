"""
LIVER DISEASE DETECTION FROM ULTRASOUND IMAGES USING MACHINE LEARNING
Streamlit Web Application for Academic & Mini-Project Demonstration
Enhanced with Modern Medical-Tech UI, Interactive Steppers, 1-Click Demos, and Report Exporting.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from preprocessing import preprocess_image, load_image_as_rgb
from feature_extraction import (
    extract_features,
    extract_hog_features,
    extract_glcm_features
)
from dataset_utils import (
    scan_dataset,
    extract_dataset_features,
    perform_train_test_split,
    generate_sample_dataset
)
from model_trainer import (
    AVAILABLE_ALGORITHMS,
    train_and_evaluate_model,
    compare_all_models,
    save_model,
    load_saved_model,
    predict_single_image
)
from ui_components import (
    set_custom_style,
    render_header,
    render_pipeline_stepper,
    render_disclaimer,
    render_metric_cards,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_glcm_radar,
    plot_roc_curves,
    plot_model_comparison_chart,
    render_prediction_result,
    generate_diagnostic_text_report
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Liver Ultrasound Classifier • Royal AI Suite",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
set_custom_style()

# Initialize session state variables
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = "dataset"
if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = None
if "dataset_features" not in st.session_state:
    st.session_state.dataset_features = None  # tuple (X, y, classes, summary)
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "active_model_result" not in st.session_state:
    st.session_state.active_model_result = None
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "selected_quick_sample" not in st.session_state:
    st.session_state.selected_quick_sample = None

# Auto-scan dataset on first load if folder exists
if st.session_state.dataset_info is None and os.path.exists(st.session_state.dataset_path):
    st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)

# Auto-load saved model if available
if st.session_state.active_model_result is None:
    default_model_path = os.path.join("models", "liver_ultrasound_model.pkl")
    if os.path.exists(default_model_path):
        try:
            st.session_state.active_model_result = load_saved_model(default_model_path)
        except Exception:
            pass

# Sidebar Pipeline Status & Quick Controls
with st.sidebar:
    st.markdown("### 🎛️ Pipeline Status Monitor")
    
    # 1. Dataset status
    if st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0:
        ds = st.session_state.dataset_info
        st.markdown(f"**📁 Dataset:** <span class='status-badge-ready'>✓ {ds['total_images']} Images ({len(ds['classes'])} Classes)</span>", unsafe_allow_html=True)
    else:
        st.markdown("**📁 Dataset:** <span class='status-badge-pending'>⏳ Not Loaded</span>", unsafe_allow_html=True)

    # 2. Features status
    if st.session_state.dataset_features is not None:
        X_feat, _, _, _ = st.session_state.dataset_features
        st.markdown(f"**🔍 Features:** <span class='status-badge-ready'>✓ {X_feat.shape[0]} Samples ({X_feat.shape[1]:,} Dims)</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🔍 Features:** <span class='status-badge-pending'>⏳ Not Extracted</span>", unsafe_allow_html=True)

    # 3. Model status
    if st.session_state.active_model_result is not None:
        mod = st.session_state.active_model_result
        acc_pct = mod.get("accuracy", 0.0) * 100
        st.markdown(f"**🤖 Model:** <span class='status-badge-ready'>✓ {mod['algorithm']} ({acc_pct:.1f}% Acc)</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🤖 Model:** <span class='status-badge-pending'>⏳ Not Trained</span>", unsafe_allow_html=True)

    # 4. Inference status
    if st.session_state.active_model_result is not None:
        st.markdown("**🔮 Inference:** <span class='status-badge-ready'>✓ Ready to Classify</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🔮 Inference:** <span class='status-badge-pending'>⏳ Awaiting Model</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚡ 1-Click Quick Actions")
    if st.button("🚀 Fast Auto-Pipeline (3s)", use_container_width=True, help="Auto-generates data, extracts HOG+GLCM, and trains SVM"):
        with st.spinner("Executing end-to-end pipeline in background..."):
            if st.session_state.dataset_info is None or st.session_state.dataset_info["total_images"] == 0:
                generate_sample_dataset(st.session_state.dataset_path, 15)
                st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)
            
            X, y, classes, summary = extract_dataset_features(st.session_state.dataset_info["image_items"])
            st.session_state.dataset_features = (X, y, classes, summary)
            
            X_train, X_test, y_train, y_test, split_info = perform_train_test_split(X, y, test_size=0.20, random_state=42)
            st.session_state.split_data = (X_train, X_test, y_train, y_test, split_info)
            
            model_res = train_and_evaluate_model("SVM", X_train, y_train, X_test, y_test, classes)
            st.session_state.active_model_result = model_res
            save_model(model_res, "models", "liver_ultrasound_model.pkl")
        st.sidebar.success(f"Pipeline ready! Accuracy: {model_res['accuracy']*100:.1f}%")
        st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Dataset Configuration")
    dataset_input_dir = st.text_input(
        "Dataset Directory Path",
        value=st.session_state.dataset_path,
        help="Directory containing class subfolders (e.g. dataset/Normal, dataset/Fatty_Liver, dataset/Cirrhosis)"
    )
    if dataset_input_dir != st.session_state.dataset_path:
        st.session_state.dataset_path = dataset_input_dir
        st.session_state.dataset_info = None
        st.session_state.dataset_features = None
        st.session_state.split_data = None
        st.session_state.active_model_result = None

    if st.button("🔄 Rescan Dataset Folder", use_container_width=True):
        st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💾 Model Management")
    default_model_path = os.path.join("models", "liver_ultrasound_model.pkl")
    if os.path.exists(default_model_path):
        if st.button("📂 Reload Saved Model (.pkl)", use_container_width=True):
            try:
                loaded = load_saved_model(default_model_path)
                st.session_state.active_model_result = loaded
                st.sidebar.success(f"Loaded {loaded.get('algorithm', 'Model')} successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")

    st.markdown("---")
    st.caption(
        "**Academic Project:** Liver Disease Detection from Ultrasound Images Using Machine Learning\n\n"
        "**Stack:** Python • Streamlit • OpenCV • Scikit-Learn • HOG • GLCM\n\n"
        "**Notice:** Educational & Research demonstration only."
    )

# Render main header and disclaimer banner
render_header()

# Render interactive pipeline stepper
ds_ready = bool(st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0)
feat_ready = bool(st.session_state.dataset_features is not None)
mod_ready = bool(st.session_state.active_model_result is not None)
mod_name = st.session_state.active_model_result.get("algorithm", "SVM") if mod_ready else ""
img_count = st.session_state.dataset_info["total_images"] if ds_ready else 0

render_pipeline_stepper(
    dataset_ready=ds_ready,
    features_ready=feat_ready,
    model_ready=mod_ready,
    model_name=mod_name,
    total_images=img_count
)

render_disclaimer()

# Create 7 application tabs
tab_overview, tab_dataset, tab_preprocess, tab_features, tab_training, tab_eval, tab_predict = st.tabs([
    "🏠 Overview & Pipeline",
    "📁 Dataset Explorer",
    "🖼️ Preprocessing Studio",
    "🔍 Feature Extraction",
    "🤖 Model Training",
    "📊 Performance & Benchmarks",
    "🔮 Live Prediction & Inference"
])

# ==============================================================================
# TAB 1: OVERVIEW & PIPELINE
# ==============================================================================
with tab_overview:
    st.markdown("### 📌 Project Title & Core Architecture")
    st.markdown(
        """
        **LIVER DISEASE DETECTION FROM ULTRASOUND IMAGES USING MACHINE LEARNING**
        
        This academic project demonstrates a complete, transparent, and reproducible **Traditional Machine Learning & Computer Vision Pipeline** for classifying B-mode liver ultrasound scans into clinical categories:
        - 🟢 **Normal Liver Parenchyma:** Homogeneous medium echogenicity, regular acoustic speckle pattern.
        - 🟡 **Fatty Liver (Steatosis):** Diffuse hyperechogenicity ("bright liver") with depth attenuation.
        - 🔴 **Cirrhosis / Fibrosis:** Coarse heterogeneous nodularity, irregular fibrotic acoustic bands.
        
        Unlike black-box Deep Learning (CNNs), this system leverages interpretable statistical features (**HOG** gradients and **GLCM** second-order texture matrices) paired with robust classifiers (**SVM**, **Random Forest**, and **KNN**).
        """
    )

    # 1-Click Fast Pipeline Demo Box
    st.markdown(
        """
        <div class="quick-action-box">
            <h4>⚡ 1-Click Fast Pipeline Demo</h4>
            <p>Want to test the full end-to-end Machine Learning pipeline right away? Click below to automatically generate or scan sample ultrasound data, extract HOG + GLCM texture features, train the SVM classifier, and evaluate model performance in under 3 seconds.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🚀 EXECUTE COMPLETE 1-CLICK PIPELINE DEMO", use_container_width=True):
        with st.spinner("Executing end-to-end pipeline (Data Check → HOG + GLCM Extraction → SVM Training → Evaluation)..."):
            if st.session_state.dataset_info is None or st.session_state.dataset_info["total_images"] == 0:
                generate_sample_dataset(st.session_state.dataset_path, 15)
                st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)

            X, y, classes, summary = extract_dataset_features(st.session_state.dataset_info["image_items"])
            st.session_state.dataset_features = (X, y, classes, summary)

            X_train, X_test, y_train, y_test, split_info = perform_train_test_split(X, y, test_size=0.20, random_state=42)
            st.session_state.split_data = (X_train, X_test, y_train, y_test, split_info)

            model_res = train_and_evaluate_model("SVM", X_train, y_train, X_test, y_test, classes)
            st.session_state.active_model_result = model_res
            save_model(model_res, "models", "liver_ultrasound_model.pkl")

        st.success(f"🎉 Complete Pipeline Executed Successfully! Model: SVM (RBF) | Accuracy: {model_res['accuracy']*100:.1f}% | Classes: {', '.join(classes)}")
        st.toast("1-Click Pipeline Finished!", icon="✅")
        st.rerun()

    docx_rep_path = "Liver_Disease_Detection_Project_Report.docx"
    if os.path.exists(docx_rep_path):
        with open(docx_rep_path, "rb") as f_docx:
            st.download_button(
                label="📥 DOWNLOAD COMPLETE ACADEMIC PROJECT REPORT (.DOCX)",
                data=f_docx.read(),
                file_name="Liver_Disease_Detection_Project_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                help="Download ready-to-submit Word Document (.docx) matching the academic report specification."
            )

    st.markdown("---")
    st.markdown("#### 🔄 End-to-End Machine Learning Pipeline Workflow")
    
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    with f1:
        st.info("**1. Input Image**\n\nUltrasound B-mode image acquisition")
    with f2:
        st.info("**2. Preprocessing**\n\n128x128, Gray, Denoise, Hist Eq, Norm")
    with f3:
        st.info("**3. HOG Features**\n\n9-bin edge gradient orientations")
    with f4:
        st.info("**4. GLCM Texture**\n\nContrast, Energy, Homogeneity (0°-135°)")
    with f5:
        st.info("**5. ML Classifier**\n\nSVM / Random Forest / KNN Models")
    with f6:
        st.info("**6. Prediction**\n\nDiagnostic class & Confidence score")

    st.markdown("---")
    col_tech, col_arch = st.columns([1, 1])

    with col_tech:
        st.markdown("#### 🛠️ Technology Stack & Libraries")
        tech_data = {
            "Component": [
                "Programming Language",
                "Web UI Framework",
                "Image Processing",
                "HOG Feature Extraction",
                "GLCM Texture Analysis",
                "Machine Learning Models",
                "Data Manipulation",
                "Visualization",
                "Model Serialization"
            ],
            "Technology": [
                "Python 3.10+",
                "Streamlit",
                "OpenCV & Pillow",
                "Scikit-Image (HOG)",
                "Scikit-Image (graycomatrix / props)",
                "Scikit-Learn (SVM, RF, KNN)",
                "Pandas & NumPy",
                "Matplotlib & Seaborn",
                "Joblib"
            ]
        }
        st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)

    with col_arch:
        st.markdown("#### 🩺 Clinical Ultrasound Acoustic Characteristics")
        st.markdown(
            """
            - **Normal Liver Parenchyma:** Homogeneous medium-level echogenicity, fine acoustic speckle distribution, smooth hepatic contour, and clear vascular architecture.
            - **Fatty Liver (Steatosis):** Diffuse parenchymal hyperechogenicity ("bright liver"), increased acoustic attenuation in deep zones, and vascular blurring.
            - **Cirrhosis (Fibrosis):** Coarse heterogeneous nodular texture, surface irregularity/nodularity, fibrotic reticular acoustic bands, and acoustic shadowing.
            """
        )

    # Expandable Viva & Project FAQ
    with st.expander("🎓 Project Defense & Viva Quick Reference Guide", expanded=False):
        st.markdown(
            """
            **Q1: Why use Traditional Machine Learning (HOG + GLCM + SVM) instead of Deep Learning (CNNs)?**  
            *Answer:* Ultrasound datasets in academic and clinical pilot studies often have limited sample sizes where Deep Learning overfits. Traditional ML requires significantly fewer training samples, offers transparent mathematical feature representations (interpretable texture statistics), and computes inferences in milliseconds on standard CPU hardware.
            
            **Q2: What is the purpose of Gaussian Blur and Histogram Equalization in Ultrasound Preprocessing?**  
            *Answer:* Ultrasound images inherently suffer from acoustic speckle noise caused by constructive and destructive interference of backscattered acoustic waves. Gaussian filtering smooths high-frequency speckle while preserving anatomical edges. Histogram Equalization or CLAHE stretches the narrowband dynamic contrast range to reveal subtle parenchymal variations.
            
            **Q3: What does the GLCM matrix capture?**  
            *Answer:* Gray-Level Co-occurrence Matrix (GLCM) tabulates how often different combinations of pixel brightness levels occur at specified spatial offsets $(\\Delta x, \\Delta y)$ and angular orientations $(0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ)$. Second-order metrics like *Contrast*, *Homogeneity*, *Energy*, and *Correlation* quantitatively differentiate smooth liver parenchyma from coarse fibrotic nodularity.
            """
        )


# ==============================================================================
# TAB 2: DATASET EXPLORER
# ==============================================================================
with tab_dataset:
    st.markdown("### 📁 Dataset Management & Dynamic Class Discovery")
    
    col_path, col_btn = st.columns([3, 1])
    with col_path:
        curr_dir = st.text_input(
            "Dataset Directory Path",
            value=st.session_state.dataset_path,
            key="dataset_tab_path_input"
        )
        st.session_state.dataset_path = curr_dir
    with col_btn:
        st.write("")
        st.write("")
        if st.button("📂 Load / Rescan Dataset", use_container_width=True):
            st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)
            st.rerun()

    if st.session_state.dataset_info is None:
        st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)

    ds_info = st.session_state.dataset_info

    if not ds_info["exists"] or ds_info["total_images"] == 0:
        st.warning(f"⚠️ {ds_info.get('error', 'No images found in dataset directory.')}")
        st.markdown(
            """
            **How to organize your dataset:**
            Place your images in class subfolders inside the `dataset/` directory:
            ```text
            dataset/
            ├── Normal/
            │   ├── scan_01.png
            ├── Fatty_Liver/
            │   ├── scan_02.png
            └── Cirrhosis/
                └── scan_03.png
            ```
            """
        )

        st.markdown("---")
        st.markdown("#### 🧪 Generate Sample Educational Dataset (1-Click)")
        st.info("No ultrasound dataset handy? Generate a synthetic acoustic texture dataset with realistic Rayleigh speckle and acoustic properties in seconds.")
        
        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            num_samples = st.slider("Samples per class", min_value=10, max_value=40, value=15, step=5)
        with gen_col2:
            st.write("")
            st.write("")
            if st.button("✨ Generate Sample Dataset", use_container_width=True):
                with st.spinner("Generating synthetic ultrasound dataset..."):
                    gen_result = generate_sample_dataset(
                        base_dir=st.session_state.dataset_path,
                        samples_per_class=num_samples
                    )
                    st.session_state.dataset_info = scan_dataset(st.session_state.dataset_path)
                    st.success(f"Generated {gen_result['total_created']} images across: {', '.join(gen_result['classes'])}!")
                    st.rerun()

    else:
        st.success(f"✓ Dataset verified at `{st.session_state.dataset_path}` with {ds_info['total_images']} total images.")
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Total Images", ds_info["total_images"])
        with mcol2:
            st.metric("Number of Classes", len(ds_info["classes"]))
        with mcol3:
            st.metric("Detected Classes", ", ".join(ds_info["classes"]))

        st.markdown("---")

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.markdown("#### 📊 Class Distribution Chart")
            fig = plot_class_distribution(ds_info["class_counts"])
            st.pyplot(fig)
        with col_table:
            st.markdown("#### 📋 Images Count Breakdown")
            counts_df = pd.DataFrame([
                {"Class Name": k, "Image Count": v, "Share (%)": f"{round(v / ds_info['total_images'] * 100, 1)}%"}
                for k, v in ds_info["class_counts"].items()
            ])
            st.dataframe(counts_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🖼️ Dataset Image Gallery Preview")
        preview_class = st.selectbox("Select Class to Preview", ds_info["classes"])
        class_images = [item for item in ds_info["image_items"] if item["class_name"] == preview_class]
        
        num_preview = min(6, len(class_images))
        if num_preview > 0:
            preview_cols = st.columns(num_preview)
            for i, pcol in enumerate(preview_cols):
                with pcol:
                    img_item = class_images[i]
                    pil_img = Image.open(img_item["path"])
                    st.image(pil_img, caption=f"{img_item['filename']}", use_container_width=True)


# ==============================================================================
# TAB 3: PREPROCESSING STUDIO
# ==============================================================================
with tab_preprocess:
    st.markdown("### 🖼️ Interactive Image Preprocessing Studio")
    st.markdown(
        """
        The preprocessing module standardizes raw ultrasound scans and enhances subtle parenchymal textures:
        $$\\text{Raw Image} \\longrightarrow \\text{Resize (128}\\times\\text{128)} \\longrightarrow \\text{Grayscale} \\longrightarrow \\text{Gaussian Denoise} \\longrightarrow \\text{Contrast Enhancement} \\longrightarrow \\text{Normalization [0, 1]}$$
        """
    )

    col_up, col_sample = st.columns([1, 1])
    sample_image_to_process = None

    with col_up:
        uploaded_preprocess_file = st.file_uploader(
            "Upload an image to test preprocessing",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="preprocess_uploader"
        )
        if uploaded_preprocess_file is not None:
            sample_image_to_process = Image.open(uploaded_preprocess_file)

    with col_sample:
        if st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0:
            sample_choices = [item["path"] for item in st.session_state.dataset_info["image_items"][:20]]
            selected_sample = st.selectbox(
                "Or choose from loaded dataset",
                options=sample_choices,
                format_func=lambda x: f"{os.path.basename(os.path.dirname(x))} / {os.path.basename(x)}"
            )
            if sample_image_to_process is None and selected_sample:
                sample_image_to_process = Image.open(selected_sample)

    with st.expander("🛠️ Interactive Preprocessing Hyperparameter Tuner", expanded=False):
        tune_c1, tune_c2, tune_c3 = st.columns(3)
        with tune_c1:
            blur_k = st.selectbox("Gaussian Blur Kernel Size", [3, 5, 7], index=1)
        with tune_c2:
            enh_method = st.selectbox("Contrast Enhancement Method", ["Histogram Equalization", "CLAHE"], index=0)
        with tune_c3:
            clahe_clip = st.slider("CLAHE Clip Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.5) if enh_method == "CLAHE" else 2.0

    if sample_image_to_process is not None:
        st.markdown("---")
        st.markdown("#### 🔍 Step-by-Step Preprocessing Stages")

        pre_results = preprocess_image(
            sample_image_to_process,
            target_size=(128, 128),
            blur_kernel=(blur_k, blur_k),
            enhancement_method=enh_method,
            clahe_clip_limit=clahe_clip
        )

        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        
        with pcol1:
            st.markdown("<div class='step-pill'>STAGE 1</div>", unsafe_allow_html=True)
            st.image(pre_results["original"], caption="Original Ultrasound", use_container_width=True)
            st.caption(f"Shape: {pre_results['stats']['original_shape']}")

        with pcol2:
            st.markdown("<div class='step-pill'>STAGE 2 & 3</div>", unsafe_allow_html=True)
            st.image(pre_results["grayscale"], caption="Grayscale (128x128)", clamp=True, use_container_width=True)
            st.caption(f"Range: [{pre_results['stats']['raw_min']}, {pre_results['stats']['raw_max']}]")

        with pcol3:
            st.markdown("<div class='step-pill'>STAGE 4</div>", unsafe_allow_html=True)
            st.image(pre_results["denoised"], caption=f"Gaussian Blur ({blur_k}x{blur_k})", clamp=True, use_container_width=True)
            st.caption("Speckle filtered")

        with pcol4:
            st.markdown("<div class='step-pill'>STAGE 5</div>", unsafe_allow_html=True)
            st.image(pre_results["enhanced"], caption=f"Enhanced ({enh_method})", clamp=True, use_container_width=True)
            st.caption(f"Gain: {pre_results['stats']['contrast_gain']:.2f}x")

        with pcol5:
            st.markdown("<div class='step-pill'>STAGE 6</div>", unsafe_allow_html=True)
            st.image(pre_results["normalized_display"], caption="Normalized [0.0, 1.0]", clamp=True, use_container_width=True)
            st.caption("Float32 Standardized")

        st.markdown("---")
        st.markdown("#### 📈 Histogram & Statistical Analysis")
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 3.2))
            ax_hist.hist(pre_results["grayscale"].ravel(), bins=64, color="#78716c", alpha=0.7, label="Original Grayscale")
            ax_hist.hist(pre_results["enhanced"].ravel(), bins=64, color="#ea580c", alpha=0.75, label=f"Enhanced ({enh_method})")
            ax_hist.set_title("Intensity Distribution (Before vs After Enhancement)", fontsize=10.5, weight="bold", color="#1c1917")
            ax_hist.set_xlabel("Pixel Intensity (0 - 255)", fontsize=9, color="#450a0a")
            ax_hist.set_ylabel("Pixel Count", fontsize=9, color="#450a0a")
            ax_hist.legend(fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_hist)

        with stat_col2:
            st.markdown("##### 📊 Preprocessing Metrics Table")
            stats_table = pd.DataFrame([
                {"Stage": "Raw Grayscale", "Min": pre_results["stats"]["raw_min"], "Max": pre_results["stats"]["raw_max"], "Mean": round(pre_results["stats"]["raw_mean"], 2), "Std Dev": round(pre_results["stats"]["raw_std"], 2)},
                {"Stage": "Contrast Enhanced", "Min": pre_results["stats"]["enhanced_min"], "Max": pre_results["stats"]["enhanced_max"], "Mean": round(pre_results["stats"]["enhanced_mean"], 2), "Std Dev": round(pre_results["stats"]["enhanced_std"], 2)},
                {"Stage": "Normalized (Float32)", "Min": round(pre_results["stats"]["norm_min"], 4), "Max": round(pre_results["stats"]["norm_max"], 4), "Mean": round(pre_results["normalized"].mean(), 4), "Std Dev": round(pre_results["normalized"].std(), 4)}
            ])
            st.dataframe(stats_table, hide_index=True, use_container_width=True)
    else:
        st.info("Upload an image or pick from dataset above to inspect the preprocessing studio.")


# ==============================================================================
# TAB 4: FEATURE EXTRACTION
# ==============================================================================
with tab_features:
    st.markdown("### 🔍 Feature Engineering: HOG Gradients + GLCM Texture Matrix")
    st.markdown(
        """
        Two complementary feature descriptors are fused together for each ultrasound scan:
        1. **HOG (Histogram of Oriented Gradients):** Evaluates local edge directions and structural tissue boundaries ($9$ angular bins, $8\\times8$ cell size).
        2. **GLCM (Gray-Level Co-occurrence Matrix):** Computes statistical spatial relationship between pixel pairs across angles ($0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ$).
        """
    )

    feat_sample_img = None
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        feat_upload = st.file_uploader(
            "Upload image for feature analysis",
            type=["jpg", "jpeg", "png", "bmp", "tif"],
            key="feature_tab_uploader"
        )
        if feat_upload is not None:
            feat_sample_img = Image.open(feat_upload)

    with fcol2:
        if st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0:
            sample_choices = [item["path"] for item in st.session_state.dataset_info["image_items"][:20]]
            selected_sample = st.selectbox(
                "Or choose from loaded dataset",
                options=sample_choices,
                format_func=lambda x: f"{os.path.basename(os.path.dirname(x))} / {os.path.basename(x)}",
                key="feat_sample_selector"
            )
            if feat_sample_img is None and selected_sample:
                feat_sample_img = Image.open(selected_sample)

    if feat_sample_img is not None:
        st.markdown("---")
        with st.spinner("Extracting HOG and GLCM feature representations..."):
            combined_vector, feat_meta = extract_features(feat_sample_img)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("HOG Features", f"{feat_meta['hog_count']:,}")
        with c2:
            st.metric("GLCM Features", f"{feat_meta['glcm_count']:,}")
        with c3:
            st.metric("Combined Vector Size", f"{feat_meta['total_count']:,}")
        with c4:
            st.metric("Vector Shape", f"{feat_meta['feature_shape']}")

        st.markdown("---")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("#### 🌟 HOG Gradient Vector Visualization")
            st.image(feat_meta["hog_image"], caption="HOG Gradient Vector Map (Orientations=9, Cell=8x8)", clamp=True, use_container_width=True)

        with vcol2:
            st.markdown("#### 🔬 GLCM Second-Order Texture Radar")
            fig_radar = plot_glcm_radar(feat_meta["glcm_summary"])
            st.pyplot(fig_radar)

        st.markdown("---")
        st.markdown("#### 📋 GLCM Numerical Texture Statistics Table")
        glcm_rows = []
        for prop_name, prop_data in feat_meta["glcm_summary"].items():
            glcm_rows.append({
                "Texture Property": prop_name.capitalize(),
                "Mean Value": f"{prop_data['mean']:.5f}",
                "Std Dev": f"{prop_data['std']:.5f}",
                "Min": f"{prop_data['min']:.5f}",
                "Max": f"{prop_data['max']:.5f}"
            })
        st.dataframe(pd.DataFrame(glcm_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Upload an image above or load a dataset to visualize feature extraction.")


# ==============================================================================
# TAB 5: MODEL TRAINING
# ==============================================================================
with tab_training:
    st.markdown("### 🤖 Dataset Feature Extraction & Model Training")
    
    if st.session_state.dataset_info is None or st.session_state.dataset_info["total_images"] == 0:
        st.warning("⚠️ Please scan or generate a dataset in the **📁 Dataset Explorer** tab first.")
    else:
        ds_info = st.session_state.dataset_info
        
        st.markdown("#### Step 1: Batch Feature Extraction")
        st.write(f"Dataset contains **{ds_info['total_images']}** images across **{len(ds_info['classes'])}** classes: `{', '.join(ds_info['classes'])}`.")

        extract_btn = st.button("⚡ EXTRACT FEATURES FOR ALL IMAGES", use_container_width=True)

        if extract_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(curr, total):
                progress_bar.progress(curr / total)
                status_text.text(f"Extracting features: {curr} / {total} images processed ({int(curr/total*100)}%)")

            with st.spinner("Processing images and extracting HOG + GLCM features..."):
                X, y, unique_classes, summary = extract_dataset_features(
                    ds_info["image_items"],
                    progress_callback=update_progress
                )
                st.session_state.dataset_features = (X, y, unique_classes, summary)
                st.session_state.split_data = None
                st.session_state.active_model_result = None

            status_text.empty()
            progress_bar.empty()
            st.success(f"✓ Extracted features for {len(X)} images! Total dimensions: {X.shape[1]:,} features per image.")
            st.rerun()

        if st.session_state.dataset_features is not None:
            X, y, unique_classes, summary = st.session_state.dataset_features

            st.markdown("---")
            st.markdown("#### Step 2: Stratified Train / Test Dataset Split")
            
            scol1, scol2 = st.columns([2, 1])
            with scol1:
                test_size = st.slider("Test Set Ratio (Default: 20%)", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
            with scol2:
                random_seed = st.number_input("Random State Seed", value=42, step=1)

            X_train, X_test, y_train, y_test, split_info = perform_train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )
            st.session_state.split_data = (X_train, X_test, y_train, y_test, split_info)

            sp_col1, sp_col2, sp_col3, sp_col4 = st.columns(4)
            with sp_col1:
                st.metric("Training Samples", f"{split_info['train_samples']} ({split_info['train_pct']}%)")
            with sp_col2:
                st.metric("Testing Samples", f"{split_info['test_samples']} ({split_info['test_pct']}%)")
            with sp_col3:
                st.metric("Feature Vector Dims", f"{X.shape[1]:,}")
            with sp_col4:
                st.metric("Stratified Split", "Yes (Balanced)" if split_info["stratified"] else "No (Safe Fallback)")

            st.markdown("---")
            st.markdown("#### Step 3: Choose Algorithm & Train Pipeline")

            algo_col1, algo_col2 = st.columns([2, 1])
            with algo_col1:
                selected_algo = st.selectbox(
                    "Select Algorithm:",
                    options=list(AVAILABLE_ALGORITHMS.keys()),
                    format_func=lambda x: f"{x} - {AVAILABLE_ALGORITHMS[x]['name']}"
                )
            with algo_col2:
                st.write("")
                st.write("")
                train_btn = st.button("🚀 TRAIN CLASSIFIER", use_container_width=True)

            with st.expander("⚙️ Optional Hyperparameter Tuning", expanded=False):
                custom_params = {}
                if selected_algo == "SVM":
                    h1, h2 = st.columns(2)
                    with h1:
                        custom_params["C"] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
                    with h2:
                        custom_params["kernel"] = st.selectbox("Kernel Function", ["rbf", "linear", "poly"], index=0)
                elif selected_algo == "Random Forest":
                    h1, h2 = st.columns(2)
                    with h1:
                        custom_params["n_estimators"] = st.slider("Number of Trees", 20, 200, 100, 10)
                    with h2:
                        custom_params["criterion"] = st.selectbox("Splitting Criterion", ["gini", "entropy"], index=0)
                elif selected_algo == "KNN":
                    h1, h2 = st.columns(2)
                    with h1:
                        custom_params["n_neighbors"] = st.slider("K Neighbors", 1, 15, 5, 2)
                    with h2:
                        custom_params["weights"] = st.selectbox("Weight Function", ["uniform", "distance"], index=0)

            if train_btn:
                with st.spinner(f"Training {selected_algo} with StandardScaler Pipeline..."):
                    model_result = train_and_evaluate_model(
                        algorithm_key=selected_algo,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        class_names=unique_classes,
                        custom_params=custom_params
                    )
                    st.session_state.active_model_result = model_result
                    saved_path = save_model(model_result, save_dir="models", filename="liver_ultrasound_model.pkl")

                st.success(f"✓ Model {selected_algo} trained successfully! Saved to `{saved_path}`.")
                st.rerun()

            if st.session_state.active_model_result is not None:
                res = st.session_state.active_model_result
                st.markdown("##### 📌 Current Active Model Status")
                status_col1, status_col2, status_col3, status_col4 = st.columns(4)
                with status_col1:
                    st.write(f"**Algorithm:** `{res['algorithm']}`")
                with status_col2:
                    st.write(f"**Test Accuracy:** `{res['accuracy']*100:.2f}%`")
                with status_col3:
                    st.write(f"**Training Time:** `{res.get('train_duration_sec', 0):.4f}s`")
                with status_col4:
                    st.write(f"**Trained At:** `{res.get('trained_at', 'N/A')}`")


# ==============================================================================
# TAB 6: PERFORMANCE & BENCHMARKS
# ==============================================================================
with tab_eval:
    st.markdown("### 📊 Model Performance & Comparative Benchmarks")

    if st.session_state.active_model_result is None:
        st.warning("⚠️ No active trained model found. Please train a model in the **🤖 Model Training** tab or run the 1-Click Fast Pipeline from the sidebar.")
    else:
        res = st.session_state.active_model_result
        st.markdown(f"#### Active Classifier: **{res.get('algorithm_full_name', res['algorithm'])}**")
        
        render_metric_cards(
            accuracy=res["accuracy"],
            precision=res["precision"],
            recall=res["recall"],
            f1=res["f1_score"]
        )

        st.markdown("---")
        
        cm_col, rep_col = st.columns([1.2, 1])

        with cm_col:
            st.markdown("#### 🎯 Confusion Matrix")
            if "confusion_matrix" in res and res["confusion_matrix"] is not None:
                fig_cm = plot_confusion_matrix(res["confusion_matrix"], res["class_names"])
                st.pyplot(fig_cm)
            else:
                st.info("Confusion matrix not available.")

        with rep_col:
            st.markdown("#### 📋 Detailed Classification Report")
            if "classification_report_df" in res and res["classification_report_df"] is not None:
                st.dataframe(res["classification_report_df"], hide_index=True, use_container_width=True)
            elif "classification_report_dict" in res:
                st.json(res["classification_report_dict"])

        if "roc_data" in res and res["roc_data"]:
            st.markdown("---")
            st.markdown("#### 📈 Multi-Class Receiver Operating Characteristic (ROC)")
            fig_roc = plot_roc_curves(res["roc_data"])
            st.pyplot(fig_roc)

        st.markdown("---")
        st.markdown("### ⚖️ Multi-Model Benchmark Arena (SVM vs Random Forest vs KNN)")
        st.markdown("Compare all three traditional ML algorithms on the identical stratified train/test split.")

        if st.session_state.split_data is not None:
            X_train, X_test, y_train, y_test, split_info = st.session_state.split_data
            
            if st.button("⚖️ RUN FULL MULTI-MODEL BENCHMARK", use_container_width=True):
                with st.spinner("Training & comparing SVM, Random Forest, and KNN..."):
                    comp_df, comp_dict = compare_all_models(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        class_names=split_info["classes"]
                    )
                    st.session_state.comparison_result = comp_df

            if st.session_state.comparison_result is not None:
                comp_df = st.session_state.comparison_result
                
                c_tbl, c_plot = st.columns([1.2, 1.8])
                with c_tbl:
                    st.markdown("#### 📋 Benchmark Metrics Table")
                    st.dataframe(comp_df, hide_index=True, use_container_width=True)
                with c_plot:
                    st.markdown("#### 📈 Performance Comparison Bar Chart")
                    fig_comp = plot_model_comparison_chart(comp_df)
                    st.pyplot(fig_comp)
        else:
            st.info("Train a model first to enable the multi-model comparison arena.")


# ==============================================================================
# TAB 7: LIVE PREDICTION & INFERENCE
# ==============================================================================
with tab_predict:
    st.markdown("### 🔮 Live Ultrasound Image Prediction & Diagnostic Studio")
    
    active_model = st.session_state.active_model_result
    if active_model is None:
        default_model_path = os.path.join("models", "liver_ultrasound_model.pkl")
        if os.path.exists(default_model_path):
            try:
                active_model = load_saved_model(default_model_path)
                st.session_state.active_model_result = active_model
            except Exception:
                pass

    if active_model is None:
        st.warning("⚠️ No trained model is currently loaded. Click '🚀 Fast Auto-Pipeline (3s)' in the sidebar to prepare a model instantly.")
    else:
        st.info(f"Loaded Active Classifier: **{active_model.get('algorithm_full_name', active_model['algorithm'])}** | Trained Classes: `{', '.join(active_model['class_names'])}`")

        # 1-Click Interactive Ultrasound Demo Cards
        st.markdown("#### ⚡ 1-Click Ultrasound Test Scans")
        st.caption("Click any sample card below to test the full pipeline immediately without uploading files:")
        
        if st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0:
            ds_items = st.session_state.dataset_info["image_items"]
            classes_avail = st.session_state.dataset_info["classes"]
            
            p_cols = st.columns(len(classes_avail))
            for idx, cname in enumerate(classes_avail):
                matching = [it for it in ds_items if it["class_name"] == cname]
                if matching:
                    with p_cols[idx]:
                        st.markdown(
                            f"""
                            <div class="sample-preset-card">
                                <h5>{cname.replace('_', ' ')}</h5>
                                <p>Test demo scan with {cname.lower()} acoustic profile</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button(f"🔍 Load & Classify {cname}", key=f"btn_preset_{cname}", use_container_width=True):
                            st.session_state.selected_quick_sample = matching[0]["path"]

        st.markdown("---")
        col_upload, col_ds_pick = st.columns([1, 1])
        predict_img = None
        img_source_name = "Uploaded Ultrasound Scan"

        with col_upload:
            uploaded_pred_file = st.file_uploader(
                "Or Upload Your Ultrasound Image (PNG, JPG, BMP, TIF)",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                key="predict_file_uploader"
            )
            if uploaded_pred_file is not None:
                predict_img = Image.open(uploaded_pred_file)
                img_source_name = uploaded_pred_file.name

        with col_ds_pick:
            quick_sample_path = st.session_state.get("selected_quick_sample", None)
            if st.session_state.dataset_info and st.session_state.dataset_info["total_images"] > 0:
                sample_choices = [item["path"] for item in st.session_state.dataset_info["image_items"]]
                idx_sel = sample_choices.index(quick_sample_path) if quick_sample_path in sample_choices else 0
                selected_sample = st.selectbox(
                    "Or pick an ultrasound scan from dataset:",
                    options=sample_choices,
                    index=idx_sel,
                    format_func=lambda x: f"[{os.path.basename(os.path.dirname(x))}] {os.path.basename(x)}",
                    key="pred_sample_selector"
                )
                if predict_img is None and selected_sample:
                    predict_img = Image.open(selected_sample)
                    img_source_name = f"{os.path.basename(os.path.dirname(selected_sample))} / {os.path.basename(selected_sample)}"

        if predict_img is not None:
            st.markdown("---")
            st.markdown("#### 🔬 Real-Time Pipeline Processing Stages")

            pre = preprocess_image(predict_img)

            p1, p2, p3 = st.columns(3)
            with p1:
                st.image(pre["original"], caption=f"STEP 1: Original Scan ({img_source_name})", use_container_width=True)
            with p2:
                st.image(pre["enhanced"], caption="STEP 2 & 3: Preprocessed & Enhanced (128x128)", clamp=True, use_container_width=True)
            with p3:
                feat_vec, feat_meta = extract_features(predict_img)
                st.image(feat_meta["hog_image"], caption="STEP 4: HOG Gradients Extracted", clamp=True, use_container_width=True)

            pred_out = predict_single_image(feat_vec, active_model)

            st.markdown("---")
            st.markdown("#### 🎯 Diagnostic Classification Result")

            render_prediction_result(
                predicted_class=pred_out["predicted_class"],
                algorithm=active_model.get("algorithm_full_name", active_model["algorithm"]),
                confidence=pred_out["confidence"],
                probabilities=pred_out["probabilities"]
            )

            # Diagnostic Report Export Option
            report_content = generate_diagnostic_text_report(
                predicted_class=pred_out["predicted_class"],
                algorithm=active_model.get("algorithm_full_name", active_model["algorithm"]),
                confidence=pred_out["confidence"],
                probabilities=pred_out["probabilities"],
                image_name=img_source_name
            )

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    label="📄 Download Diagnostic Summary Report (.md)",
                    data=report_content,
                    file_name=f"liver_ultrasound_report_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with dcol2:
                docx_rep_path = "Liver_Disease_Detection_Project_Report.docx"
                if os.path.exists(docx_rep_path):
                    with open(docx_rep_path, "rb") as f_docx:
                        st.download_button(
                            label="📥 Download Full Word Report (.docx)",
                            data=f_docx.read(),
                            file_name="Liver_Disease_Detection_Project_Report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
