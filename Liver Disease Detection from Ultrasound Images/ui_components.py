"""
Royal Medical-Tech UI Components, Visualizations, and Style System for Streamlit App
Features a luxurious Royal Red & Radiant Orange color combination, elegant typography,
enhanced accessibility (WCAG compliant contrast, clear focus states), and regal aesthetic.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st


def set_custom_style():
    """Injects custom CSS styling for a luxurious, royal medical-tech workstation UI in Red & Orange."""
    custom_css = """
    <style>
    /* Google Fonts: Outfit (Regal & Modern Display) + Plus Jakarta Sans + JetBrains Mono */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --royal-red: #dc2626;
        --royal-dark-red: #991b1b;
        --royal-wine: #450a0a;
        --royal-orange: #ea580c;
        --royal-sunburst: #f97316;
        --royal-amber: #f59e0b;
        --royal-gold: #fbbf24;
        --royal-gold-light: #fef08a;
        --bg-warm-light: #fffaf5;
        --card-border: #fed7aa;
        --text-primary: #1c1917;
        --text-secondary: #57534e;
        --text-muted: #78716c;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Accessibility: Universal visible focus outline for keyboard navigation */
    a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible {
        outline: 3px solid #ea580c !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 10px rgba(234, 88, 12, 0.45) !important;
    }

    /* Custom Royal Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #fff7ed;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626 0%, #ea580c 100%);
        border-radius: 8px;
        border: 2px solid #fff7ed;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #b91c1c 0%, #c2410c 100%);
    }

    /* =========================================================================
       STREAMLIT NATIVE COMPONENT OVERRIDES (ROYAL RED & SUNBURST ORANGE)
       ========================================================================= */

    /* Primary & Standard Buttons: Royal Gradient with Golden Glow */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 30%, #ea580c 70%, #f97316 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.03em !important;
        border: 1px solid rgba(251, 191, 36, 0.5) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 6px 20px -3px rgba(220, 38, 38, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(234, 88, 12, 0.52), 0 0 12px rgba(251, 191, 36, 0.3) !important;
        filter: brightness(1.08) !important;
        border-color: #fbbf24 !important;
    }

    div.stButton > button:active, div.stDownloadButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 3px 10px rgba(220, 38, 38, 0.4) !important;
    }

    /* Streamlit Navigation Tabs: Royal High-Contrast Tab Bar */
    div[data-baseweb="tab-list"] {
        border-bottom: 2.5px solid #fed7aa !important;
        gap: 6px !important;
        padding-bottom: 2px !important;
        margin-bottom: 1.5rem !important;
        background: transparent !important;
    }

    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        color: #57534e !important;
        background: transparent !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #ea580c !important;
        background: rgba(234, 88, 12, 0.06) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #b91c1c !important;
        font-weight: 800 !important;
        background: linear-gradient(180deg, rgba(234, 88, 12, 0.08) 0%, rgba(220, 38, 38, 0.12) 100%) !important;
        border-bottom: 3.5px solid #ea580c !important;
    }

    div[data-baseweb="tab-border"] {
        background-color: transparent !important;
    }

    div[data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #dc2626 0%, #ea580c 50%, #f59e0b 100%) !important;
        height: 3.5px !important;
        border-radius: 3px !important;
    }

    /* Sidebar: Velvet Royal Luxury Dark with Gold & Orange Trim */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #18080a 0%, #260c10 40%, #1c080b 100%) !important;
        border-right: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.25) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #fbbf24 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.6) !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #ffedd5 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(234, 88, 12, 0.25) !important;
        margin: 1.2rem 0 !important;
    }

    section[data-testid="stSidebar"] input {
        background: #2a0e13 !important;
        color: #ffffff !important;
        border: 1px solid rgba(251, 146, 60, 0.45) !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] input:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 10px rgba(249, 115, 22, 0.4) !important;
    }

    /* Form Inputs and Interactive Controls */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border-color: #fed7aa !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #ea580c !important;
        box-shadow: 0 0 0 2px rgba(234, 88, 12, 0.25) !important;
    }

    /* Sliders */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ea580c !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.5) !important;
    }

    /* Expanders */
    div.streamlit-expanderHeader {
        background: #fff7ed !important;
        border: 1px solid #fed7aa !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        color: #9a3412 !important;
    }
    div.streamlit-expanderHeader:hover {
        background: #ffedd5 !important;
        border-color: #ea580c !important;
    }

    /* =========================================================================
       ROYAL BRAND HEADER & HERO SECTION
       ========================================================================= */
    .hero-header {
        background: linear-gradient(135deg, #180709 0%, #360a0f 35%, #7f1d1d 72%, #9a3412 100%);
        padding: 32px 40px;
        border-radius: 24px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 18px 44px -10px rgba(220, 38, 38, 0.42), 0 0 0 1px rgba(251, 191, 36, 0.22);
        border: 1.5px solid rgba(251, 146, 60, 0.35);
        position: relative;
        overflow: hidden;
    }

    /* Ambient Royal Gold & Amber Glow Orbs */
    .hero-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(220, 38, 38, 0.38) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-header::after {
        content: "";
        position: absolute;
        bottom: -50%;
        right: -10%;
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(249, 115, 22, 0.32) 0%, rgba(245, 158, 11, 0.18) 40%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.35) 0%, rgba(245, 158, 11, 0.28) 100%);
        color: #fbbf24;
        padding: 6px 16px;
        border-radius: 28px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 14px;
        border: 1.5px solid rgba(251, 191, 36, 0.48);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.25);
        text-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }

    .hero-header h1 {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
        background: linear-gradient(90deg, #ffffff 0%, #ffedd5 50%, #fed7aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.35rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 10px 0 !important;
        line-height: 1.18 !important;
        text-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    .hero-header p {
        color: #fdba74 !important;
        font-size: 1.08rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        max-width: 900px !important;
        line-height: 1.5 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    /* =========================================================================
       PIPELINE PROGRESS STEPPER (ROYAL RED & SUNBURST ACCENTS)
       ========================================================================= */
    .pipeline-stepper-container {
        background: #ffffff;
        border: 1.5px solid #fed7aa;
        border-radius: 18px;
        padding: 18px 22px;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.06), 0 1px 3px rgba(0,0,0,0.03);
    }

    .stepper-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.84rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9a3412;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .stepper-track {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    @media (max-width: 768px) {
        .stepper-track {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    .step-item {
        background: #fffaf5;
        border: 1.5px solid #fed7aa;
        border-radius: 14px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .step-item.active {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-color: #ea580c;
        box-shadow: 0 6px 16px rgba(234, 88, 12, 0.22);
        transform: translateY(-2px);
    }

    .step-item.complete {
        background: linear-gradient(135deg, #fef2f2 0%, #fff7ed 100%);
        border-color: #dc2626;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.12);
    }

    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 0.85rem;
        background: #fee2e2;
        color: #991b1b;
        flex-shrink: 0;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }

    .step-item.active .step-num {
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.45);
    }

    .step-item.complete .step-num {
        background: linear-gradient(135deg, #ea580c 0%, #f59e0b 100%);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.45);
    }

    .step-info-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        font-weight: 800;
        color: #1c1917;
        line-height: 1.2;
    }

    .step-info-sub {
        font-size: 0.76rem;
        color: #78716c;
        margin-top: 3px;
        font-weight: 500;
    }

    /* =========================================================================
       NOTICE & DISCLAIMER BANNER
       ========================================================================= */
    .disclaimer-banner {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-left: 5px solid #ea580c;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: #9a3412;
        font-size: 0.90rem;
        line-height: 1.5;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(234, 88, 12, 0.08);
        border-top: 1px solid #fed7aa;
        border-right: 1px solid #fed7aa;
        border-bottom: 1px solid #fed7aa;
    }

    /* =========================================================================
       QUICK ACTION CARD (ROYAL VELVET NIGHT)
       ========================================================================= */
    .quick-action-box {
        background: linear-gradient(135deg, #180709 0%, #360a0f 50%, #7c2d12 100%);
        color: #ffffff;
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 26px;
        border: 1.5px solid rgba(251, 146, 60, 0.35);
        box-shadow: 0 10px 32px rgba(24, 7, 9, 0.45);
        position: relative;
        overflow: hidden;
    }

    .quick-action-box::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(249, 115, 22, 0.28) 0%, transparent 70%);
        pointer-events: none;
    }

    .quick-action-box h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #fbbf24 !important;
        margin: 0 0 8px 0 !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }

    .quick-action-box p {
        color: #ffedd5 !important;
        font-size: 0.96rem !important;
        line-height: 1.55 !important;
        margin-bottom: 16px !important;
    }

    /* =========================================================================
       MODERN METRIC CARDS (ROYAL ACCENT STRIP)
       ========================================================================= */
    .metric-card-pro {
        background: #ffffff;
        border: 1.5px solid #fed7aa;
        border-top: 5px solid #dc2626;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
    }

    .metric-card-pro:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(234, 88, 12, 0.18);
        border-color: #ea580c;
        border-top-color: #ea580c;
    }

    .metric-card-pro .tag {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #78716c;
        margin-bottom: 6px;
    }

    .metric-card-pro .val {
        font-size: 2.25rem;
        font-weight: 900;
        color: #b91c1c;
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
        line-height: 1.1;
    }

    .metric-card-pro .desc {
        font-size: 0.84rem;
        color: #ea580c;
        font-weight: 700;
        margin-top: 8px;
    }

    /* =========================================================================
       DIAGNOSTIC RESULT CARDS (ROYAL SEVERITY STYLES)
       ========================================================================= */
    .result-card-normal {
        background: linear-gradient(135deg, #fffbeb 0%, #f0fdf4 100%);
        border: 2.5px solid #059669;
        border-radius: 20px;
        padding: 30px 34px;
        margin: 22px 0;
        box-shadow: 0 14px 36px rgba(5, 150, 105, 0.16);
    }

    .result-card-fatty {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 65%, #ffffff 100%);
        border: 2.5px solid #ea580c;
        border-radius: 20px;
        padding: 30px 34px;
        margin: 22px 0;
        box-shadow: 0 14px 36px rgba(234, 88, 12, 0.20);
    }

    .result-card-cirrhosis {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 65%, #ffffff 100%);
        border: 2.5px solid #dc2626;
        border-radius: 20px;
        padding: 30px 34px;
        margin: 22px 0;
        box-shadow: 0 14px 36px rgba(220, 38, 38, 0.24);
    }

    .result-card-default {
        background: linear-gradient(135deg, #fffaf5 0%, #ffffff 100%);
        border: 2.5px solid #ea580c;
        border-radius: 20px;
        padding: 30px 34px;
        margin: 22px 0;
        box-shadow: 0 14px 36px rgba(234, 88, 12, 0.16);
    }

    .badge-normal {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 14px;
        box-shadow: 0 3px 10px rgba(5, 150, 105, 0.3);
    }

    .badge-fatty {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 60%, #f97316 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4);
    }

    .badge-cirrhosis {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 60%, #ef4444 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.45);
    }

    .badge-default {
        background: linear-gradient(135deg, #b91c1c 0%, #ea580c 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 14px;
    }

    .result-class-name {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 900;
        color: #1c1917;
        margin-bottom: 18px;
        text-transform: uppercase;
        letter-spacing: -0.02em;
    }

    .result-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 0;
        border-bottom: 1px solid #fed7aa;
        font-size: 0.98rem;
    }

    .result-row:last-child {
        border-bottom: none;
    }

    .result-row-lbl {
        color: #57534e;
        font-weight: 600;
    }

    .result-row-val {
        color: #1c1917;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .result-clinical-disclaimer {
        margin-top: 20px;
        padding: 14px 18px;
        background: #fff7ed;
        border-radius: 12px;
        border: 1px solid #fed7aa;
        color: #44403c;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* =========================================================================
       MISC BADGES, PRESETS, AND STEPS
       ========================================================================= */
    .step-pill {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.3);
    }

    .status-badge-ready {
        display: inline-block;
        background: linear-gradient(135deg, rgba(234, 88, 12, 0.25) 0%, rgba(245, 158, 11, 0.25) 100%);
        color: #ea580c;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 800;
        border: 1px solid rgba(234, 88, 12, 0.45);
    }

    .status-badge-pending {
        display: inline-block;
        background: rgba(120, 113, 108, 0.15);
        color: #78716c;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid rgba(120, 113, 108, 0.3);
    }

    .sample-preset-card {
        background: #ffffff;
        border: 1.5px solid #fed7aa;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        transition: all 0.25s ease;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    }

    .sample-preset-card:hover {
        border-color: #ea580c;
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(234, 88, 12, 0.2);
    }

    .sample-preset-card h5 {
        margin: 8px 0 4px 0;
        font-size: 1.02rem;
        font-weight: 800;
        color: #1c1917;
        font-family: 'Outfit', sans-serif;
    }

    .sample-preset-card p {
        font-size: 0.80rem;
        color: #78716c;
        margin: 0 0 10px 0;
        line-height: 1.4;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Renders the royal application title and subtitle."""
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-badge">👑 ROYAL CLINICAL & ACADEMIC AI SUITE</div>
            <h1>LIVER ULTRASOUND IMAGE CLASSIFICATION</h1>
            <p>High-Precision Machine Learning & Computer Vision Diagnostic Workstation • HOG Gradients + GLCM Texture Matrices + Multi-Classifier Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_stepper(dataset_ready: bool, features_ready: bool, model_ready: bool, model_name: str = None, total_images: int = 0):
    """Renders the royal 4-step pipeline status tracker at the top of the workstation."""
    step1_cls = "complete" if dataset_ready else "active"
    step2_cls = "complete" if features_ready else ("active" if dataset_ready else "")
    step3_cls = "complete" if model_ready else ("active" if features_ready else "")
    step4_cls = "complete" if model_ready else ""

    s1_sub = f"{total_images} Images Loaded" if dataset_ready else "Load or Scan Dataset"
    s2_sub = "HOG + GLCM Vector Ready" if features_ready else "Awaiting Feature Extraction"
    s3_sub = f"{model_name}" if model_ready else "Select & Train Classifier"
    s4_sub = "Live Prediction Available" if model_ready else "Requires Trained Model"

    s1_icon = "✓" if dataset_ready else "1"
    s2_icon = "✓" if features_ready else "2"
    s3_icon = "✓" if model_ready else "3"
    s4_icon = "✓" if model_ready else "4"

    status_tag = "🟢 ROYAL SUITE READY FOR INFERENCE" if model_ready else "🟠 PIPELINE CONFIGURATION IN PROGRESS"

    stepper_html = f"""
    <div class="pipeline-stepper-container">
        <div class="stepper-title">
            <span>⚙️ ML PIPELINE PROGRESSION TRACKER</span>
            <span>{status_tag}</span>
        </div>
        <div class="stepper-track">
            <div class="step-item {step1_cls}">
                <div class="step-num">{s1_icon}</div>
                <div>
                    <div class="step-info-title">1. Dataset</div>
                    <div class="step-info-sub">{s1_sub}</div>
                </div>
            </div>
            <div class="step-item {step2_cls}">
                <div class="step-num">{s2_icon}</div>
                <div>
                    <div class="step-info-title">2. Features</div>
                    <div class="step-info-sub">{s2_sub}</div>
                </div>
            </div>
            <div class="step-item {step3_cls}">
                <div class="step-num">{s3_icon}</div>
                <div>
                    <div class="step-info-title">3. Model</div>
                    <div class="step-info-sub">{s3_sub}</div>
                </div>
            </div>
            <div class="step-item {step4_cls}">
                <div class="step-num">{s4_icon}</div>
                <div>
                    <div class="step-info-title">4. Inference</div>
                    <div class="step-info-sub">{s4_sub}</div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(stepper_html, unsafe_allow_html=True)


def render_disclaimer():
    """Renders the educational and research disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Academic & Research Notice:</strong> This clinical AI suite is engineered for academic research, coursework demonstration, and benchmark evaluation. It does not replace certified radiological and medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_cards(accuracy: float, precision: float, recall: float, f1: float):
    """Renders royal 4-metric score cards with Red & Orange accents."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="tag">Overall Accuracy</div>
                <div class="val">{accuracy * 100:.2f}%</div>
                <div class="desc">Global Classification Match</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="tag">Precision (Weighted)</div>
                <div class="val">{precision * 100:.2f}%</div>
                <div class="desc">Positive Predictive Value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="tag">Recall (Sensitivity)</div>
                <div class="val">{recall * 100:.2f}%</div>
                <div class="desc">True Positive Catch Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="tag">F1-Score (Harmonic)</div>
                <div class="val">{f1 * 100:.2f}%</div>
                <div class="desc">Precision-Recall Balance</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """
    Renders an enhanced Seaborn confusion matrix heatmap in Royal Orange-Red colormap.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_pct = np.nan_to_num(cm_pct)

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="OrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.5,
        linecolor="#ffffff",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold", "color": "#1c1917"}
    )
    ax.set_title("Ultrasound Confusion Matrix", fontsize=12, weight="bold", pad=15, color="#1c1917")
    ax.set_xlabel("Predicted Class", fontsize=10, weight="bold", labelpad=10, color="#450a0a")
    ax.set_ylabel("Actual Ground Truth", fontsize=10, weight="bold", labelpad=10, color="#450a0a")
    plt.tight_layout()
    return fig


def plot_class_distribution(class_counts: dict):
    """
    Plots a polished horizontal bar chart of dataset classes in Royal Red & Sunburst Orange.
    """
    fig, ax = plt.subplots(figsize=(7, max(3, len(class_counts) * 0.75)))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    royal_palette = ["#dc2626", "#ea580c", "#f59e0b", "#991b1b", "#c2410c"]
    colors = [royal_palette[i % len(royal_palette)] for i in range(len(classes))]

    bars = ax.barh(classes, counts, color=colors, height=0.55, edgecolor="#7f1d1d", alpha=0.9)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(max(counts) * 0.02, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)} scans",
            va="center",
            ha="left",
            fontsize=9.5,
            weight="bold",
            color="#1c1917"
        )

    ax.set_xlabel("Number of Ultrasound Scans", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_ylabel("Diagnostic Class", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_title("Dataset Class Sample Distribution", fontsize=11.5, weight="bold", pad=12, color="#1c1917")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#fed7aa")
    ax.spines["bottom"].set_color("#fed7aa")
    plt.tight_layout()
    return fig


def plot_glcm_radar(glcm_summary: dict):
    """
    Plots a radar (spider) chart for normalized GLCM texture properties with Red-Orange styling.
    """
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    labels = [p.capitalize() for p in properties]

    raw_values = [glcm_summary[p]["mean"] for p in properties]
    norm_values = []
    for val in raw_values:
        norm_values.append(min(1.0, max(0.05, abs(val) if abs(val) <= 1.0 else np.log10(abs(val) + 1.0) / 2.0)))

    values = norm_values + [norm_values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#ea580c", alpha=0.35)
    ax.plot(angles, values, color="#dc2626", linewidth=2.5, marker="o", markersize=6.5)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9.5, weight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("GLCM Second-Order Texture Radar", fontsize=11.5, weight="bold", pad=15, color="#1c1917")
    plt.tight_layout()
    return fig


def plot_roc_curves(roc_data: dict):
    """
    Plots Multi-Class One-vs-Rest ROC Curves using a Royal palette.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    
    royal_roc_colors = ["#dc2626", "#ea580c", "#d97706", "#991b1b", "#b45309"]
    for idx, (cls_name, data) in enumerate(roc_data.items()):
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        color = royal_roc_colors[idx % len(royal_roc_colors)]
        ax.plot(
            fpr, tpr,
            label=f"{cls_name} (AUC = {roc_auc:.3f})",
            color=color,
            lw=2.4
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_title("Multi-Class ROC Performance Curves", fontsize=11.5, weight="bold", pad=12, color="#1c1917")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#fed7aa")
    plt.tight_layout()
    return fig


def plot_model_comparison_chart(comparison_df: pd.DataFrame):
    """
    Plots a multi-metric comparison bar chart for all algorithms in Royal Red & Sunburst Orange.
    """
    df_melted = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    royal_algo_palette = ["#dc2626", "#ea580c", "#f59e0b", "#991b1b"]
    
    sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Algorithm",
        palette=royal_algo_palette,
        ax=ax,
        edgecolor="#7f1d1d"
    )

    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_xlabel("Evaluation Metric", fontsize=9.5, weight="bold", color="#450a0a")
    ax.set_title("Machine Learning Classifiers Performance Benchmark", fontsize=11.5, weight="bold", pad=12, color="#1c1917")
    ax.legend(title="Algorithm", frameon=True, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#fed7aa")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    return fig


def render_prediction_result(
    predicted_class: str,
    algorithm: str,
    confidence: float = None,
    probabilities: dict = None,
    feature_summary: dict = None
):
    """
    Renders the standardized prediction result card with dynamic royal severity styling.
    """
    conf_str = f"{confidence * 100:.2f}%" if confidence is not None else "N/A (Decision Boundary Model)"

    p_lower = predicted_class.lower()
    if "normal" in p_lower:
        card_class = "result-card-normal"
        badge_class = "badge-normal"
        severity_label = "🟢 NORMAL / HEALTHY HEPATIC PARENCHYMA"
        clinical_note = "Acoustic texture exhibits homogeneous medium echogenicity with uniform speckle distribution and clear vascular architecture."
    elif "fatty" in p_lower:
        card_class = "result-card-fatty"
        badge_class = "badge-fatty"
        severity_label = "🟡 FATTY LIVER (STEATOSIS) DETECTED"
        clinical_note = "Acoustic texture shows diffuse parenchymal hyperechogenicity ('bright liver') with characteristic acoustic sound attenuation."
    elif "cirrhosis" in p_lower or "fibrosis" in p_lower:
        card_class = "result-card-cirrhosis"
        badge_class = "badge-cirrhosis"
        severity_label = "🔴 CIRRHOSIS / ADVANCED FIBROSIS DETECTED"
        clinical_note = "Acoustic texture demonstrates coarse heterogeneous nodularity with irregular hyperechoic fibrotic reticular acoustic bands."
    else:
        card_class = "result-card-default"
        badge_class = "badge-default"
        severity_label = f"👑 CLASSIFIED: {predicted_class.upper()}"
        clinical_note = "Ultrasound scan classified according to learned HOG gradient and GLCM texture vectors."

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="{badge_class}">{severity_label}</div>
            <div class="result-class-name">Diagnostic Result: {predicted_class}</div>
            
            <div class="result-row">
                <span class="result-row-lbl">🤖 Active Classifier:</span>
                <span class="result-row-val">{algorithm}</span>
            </div>
            <div class="result-row">
                <span class="result-row-lbl">🔍 Feature Vector:</span>
                <span class="result-row-val">HOG Gradients + GLCM Texture (Multi-Angular Fused)</span>
            </div>
            <div class="result-row">
                <span class="result-row-lbl">🎯 Confidence Level:</span>
                <span class="result-row-val">{conf_str}</span>
            </div>
            
            <div class="result-clinical-disclaimer">
                <strong>🔬 Acoustic Texture Insight:</strong> {clinical_note}<br>
                <small style="color:#78716c; margin-top:5px; display:block;">⚠️ <em>For educational and academic presentation purposes only. Not a medical diagnosis.</em></small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if probabilities and len(probabilities) > 1:
        st.markdown("#### 📊 Multi-Class Probability Breakdown")
        prob_cols = st.columns(len(probabilities))
        for i, (cls_name, prob) in enumerate(probabilities.items()):
            with prob_cols[i % len(prob_cols)]:
                st.metric(
                    label=f"Class: {cls_name}",
                    value=f"{prob * 100:.1f}%"
                )
                st.progress(float(prob))


def generate_diagnostic_text_report(
    predicted_class: str,
    algorithm: str,
    confidence: float,
    probabilities: dict,
    image_name: str
) -> str:
    """Generates a downloadable markdown academic report for the classification result."""
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_str = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"
    
    prob_lines = ""
    if probabilities:
        prob_lines = "\n".join([f"- **{k}**: {v*100:.2f}%" for k, v in probabilities.items()])
    else:
        prob_lines = "- Not available for this model type"

    report = f"""# 🩺 Liver Ultrasound Image Classification Report
**Generated by:** Liver Ultrasound ML Classifier System (Royal Suite)  
**Timestamp:** {now_str}  
**Input Image Reference:** `{image_name}`  

---

## 📌 Executive Diagnostic Summary
- **Predicted Class:** **{predicted_class}**
- **Classification Model:** {algorithm}
- **Confidence Level:** {conf_str}

## 📊 Class Probability Breakdown
{prob_lines}

## 🔬 Feature Extraction Methodology
1. **Grayscale Conversion & Standardization:** Resized to 128x128 pixels.
2. **Noise Reduction:** Gaussian Blur spatial filtering for ultrasound speckle suppression.
3. **Contrast Enhancement:** Histogram Equalization / CLAHE to accentuate parenchymal textures.
4. **HOG (Histogram of Oriented Gradients):** 9 Orientations, 8x8 pixels per cell, 2x2 cells per block.
5. **GLCM (Gray-Level Co-occurrence Matrix):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation across 4 angles (0°, 45°, 90°, 135°).

---
*Notice: This report was generated by an academic machine learning research workstation and is intended strictly for educational evaluation and project demonstration.*
"""
    return report
