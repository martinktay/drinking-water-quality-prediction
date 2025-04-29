"""Main application module for the Drinking Water Quality Prediction system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib
import json
import yaml
from typing import Dict, Any, Optional, Tuple

from models import train_and_save_models
from utils.logging import setup_logging
from utils.startup import perform_startup_checks

# Configure logging
setup_logging(log_file='logs/app.log')

# Perform startup checks
startup_warnings = perform_startup_checks()

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    st.error(f"Failed to load configuration: {str(e)}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Drinking Water Quality Prediction",
    layout="wide"
)

# Display startup warnings if any
if startup_warnings:
    with st.expander("⚠️ Startup Warnings", expanded=True):
        for warning in startup_warnings:
            st.warning(warning)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #ffffff;
        margin-bottom: 2.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 2.8rem;
        color: #ffffff;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #2c3e50;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 1.5rem 0;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 1.6rem;
        color: #bdc3c7;
    }
    .best-model {
        background-color: #28a745;
        color: white;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1.5rem 0;
        font-size: 1.4rem;
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .stDataFrame {
        background-color: #2c3e50;
        color: #ffffff;
        font-size: 1.2rem;
    }
    .stMarkdown {
        color: #ffffff;
        font-size: 1.4rem;
    }
    /* Increase font size for all text elements */
    p, div {
        font-size: 1.2rem !important;
    }
    /* Make input labels larger */
    .st-bw {
        font-size: 1.3rem !important;
    }
    /* Increase size of number inputs */
    .stNumberInput [data-baseweb="input"] {
        font-size: 1.3rem !important;
    }
    /* Make buttons larger */
    .stButton button {
        font-size: 1.4rem !important;
        padding: 0.8rem 1.6rem !important;
    }
    /* Increase size of help text */
    .stTooltipIcon {
        font-size: 1.2rem !important;
    }
    /* Make headers in markdown larger */
    h1 {
        font-size: 3rem !important;
    }
    h2 {
        font-size: 2.5rem !important;
    }
    h3 {
        font-size: 2rem !important;
    }
    /* Increase size of plot labels and titles */
    .js-plotly-plot .plotly .main-svg text {
        font-size: 1.4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Handle connection errors


def load_data_safely():
    try:
        df = pd.read_csv('data/processed/processed_data.csv', index_col=False)
        # Remove any unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def load_model_safely(model_name):
    try:
        model_path = f'models/{model_name}.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception as e:
        st.warning(f"Could not load model {model_name}: {str(e)}")
        return None


def load_metrics_safely():
    metrics = {'initial': None, 'tuned': None}
    try:
        with open('reports/model_performance/model_metrics.json', 'r') as f:
            metrics['initial'] = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open('reports/model_performance/model_metrics_tuned.json', 'r') as f:
            metrics['tuned'] = json.load(f)
    except FileNotFoundError:
        pass
    return metrics

# Main application


def main():
    """Main application function."""
    st.markdown("<h1 class='main-header'>Drinking Water Quality Prediction</h1>",
                unsafe_allow_html=True)

    # Data Overview Section
    st.markdown("<h2 class='section-header'>📊 Data Overview</h2>",
                unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df = load_data_safely()

    if df is not None:
        # Feature Descriptions
        st.markdown("### Feature Descriptions")
        st.write("The following water quality parameters are used for prediction:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **pH (0-14)**  
            Measures acidity/alkalinity  
            Safe range: 6.5-8.5
            
            **Iron (mg/L)**  
            Essential mineral  
            Safe limit: <0.3 mg/L
            
            **Nitrate (mg/L)**  
            Common contaminant  
            Safe limit: <10 mg/L
            
            **Chloride (mg/L)**  
            Natural mineral  
            Safe limit: <250 mg/L
            
            **Lead (mg/L)**  
            Toxic metal  
            Safe limit: <0.015 mg/L
            """)

        with col2:
            st.markdown("""
            **Zinc (mg/L)**  
            Essential mineral  
            Safe limit: <5 mg/L
            
            **Turbidity (NTU)**  
            Water clarity  
            Safe limit: <5 NTU
            
            **Fluoride (mg/L)**  
            Dental health  
            Safe range: 0.7-1.2 mg/L
            
            **Copper (mg/L)**  
            Essential mineral  
            Safe limit: <1.3 mg/L
            
            **Sulfate (mg/L)**  
            Natural mineral  
            Safe limit: <250 mg/L
            """)

        with col3:
            st.markdown("""
            **Conductivity (µS/cm)**  
            Dissolved ions  
            Safe range: 50-1500 µS/cm
            
            **Chlorine (mg/L)**  
            Disinfectant  
            Safe range: 0.2-4.0 mg/L
            
            **Total Dissolved Solids (mg/L)**  
            Overall water quality  
            Safe limit: <1000 mg/L
            
            **Water Temperature (°C)**  
            Affects treatment  
            Safe range: 0-30°C
            
            **Air Temperature (°C)**  
            Environmental factor  
            Normal range: -5-40°C
            """)

        # Display basic statistics
        st.markdown("### Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # Target Distribution
        st.markdown("### Water Quality Distribution")
        target_counts = df['Target'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title='Water Quality Distribution',
            color=target_counts.index,
            color_discrete_map={0: '#dc3545', 1: '#28a745'},
            labels={0: 'Not Potable', 1: 'Potable'}
        )
        fig.update_layout(
            title_font_size=20,
            showlegend=True,
            legend_title_text='Water Quality',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model Performance Section
    st.markdown("<h2 class='section-header'>📈 Model Performance</h2>",
                unsafe_allow_html=True)

    metrics = load_metrics_safely()

    if metrics['initial']:
        st.markdown("### Initial Model Performance")
        df_initial = pd.DataFrame(metrics['initial']).T
        st.dataframe(df_initial.style.format("{:.4f}"), use_container_width=True)

        # Visualize initial model performance
        fig = px.bar(
            df_initial,
            title='Initial Model Performance Comparison',
            labels={'value': 'Score', 'index': 'Metric'},
            barmode='group'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    if metrics['tuned']:
        st.markdown("### Tuned Model Performance")
        df_tuned = pd.DataFrame(metrics['tuned']).T
        st.dataframe(df_tuned.style.format("{:.4f}"), use_container_width=True)

        # Find and highlight best model
        best_model, best_f1 = find_best_model(metrics)
        if best_model:
            st.markdown(f"""
            <div class="best-model">
                <h3>Best Performing Model: {best_model}</h3>
                <p>F1 Score: {best_f1:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Visualize tuned model performance
        fig = px.bar(
            df_tuned,
            title='Tuned Model Performance Comparison',
            labels={'value': 'Score', 'index': 'Metric'},
            barmode='group'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    # Prediction Section
    st.markdown("<h2 class='section-header'>🔮 Water Quality Prediction</h2>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", value=7.0, min_value=0.0,
                             max_value=14.0, help="Normal: 6.5-8.5")
        iron = st.number_input("Iron (mg/L)", value=0.1,
                               min_value=0.0, help="Normal: 0.0-0.3")
        nitrate = st.number_input("Nitrate (mg/L)", value=5.0,
                                  min_value=0.0, help="Normal: 0.0-10")
        chloride = st.number_input(
            "Chloride (mg/L)", value=100.0, min_value=0.0, help="Normal: 0-250")
        lead = st.number_input("Lead (mg/L)", value=0.005,
                               min_value=0.0, help="Normal: 0.0-0.015")

    with col2:
        zinc = st.number_input("Zinc (mg/L)", value=2.0,
                               min_value=0.0, help="Normal: 0.0-5.0")
        turbidity = st.number_input(
            "Turbidity (NTU)", value=2.0, min_value=0.0, help="Normal: 0.0-5.0")
        fluoride = st.number_input("Fluoride (mg/L)", value=1.0,
                                   min_value=0.0, help="Normal: 0.0-2.0")
        copper = st.number_input("Copper (mg/L)", value=0.5,
                                 min_value=0.0, help="Normal: 0.0-1.3")
        sulfate = st.number_input("Sulfate (mg/L)", value=100.0,
                                  min_value=0.0, help="Normal: 0-250")

    with col3:
        conductivity = st.number_input(
            "Conductivity (µS/cm)", value=500.0, min_value=0.0, help="Normal: 50-1500")
        chlorine = st.number_input("Chlorine (mg/L)", value=1.0,
                                   min_value=0.0, help="Normal: 0.2-4.0")
        tds = st.number_input("Total Dissolved Solids (mg/L)",
                              value=500.0, min_value=0.0, help="Normal: 0-1000")
        water_temp = st.number_input(
            "Water Temperature (°C)", value=20.0, help="Normal: 0-30")
        air_temp = st.number_input("Air Temperature (°C)",
                                   value=25.0, help="Normal: -5-40")

    if st.button("Predict Water Quality"):
        input_data = pd.DataFrame([[
            ph, iron, nitrate, chloride, lead, zinc, turbidity,
            fluoride, copper, sulfate, conductivity, chlorine,
            tds, water_temp, air_temp
        ]], columns=[
            'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
            'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
            'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
            'Air Temperature'
        ])

        # Load models and make predictions
        for model_name in ['randomforest', 'xgboost', 'lightgbm', 'decisiontree', 'linearsvc']:
            try:
                model = joblib.load(f'models/{model_name}.pkl')
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] if hasattr(
                    model, 'predict_proba') else 0.5

                prediction = "Safe" if pred == 1 else "Unsafe"
                color = "#28a745" if prediction == "Safe" else "#dc3545"

                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                    <h3>{model_name.title()}</h3>
                    <p>Prediction: {prediction}</p>
                    <p>Confidence: {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")


def find_best_model(metrics):
    if not metrics['tuned']:
        return None, None

    best_model = None
    best_f1 = 0
    for model_name, model_metrics in metrics['tuned'].items():
        if model_metrics['f1'] > best_f1:
            best_f1 = model_metrics['f1']
            best_model = model_name
    return best_model, best_f1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An error occurred. Please refresh the page.")
        st.error(f"Error details: {str(e)}")
