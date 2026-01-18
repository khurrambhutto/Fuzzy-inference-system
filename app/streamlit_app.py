"""
Heart Disease Risk Assessment - Streamlit UI

A beautiful, interactive demo of the Fuzzy Inference System.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from membership_functions import MembershipFunctions as MF
from fuzzy_system import HeartDiseaseFIS
from inference import FuzzyInference


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Heart Disease Risk - Fuzzy FIS",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.8rem;
    }
    
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    .metric-box {
        background: linear-gradient(145deg, #1e293b, #334155);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        color: #f1f5f9;
    }
    
    .step-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .sidebar .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #374151, #4b5563) !important;
        border-radius: 10px;
        padding: 10px 20px;
        color: #e5e7eb !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(145deg, #4b5563, #6b7280) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# INITIALIZE SYSTEMS
# ============================================
@st.cache_resource
def load_fis():
    """Load and cache the fuzzy inference system."""
    return HeartDiseaseFIS()

@st.cache_resource
def load_inference():
    """Load inference engine."""
    return FuzzyInference()

fis = load_fis()
inference_engine = load_inference()


# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Risk Assessment</h1>
    <p>Fuzzy Inference System using Mamdani Method</p>
</div>
""", unsafe_allow_html=True)


# ============================================
# SIDEBAR - INPUT CONTROLS
# ============================================
with st.sidebar:
    st.markdown("## 🎛️ Patient Parameters")
    st.markdown("---")
    
    age = st.slider(
        "🎂 Age (years)",
        min_value=29, max_value=77, value=50,
        help="Patient's age in years"
    )
    
    trestbps = st.slider(
        "💉 Resting Blood Pressure (mm Hg)",
        min_value=90, max_value=200, value=120,
        help="Resting blood pressure on admission"
    )
    
    chol = st.slider(
        "🧪 Cholesterol (mg/dl)",
        min_value=120, max_value=564, value=200,
        help="Serum cholesterol level"
    )
    
    thalach = st.slider(
        "💓 Max Heart Rate (bpm)",
        min_value=70, max_value=202, value=150,
        help="Maximum heart rate achieved during exercise"
    )
    
    oldpeak = st.slider(
        "📉 ST Depression (oldpeak)",
        min_value=0.0, max_value=6.2, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    show_inference = st.checkbox("Show Inference Steps", value=True)
    show_mf = st.checkbox("Show Membership Functions", value=True)


# ============================================
# MAIN CONTENT
# ============================================
# Calculate risk
risk_score = fis.predict(age, trestbps, chol, thalach, oldpeak)
risk_label = fis.get_risk_label(risk_score)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Results", "🔍 Inference Details", "📈 Visualizations"])
    
    with tab1:
        st.markdown("### Risk Assessment Result")
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk (%)", 'font': {'size': 24, 'color': '#1a1a2e'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#1a1a2e"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#1a1a2e",
                'steps': [
                    {'range': [0, 35], 'color': '#38ef7d'},
                    {'range': [35, 65], 'color': '#ffd166'},
                    {'range': [65, 100], 'color': '#ef476f'}
                ],
                'threshold': {
                    'line': {'color': "#1a1a2e", 'width': 4},
                    'thickness': 0.8,
                    'value': risk_score * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Poppins'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Input summary
        st.markdown("### 📋 Input Summary")
        input_df = pd.DataFrame({
            'Parameter': ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression'],
            'Value': [f"{age} years", f"{trestbps} mm Hg", f"{chol} mg/dl", f"{thalach} bpm", f"{oldpeak}"],
            'Status': ['🟢' if age < 50 else '🟡' if age < 60 else '🔴',
                      '🟢' if trestbps < 130 else '🟡' if trestbps < 150 else '🔴',
                      '🟢' if chol < 200 else '🟡' if chol < 240 else '🔴',
                      '🟢' if thalach > 150 else '🟡' if thalach > 100 else '🔴',
                      '🟢' if oldpeak < 1 else '🟡' if oldpeak < 2.5 else '🔴']
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    with tab2:
        if show_inference:
            # Run detailed inference
            inputs = {
                'age': age,
                'trestbps': trestbps,
                'chol': chol,
                'thalach': thalach,
                'oldpeak': oldpeak
            }
            result = inference_engine.infer(inputs)
            
            # Step 1: Fuzzification
            st.markdown('<div class="step-header">Step 1: Fuzzification</div>', unsafe_allow_html=True)
            
            fuzz_data = []
            for var, terms in result['fuzzified'].items():
                for term, degree in terms.items():
                    if degree > 0.01:
                        fuzz_data.append({
                            'Variable': var.upper(),
                            'Term': term.capitalize(),
                            'Membership': f"{degree:.3f}"
                        })
            
            if fuzz_data:
                st.dataframe(pd.DataFrame(fuzz_data), use_container_width=True, hide_index=True)
            
            # Step 2: Rule Evaluation
            st.markdown('<div class="step-header">Step 2: Rule Evaluation</div>', unsafe_allow_html=True)
            
            rule_data = []
            for rule, (activation, consequent) in result['rule_activations'].items():
                if activation > 0.01:
                    rule_data.append({
                        'Rule': rule,
                        'Activation': f"{activation:.3f}",
                        'Output': consequent.capitalize()
                    })
            
            if rule_data:
                st.dataframe(pd.DataFrame(rule_data), use_container_width=True, hide_index=True)
            else:
                st.info("No rules significantly activated for these inputs.")
            
            # Step 3 & 4: Aggregation & Defuzzification
            st.markdown('<div class="step-header">Step 3 & 4: Aggregation & Defuzzification</div>', unsafe_allow_html=True)
            
            universe, aggregated = result['aggregated']
            
            fig_agg = go.Figure()
            fig_agg.add_trace(go.Scatter(
                x=universe, y=aggregated,
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=2),
                name='Aggregated Output'
            ))
            fig_agg.add_vline(
                x=result['risk'], 
                line_dash="dash", 
                line_color="#e63946",
                annotation_text=f"Defuzzified: {result['risk']:.3f}"
            )
            fig_agg.update_layout(
                title="Aggregated Fuzzy Output",
                xaxis_title="Risk",
                yaxis_title="Membership",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_agg, use_container_width=True)
        else:
            st.info("Enable 'Show Inference Steps' in sidebar to view details.")
    
    with tab3:
        if show_mf:
            st.markdown("### Membership Functions")
            
            # Create subplot for all MFs
            fig_mf = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Age', 'Blood Pressure', 'Cholesterol', 
                               'Max Heart Rate', 'ST Depression', 'Risk Output')
            )
            
            colors = {'young': '#38ef7d', 'middle': '#ffd166', 'old': '#ef476f',
                     'low': '#38ef7d', 'normal': '#ffd166', 'high': '#ef476f',
                     'medium': '#ffd166'}
            
            variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'risk']
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
            input_values = [age, trestbps, chol, thalach, oldpeak, risk_score]
            
            for var, pos, val in zip(variables, positions, input_values):
                universe = MF.get_universe(var, 0.1 if var in ['oldpeak', 'risk'] else 1)
                
                for term in MF.get_all_terms(var):
                    mf = MF.get_membership(var, term, universe)
                    color = colors.get(term, '#667eea')
                    
                    fig_mf.add_trace(
                        go.Scatter(
                            x=universe, y=mf,
                            name=f"{term}",
                            line=dict(color=color, width=2),
                            showlegend=(var == 'age')
                        ),
                        row=pos[0], col=pos[1]
                    )
                
                # Add vertical line for current value
                fig_mf.add_vline(
                    x=val, 
                    line_dash="dash", 
                    line_color="black",
                    row=pos[0], col=pos[1]
                )
            
            fig_mf.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_mf, use_container_width=True)
        else:
            st.info("Enable 'Show Membership Functions' in sidebar to view.")


with col2:
    # Risk Result Card
    if risk_label == "Low":
        css_class = "risk-low"
        emoji = "✅"
        message = "Low risk of heart disease"
    elif risk_label == "Medium":
        css_class = "risk-medium"
        emoji = "⚠️"
        message = "Moderate risk - consult doctor"
    else:
        css_class = "risk-high"
        emoji = "🚨"
        message = "High risk - immediate attention"
    
    st.markdown(f"""
    <div class="risk-card {css_class}">
        <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
        <h2 style="font-size: 2rem; margin: 0.5rem 0;">{risk_label} Risk</h2>
        <p style="font-size: 1.1rem; opacity: 0.9;">{message}</p>
        <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
        <h3 style="font-size: 2.5rem; margin: 0;">{risk_score:.1%}</h3>
        <p style="font-size: 0.9rem; opacity: 0.8;">Risk Score</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    
    st.markdown(f"""
    <div class="metric-box">
        <strong>Classification:</strong> {"Disease Likely" if risk_score >= 0.5 else "No Disease"}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-box">
        <strong>Confidence:</strong> {abs(risk_score - 0.5) * 2:.1%}
    </div>
    """, unsafe_allow_html=True)
    
    # Rules info
    st.markdown("### 📜 Active Rules")
    inputs = {'age': age, 'trestbps': trestbps, 'chol': chol, 
              'thalach': thalach, 'oldpeak': oldpeak}
    result = inference_engine.infer(inputs)
    
    active_rules = [(r, a, t) for r, (a, t) in result['rule_activations'].items() if a > 0.1]
    active_rules.sort(key=lambda x: x[1], reverse=True)
    
    for rule, activation, term in active_rules[:5]:
        color = "#38ef7d" if term == "low" else "#ffd166" if term == "medium" else "#ef476f"
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {color}33, transparent); 
                    padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0;
                    border-left: 3px solid {color};">
            <strong>{rule}</strong>: {activation:.2f} → {term}
        </div>
        """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Heart Disease Risk Assessment using Mamdani Fuzzy Inference System</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • scikit-fuzzy • Plotly</p>
</div>
""", unsafe_allow_html=True)

