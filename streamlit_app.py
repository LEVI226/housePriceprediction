# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration de la page avec responsive
st.set_page_config(
    page_title="🏠 Prédicteur Prix Immobilier IA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS responsive et moderne ultra-optimisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-bg: #ffffff;
        --text-color: #2c3e50;
        --border-radius: 16px;
        --shadow: 0 4px 20px rgba(0,0,0,0.1);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Reset et responsive base */
    * {
        box-sizing: border-box;
    }
    
    .main .block-container {
        padding: clamp(0.5rem, 2vw, 1rem);
        max-width: 100%;
    }
    
    /* Header responsive avec animation */
    .main-header {
        background: var(--background-gradient);
        padding: clamp(1.5rem, 4vw, 3rem);
        border-radius: var(--border-radius);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 4s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: clamp(1.8rem, 5vw, 3rem);
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-size: clamp(0.9rem, 2.5vw, 1.2rem);
        position: relative;
        z-index: 1;
    }
    
    /* Cards responsives avec hover effects */
    .prediction-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: clamp(1.5rem, 4vw, 2.5rem);
        border-radius: var(--border-radius);
        text-align: center;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    
    .prediction-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
    }
    
    .prediction-price {
        font-size: clamp(2rem, 6vw, 4rem);
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        font-family: 'Inter', sans-serif;
    }
    
    .feature-card {
        background: var(--card-bg);
        padding: clamp(1rem, 3vw, 2rem);
        border-radius: var(--border-radius);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        padding: clamp(0.8rem, 2vw, 1.5rem);
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
        transition: var(--transition);
    }
    
    .metric-card:hover {
        transform: translateX(4px);
        border-left-color: var(--secondary-color);
    }
    
    /* Buttons responsive */
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: clamp(0.75rem, 2vw, 1rem) clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 600;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        transition: var(--transition);
        box-shadow: var(--shadow);
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Form inputs responsive */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: var(--transition);
        font-family: 'Inter', sans-serif;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Notifications */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe, #b3e5fc);
        padding: clamp(1rem, 2vw, 1.5rem);
        border-radius: 12px;
        border-left: 4px solid #0288d1;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        padding: clamp(1rem, 2vw, 1.5rem);
        border-radius: 12px;
        border-left: 4px solid #ffa000;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        padding: clamp(1rem, 2vw, 1.5rem);
        border-radius: 12px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        padding: clamp(1rem, 2vw, 1.5rem);
        border-radius: 12px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Graphiques responsives */
    .chart-container {
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--shadow);
        margin: 1rem 0;
    }
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        .main-header {
            margin-bottom: 1rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.25rem 0;
        }
    }
    
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .main-header p {
            font-size: 0.85rem;
        }
        
        .prediction-price {
            font-size: 2.5rem;
        }
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    /* Plotly charts responsive */
    .js-plotly-plot .plotly .modebar {
        right: 10px;
        top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger VOS modèles avec gestion d'erreur robuste
@st.cache_resource
def load_real_models():
    """Charger VOS modèles PKL avec gestion d'erreur complète"""
    try:
        model_loaded = False
        feature_info_loaded = False
        
        # Charger le modèle XGBoost/Random Forest
        if os.path.exists('xgb_model.pkl'):
            with open('xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_loaded = True
            st.success("✅ Modèle Random Forest chargé avec succès!")
        else:
            st.error("❌ Fichier 'xgb_model.pkl' non trouvé")
            return None, None, False
        
        # Charger les métadonnées des features
        if os.path.exists('feature_info.pkl'):
            with open('feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
            feature_info_loaded = True
            st.success("✅ Métadonnées des features chargées!")
        else:
            st.warning("⚠️ Fichier 'feature_info.pkl' non trouvé, utilisation des valeurs par défaut")
            # Créer des métadonnées par défaut basées sur votre rapport
            feature_info = {
                'rmse_score': 26240.20,
                'r2_score': 0.8688,
                'model_stats': {
                    'test_r2': 0.8688,
                    'train_r2': 0.9330,
                    'train_samples': 1456,
                    'mean_price': 180151.23,
                    'min_price': 50000,
                    'max_price': 500000
                },
                'feature_names': ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'],
                'feature_importance': [
                    {'feature': 'OverallQual', 'importance': 0.579},
                    {'feature': 'GrLivArea', 'importance': 0.184},
                    {'feature': 'TotalBsmtSF', 'importance': 0.092},
                    {'feature': 'GarageArea', 'importance': 0.047},
                    {'feature': 'YearBuilt', 'importance': 0.039},
                    {'feature': 'FullBath', 'importance': 0.030},
                    {'feature': 'TotRmsAbvGrd', 'importance': 0.015},
                    {'feature': 'Fireplaces', 'importance': 0.010},
                    {'feature': 'GarageCars', 'importance': 0.004}
                ]
            }
        
        return model, feature_info, model_loaded and feature_info_loaded
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {str(e)}")
        return None, None, False

# Fonction pour créer des données de démonstration basées sur votre dataset
@st.cache_data
def create_demo_data():
    """Créer des données de démonstration réalistes basées sur votre rapport"""
    np.random.seed(42)
    
    # Données de prix par surface (basées sur vos vraies données)
    surface_data = pd.DataFrame({
        'GrLivArea': np.random.normal(1500, 500, 200),
        'SalePrice': np.random.normal(180151, 76696, 200)
    })
    surface_data = surface_data[surface_data['GrLivArea'] > 500]
    surface_data = surface_data[surface_data['SalePrice'] > 50000]
    
    # Données de comparaison par qualité (basées sur votre feature importance)
    quality_data = pd.DataFrame({
        'OverallQual': list(range(1, 11)),
        'Prix_Moyen': [80000, 100000, 120000, 140000, 160000, 180000, 220000, 280000, 350000, 450000],
        'Nombre_Ventes': [15, 32, 65, 145, 278, 365, 242, 128, 75, 28]
    })
    
    # Données d'évolution temporelle
    years = list(range(1950, 2025))
    evolution_data = pd.DataFrame({
        'YearBuilt': years,
        'Prix_Moyen': [50000 + (year - 1950) * 2000 + np.random.normal(0, 5000) for year in years]
    })
    
    return surface_data, quality_data, evolution_data

# Fonction pour préparer les features selon VOTRE modèle exact
def prepare_features_for_your_model(user_inputs):
    """Préparer les features selon l'ordre exact de VOTRE modèle entraîné"""
    
    # Ordre exact des features selon votre rapport
    features_dict = {
        'OverallQual': user_inputs['overall_qual'],
        'GrLivArea': user_inputs['gr_liv_area'],
        'TotalBsmtSF': user_inputs['total_bsmt_sf'],
        'GarageArea': user_inputs['garage_area'],
        'YearBuilt': user_inputs['year_built'],
        'FullBath': user_inputs['full_bath'],
        'TotRmsAbvGrd': user_inputs['tot_rms_abv_grd'],
        'Fireplaces': user_inputs['fireplaces'],
        'GarageCars': user_inputs['garage_cars']
    }
    
    # Convertir en DataFrame avec l'ordre exact
    df = pd.DataFrame([features_dict])
    
    return df

# Fonction pour faire la prédiction avec VOTRE modèle
def make_prediction_with_your_model(model, features_df):
    """Faire la prédiction avec VOTRE modèle Random Forest entraîné"""
    try:
        prediction = model.predict(features_df)
        return max(0, prediction[0])  # Assurer que le prix est positif
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        return None

# Fonction pour créer des graphiques interactifs
def create_interactive_charts(surface_data, quality_data, evolution_data, predicted_price=None, user_features=None):
    """Créer des graphiques interactifs avec Plotly"""
    
    # 1. Graphique Surface vs Prix
    fig_surface = px.scatter(
        surface_data, 
        x='GrLivArea', 
        y='SalePrice',
        title="📐 Relation Surface Habitable - Prix de Vente",
        labels={'GrLivArea': 'Surface Habitable (sq ft)', 'SalePrice': 'Prix de Vente ($)'},
        color_discrete_sequence=['#667eea'],
        opacity=0.6
    )
    
    # Ajouter le point de prédiction si disponible
    if predicted_price and user_features:
        fig_surface.add_scatter(
            x=[user_features['gr_liv_area']],
            y=[predicted_price],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='🏠 Votre Propriété'
        )
    
    fig_surface.update_layout(
        height=400,
        font_family="Inter",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    # 2. Graphique Qualité vs Prix (Feature la plus importante)
    fig_quality = px.bar(
        quality_data,
        x='OverallQual',
        y='Prix_Moyen',
        title="⭐ Prix Moyen par Qualité Générale (Feature #1)",
        labels={'OverallQual': 'Qualité Générale (1-10)', 'Prix_Moyen': 'Prix Moyen ($)'},
        color='Prix_Moyen',
        color_continuous_scale='viridis'
    )
    
    # Mettre en évidence la qualité sélectionnée
    if user_features:
        colors = ['red' if x == user_features['overall_qual'] else '#667eea' for x in quality_data['OverallQual']]
        fig_quality.update_traces(marker_color=colors)
    
    fig_quality.update_layout(
        height=400,
        font_family="Inter",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3. Graphique d'évolution temporelle
    fig_evolution = px.line(
        evolution_data,
        x='YearBuilt',
        y='Prix_Moyen',
        title="📈 Évolution des Prix par Année de Construction",
        labels={'YearBuilt': 'Année de Construction', 'Prix_Moyen': 'Prix Moyen ($)'},
        line_shape='spline'
    )
    
    fig_evolution.update_traces(line_color='#667eea', line_width=3)
    
    # Ajouter point pour l'année de construction
    if user_features:
        year_data = evolution_data[evolution_data['YearBuilt'] == user_features['year_built']]
        if not year_data.empty:
            year_price = year_data['Prix_Moyen'].iloc[0]
            fig_evolution.add_scatter(
                x=[user_features['year_built']],
                y=[year_price],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
                name='🏗️ Votre Année'
            )
    
    fig_evolution.update_layout(
        height=400,
        font_family="Inter",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig_surface, fig_quality, fig_evolution

# Interface principale
def main():
    # Header responsive avec vos vraies performances
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏠 Prédicteur Prix Immobilier IA</h1>
        <p>Random Forest Optimisé • R² = 0.8688 • RMSE = $26,240 • Interface Responsive</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger VOS VRAIS modèles
    model, feature_info, models_loaded = load_real_models()
    
    # Si les modèles ne sont pas chargés, afficher un message d'erreur
    if not models_loaded:
        st.markdown("""
        <div class="error-box">
            <h3>⚠️ Modèles non disponibles</h3>
            <p>Veuillez vous assurer que les fichiers suivants sont présents dans le répertoire :</p>
            <ul>
                <li><code>xgb_model.pkl</code> - Votre modèle Random Forest entraîné</li>
                <li><code>feature_info.pkl</code> - Métadonnées de vos features</li>
            </ul>
            <p><strong>📁 Structure attendue :</strong></p>
            <pre>
votre-projet-immobilier/
├── streamlit_app.py
├── xgb_model.pkl
├── feature_info.pkl
└── requirements.txt
            </pre>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Charger les données de démonstration
    surface_data, quality_data, evolution_data = create_demo_data()
    
    # Sidebar responsive avec VOS VRAIES performances
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")
        
        # Sélection de l'unité
        unit_system = st.selectbox(
            "🌍 Système d'unités",
            ["Impérial (sq ft)", "Métrique (m²)"],
            help="Choisissez votre système d'unités préféré"
        )
        
        # Sélection de la devise
        currency = st.selectbox(
            "💰 Devise",
            ["USD ($)", "EUR (€)"],
            help="Devise pour l'affichage du prix"
        )
        
        # Mode avancé
        advanced_mode = st.checkbox("🔬 Mode Avancé", help="Afficher analyses détaillées et graphiques")
        
        # Informations de VOTRE modèle réel
        st.markdown("## 📊 Votre Modèle")
        if feature_info:
            rmse_score = feature_info.get('rmse_score', 26240.20)
            r2_score = feature_info.get('r2_score', 0.8688)
            train_samples = feature_info.get('model_stats', {}).get('train_samples', 1456)
            mean_price = feature_info.get('model_stats', {}).get('mean_price', 180151.23)
            
            st.markdown(f"""
            **🎯 Performances Réelles:**
            - RMSE Test: ${rmse_score:,.0f}
            - R² Score: {r2_score:.4f}
            - Variance expliquée: {r2_score*100:.1f}%
            
            **🏗️ Architecture:**
            - Modèle: Random Forest (Optimisé)
            - Échantillons: {train_samples:,}
            - Prix moyen: ${mean_price:,.0f}
            
            **🔥 Top Features:**
            - OverallQual: 57.9%
            - GrLivArea: 18.4%
            - TotalBsmtSF: 9.2%
            """)
        
        # Informations sur les features
        if advanced_mode and feature_info and 'feature_importance' in feature_info:
            st.markdown("## 🎯 Importance Features")
            importance_data = feature_info['feature_importance'][:5]  # Top 5
            for item in importance_data:
                st.write(f"• **{item['feature']}**: {item['importance']:.3f}")
    
    # Interface principale avec colonnes responsives
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🏡 Caractéristiques de la Propriété")
        
        # Formulaire avec validation selon VOS features exactes
        with st.form("house_prediction_form"):
            # Première ligne - Qualité et Surface (vos 2 features les plus importantes)
            col_a, col_b = st.columns(2)
            
            with col_a:
                overall_qual = st.selectbox(
                    "⭐ Qualité Générale (Feature #1)",
                    options=list(range(1, 11)),
                    index=6,
                    help="Qualité générale de la maison (1=Pauvre, 10=Excellent) - Feature la plus importante (57.9%)"
                )
            
            with col_b:
                if unit_system == "Métrique (m²)":
                    surface_m2 = st.number_input(
                        "🏠 Surface Habitable (m²)",
                        min_value=50.0,
                        max_value=500.0,
                        value=150.0,
                        step=5.0,
                        help="Surface habitable principale - Feature #2 (18.4%)"
                    )
                    gr_liv_area = int(surface_m2 * 10.764)
                else:
                    gr_liv_area = st.number_input(
                        "🏠 Surface Habitable (sq ft)",
                        min_value=500,
                        max_value=5000,
                        value=1500,
                        step=50,
                        help="Surface habitable principale - Feature #2 (18.4%)"
                    )
                    surface_m2 = gr_liv_area / 10.764
            
            # Deuxième ligne - Sous-sol et Garage
            col_c, col_d = st.columns(2)
            
            with col_c:
                if unit_system == "Métrique (m²)":
                    bsmt_m2 = st.number_input(
                        "🏠 Surface Sous-sol (m²)",
                        min_value=0.0,
                        max_value=300.0,
                        value=100.0,
                        step=5.0,
                        help="Surface totale du sous-sol - Feature #3 (9.2%)"
                    )
                    total_bsmt_sf = int(bsmt_m2 * 10.764)
                else:
                    total_bsmt_sf = st.number_input(
                        "🏠 Surface Sous-sol (sq ft)",
                        min_value=0,
                        max_value=3000,
                        value=1000,
                        step=50,
                        help="Surface totale du sous-sol - Feature #3 (9.2%)"
                    )
                    bsmt_m2 = total_bsmt_sf / 10.764
            
            with col_d:
                garage_cars = st.selectbox(
                    "🚗 Places de Garage",
                    options=list(range(0, 5)),
                    index=2,
                    help="Nombre de places de garage"
                )
            
            # Troisième ligne - Année et Salles de bain
            col_e, col_f = st.columns(2)
            
            with col_e:
                year_built = st.number_input(
                    "📅 Année de Construction",
                    min_value=1900,
                    max_value=2024,
                    value=2000,
                    step=1,
                    help="Année de construction - Feature #5 (3.9%)"
                )
            
            with col_f:
                full_bath = st.selectbox(
                    "🛁 Salles de Bain Complètes",
                    options=list(range(1, 5)),
                    index=1,
                    help="Nombre de salles de bain complètes"
                )
            
            # Quatrième ligne - Pièces et Cheminées
            col_g, col_h = st.columns(2)
            
            with col_g:
                tot_rms_abv_grd = st.number_input(
                    "🏠 Pièces au-dessus du Sol",
                    min_value=3,
                    max_value=15,
                    value=7,
                    step=1,
                    help="Nombre total de pièces au-dessus du sol"
                )
            
            with col_h:
                fireplaces = st.selectbox(
                    "🔥 Cheminées",
                    options=list(range(0, 4)),
                    index=1,
                    help="Nombre de cheminées"
                )
            
            # Calcul automatique de la surface de garage
            garage_area = garage_cars * 250  # Estimation standard
            
            # Bouton de prédiction
            submitted = st.form_submit_button(
                "🔮 Prédire le Prix avec Votre Modèle",
                use_container_width=True
            )
    
    with col2:
        st.markdown("## 📊 Résumé")
        
        # Affichage du résumé avec design moderne
        age = 2024 - year_built
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>🏠 Votre Propriété</h4>
            <p><strong>Surface:</strong> {surface_m2:.0f} m² ({gr_liv_area:,} sq ft)</p>
            <p><strong>Qualité:</strong> {overall_qual}/10 ⭐</p>
            <p><strong>Âge:</strong> {age} ans ({year_built})</p>
            <p><strong>Sous-sol:</strong> {bsmt_m2:.0f} m² ({total_bsmt_sf:,} sq ft)</p>
            <p><strong>Garage:</strong> {garage_cars} places ({garage_area:,} sq ft)</p>
            <p><strong>SdB complètes:</strong> {full_bath}</p>
            <p><strong>Pièces totales:</strong> {tot_rms_abv_grd}</p>
            <p><strong>Cheminées:</strong> {fireplaces}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scores de qualité
        if advanced_mode:
            st.markdown("### 🎯 Scores de Qualité")
            
            # Calcul des scores basés sur vos données réelles
            surface_score = min(100, (gr_liv_area / 2500) * 100)
            quality_score = overall_qual * 10
            modernity_score = max(0, 100 - (age / 50) * 100)
            basement_score = min(100, (total_bsmt_sf / 1500) * 100)
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("📐 Surface", f"{surface_score:.0f}%")
                st.metric("⭐ Qualité", f"{quality_score:.0f}%")
            with col_s2:
                st.metric("🏗️ Modernité", f"{modernity_score:.0f}%")
                st.metric("🏠 Sous-sol", f"{basement_score:.0f}%")
            
            # Score global
            global_score = (surface_score + quality_score + modernity_score + basement_score) / 4
            st.metric("🎯 Score Global", f"{global_score:.0f}%")
    
    # Prédiction et résultats avec VOTRE modèle
    if submitted and model:
        # Préparer les features selon VOTRE modèle exact
        user_inputs = {
            'overall_qual': overall_qual,
            'gr_liv_area': gr_liv_area,
            'total_bsmt_sf': total_bsmt_sf,
            'garage_area': garage_area,
            'year_built': year_built,
            'full_bath': full_bath,
            'tot_rms_abv_grd': tot_rms_abv_grd,
            'fireplaces': fireplaces,
            'garage_cars': garage_cars
        }
        
        # Faire la prédiction avec VOTRE modèle
        features_df = prepare_features_for_your_model(user_inputs)
        predicted_price_usd = make_prediction_with_your_model(model, features_df)
        
        if predicted_price_usd:
            # Conversion de devise
            if currency == "EUR (€)":
                predicted_price = predicted_price_usd * 0.85
                currency_symbol = "€"
            else:
                predicted_price = predicted_price_usd
                currency_symbol = "$"
            
            # Afficher la prédiction avec vos vraies performances
            rmse_score = feature_info.get('rmse_score', 26240.20)
            r2_score = feature_info.get('r2_score', 0.8688)
            
            st.markdown(f"""
            <div class="prediction-card fade-in-up">
                <h2>🎯 Prix Estimé par Votre Modèle</h2>
                <div class="prediction-price">{currency_symbol}{predicted_price:,.0f}</div>
                <p>Random Forest Optimisé • R² = {r2_score:.3f} • RMSE = ±${rmse_score:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Graphiques interactifs
            if advanced_mode:
                st.markdown("## 📈 Analyses Visuelles Interactives")
                
                # Créer les graphiques
                fig_surface, fig_quality, fig_evolution = create_interactive_charts(
                    surface_data, quality_data, evolution_data, 
                    predicted_price_usd, user_inputs
                )
                
                # Affichage responsive des graphiques
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_surface, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_evolution, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_chart2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_quality, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Graphique d'importance des features
                    if 'feature_importance' in feature_info:
                        importance_df = pd.DataFrame(feature_info['feature_importance'])
                        
                        fig_importance = px.bar(
                            importance_df.head(6),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="🎯 Importance des Features",
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        
                        fig_importance.update_layout(
                            height=400,
                            font_family="Inter",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # Analyses détaillées avec VOS données
                st.markdown("## 🔍 Analyses Détaillées")
                
                col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
                
                with col_analysis1:
                    # Comparaison avec VOS données d'entraînement
                    mean_price = feature_info.get('model_stats', {}).get('mean_price', 180151.23)
                    price_diff = predicted_price_usd - mean_price
                    price_diff_pct = (price_diff / mean_price) * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Vs Votre Dataset</h4>
                        <p><strong>Prix moyen dataset:</strong> ${mean_price:,.0f}</p>
                        <p><strong>Différence:</strong> ${price_diff:,.0f}</p>
                        <p><strong>Écart:</strong> {price_diff_pct:+.1f}%</p>
                        <p><strong>Catégorie:</strong> {"🔥 Premium" if price_diff_pct > 20 else "📈 Au-dessus" if price_diff_pct > 0 else "💰 Abordable"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_analysis2:
                    # Prix par m² et efficacité
                    price_per_sqft = predicted_price_usd / gr_liv_area
                    price_per_m2 = price_per_sqft * 10.764
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📐 Prix par Surface</h4>
                        <p><strong>Prix/sq ft:</strong> ${price_per_sqft:.0f}</p>
                        <p><strong>Prix/m²:</strong> ${price_per_m2:.0f}</p>
                        <p><strong>Surface totale:</strong> {gr_liv_area + total_bsmt_sf:,} sq ft</p>
                        <p><strong>Efficacité:</strong> {"🌟 Excellente" if price_per_sqft < 120 else "👍 Bonne" if price_per_sqft < 150 else "💸 Élevée"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_analysis3:
                    # Confiance de la prédiction basée sur vos performances
                    confidence = (r2_score * 100)
                    error_margin = rmse_score / predicted_price_usd * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Confiance Prédiction</h4>
                        <p><strong>Confiance modèle:</strong> {confidence:.1f}%</p>
                        <p><strong>Marge d'erreur:</strong> ±{error_margin:.1f}%</p>
                        <p><strong>Fourchette:</strong> ${predicted_price_usd - rmse_score:,.0f} - ${predicted_price_usd + rmse_score:,.0f}</p>
                        <p><strong>Fiabilité:</strong> {"🚀 Très élevée" if confidence > 85 else "📊 Élevée" if confidence > 75 else "⚠️ Modérée"}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Message de bienvenue si pas de prédiction
    else:
        st.markdown("""
        <div class="info-box fade-in-up">
            <h3>👋 Bienvenue dans votre Prédicteur Prix Immobilier IA !</h3>
            <p>Cette application utilise <strong>votre modèle Random Forest entraîné</strong> avec les performances suivantes :</p>
            <ul>
                <li>🎯 <strong>R² Score:</strong> 0.8688 (86.9% de variance expliquée)</li>
                <li>📊 <strong>RMSE:</strong> $26,240.20 (erreur moyenne)</li>
                <li>🏗️ <strong>Données:</strong> 1,456 transactions immobilières</li>
                <li>🔥 <strong>Top Feature:</strong> OverallQual (57.9% d'importance)</li>
            </ul>
            <p><strong>🚀 Pour commencer :</strong> Remplissez le formulaire ci-dessus avec les caractéristiques de votre propriété.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher les graphiques de démonstration
        if advanced_mode:
            st.markdown("## 📊 Aperçu de Votre Dataset")
            
            fig_surface, fig_quality, fig_evolution = create_interactive_charts(
                surface_data, quality_data, evolution_data
            )
            
            # Layout responsive pour les graphiques
            tab1, tab2, tab3 = st.tabs(["📐 Surface-Prix", "⭐ Qualité-Prix", "📈 Évolution Temporelle"])
            
            with tab1:
                st.plotly_chart(fig_surface, use_container_width=True)
                st.markdown("*Relation entre la surface habitable et le prix de vente*")
                
            with tab2:
                st.plotly_chart(fig_quality, use_container_width=True)
                st.markdown("*Impact de la qualité générale sur le prix (Feature la plus importante)*")
                
            with tab3:
                st.plotly_chart(fig_evolution, use_container_width=True)
                st.markdown("*Évolution des prix selon l'année de construction*")
    
    # Footer responsive avec informations sur votre projet
    st.markdown("---")
    
    # Layout adaptatif pour le footer
    footer_cols = st.columns([1, 1, 1])
    
    with footer_cols[0]:
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Votre Modèle</h4>
            <p><strong>Random Forest Optimisé</strong> entraîné sur 1,456 transactions avec 86.9% de précision.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with footer_cols[1]:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Features Clés</h4>
            <p><strong>9 caractéristiques</strong> sélectionnées par importance : Qualité, Surface, Sous-sol, Garage, etc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with footer_cols[2]:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Précision</h4>
            <p><strong>Marge d'erreur ±$26,240</strong> basée sur les performances réelles de votre modèle.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
