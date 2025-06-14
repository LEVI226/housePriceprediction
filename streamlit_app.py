# streamlit_app.py - Version avec vos modèles

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

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🏠 Prédicteur Prix Immobilier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Même CSS que précédemment - je l'omets pour la brièveté]

# Fonction pour charger VOS VRAIS modèles
@st.cache_resource
def load_real_model_and_info():
    """Charger VOTRE modèle XGBoost et les informations des features"""
    try:
        # Charger VOTRE modèle
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Charger VOS informations des features
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        st.success("✅ Modèles chargés avec succès !")
        return model, feature_info, True
        
    except FileNotFoundError as e:
        st.error(f"❌ Fichiers du modèle non trouvés: {e}")
        st.info("""
        **📁 Fichiers requis :**
        - `xgb_model.pkl` (votre modèle entraîné)
        - `feature_info.pkl` (métadonnées des features)
        
        **🔧 Solution :**
        1. Exécutez votre notebook pour générer ces fichiers
        2. Placez-les dans le même dossier que streamlit_app.py
        """)
        return None, None, False
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None, None, False

# Fonction de prédiction avec VOTRE modèle
def predict_with_real_model(model, feature_info, user_inputs):
    """Prédire le prix avec VOTRE modèle XGBoost"""
    try:
        # Préparer les features selon l'ordre de VOTRE modèle
        feature_names = feature_info.get('feature_names', [
            'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars', 
            'GarageArea', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces'
        ])
        
        # Mapper les inputs utilisateur aux features du modèle
        features = []
        for feature_name in feature_names:
            if feature_name == 'GrLivArea':
                features.append(user_inputs['surface_sqft'])
            elif feature_name == 'TotalBsmtSF':
                features.append(user_inputs.get('basement_sqft', user_inputs['surface_sqft'] * 0.8))
            elif feature_name == 'OverallQual':
                features.append(user_inputs['quality'])
            elif feature_name == 'GarageCars':
                features.append(user_inputs['garage_cars'])
            elif feature_name == 'GarageArea':
                features.append(user_inputs['garage_cars'] * 250)  # Estimation
            elif feature_name == 'YearBuilt':
                features.append(user_inputs['year_built'])
            elif feature_name == 'FullBath':
                features.append(user_inputs['bathrooms'])
            elif feature_name == 'TotRmsAbvGrd':
                features.append(user_inputs.get('total_rooms', 7))
            elif feature_name == 'Fireplaces':
                features.append(user_inputs['fireplaces'])
            else:
                features.append(0)  # Valeur par défaut
        
        # Faire la prédiction avec VOTRE modèle
        prediction = model.predict([features])[0]
        
        return max(0, prediction)  # Assurer que le prix est positif
        
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return 0

# Interface principale modifiée
def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏠 Prédicteur Prix Immobilier IA</h1>
        <p>Votre Modèle XGBoost • Données Réelles • Interface Responsive</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger VOS VRAIS modèles
    model, feature_info, model_loaded = load_real_model_and_info()
    
    if not model_loaded:
        # Afficher les instructions si les modèles ne sont pas trouvés
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Modèles Non Disponibles</h3>
            <p>Pour utiliser cette application avec vos vrais modèles :</p>
            <ol>
                <li>📊 Exécutez votre notebook Jupyter</li>
                <li>💾 Générez les fichiers <code>xgb_model.pkl</code> et <code>feature_info.pkl</code></li>
                <li>📁 Placez-les dans le dossier de l'application</li>
                <li>🔄 Redémarrez l'application</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Optionnel : basculer vers le mode démonstration
        if st.button("🎮 Utiliser le Mode Démonstration"):
            st.session_state.demo_mode = True
            st.rerun()
        
        return
    
    # Sidebar avec informations de VOTRE modèle
    with st.sidebar:
        st.markdown("## 📊 Votre Modèle")
        
        if feature_info:
            st.markdown(f"""
            **🎯 Performance Réelle:**
            - RMSE: ${feature_info.get('rmse_score', 'N/A'):,.0f}
            - R² Score: {feature_info.get('model_stats', {}).get('test_r2', 'N/A'):.3f}
            - MAE: ${feature_info.get('mae_score', 'N/A'):,.0f}
            
            **🏗️ Architecture:**
            - Modèle: XGBoost Optimisé
            - Features: {len(feature_info.get('feature_names', []))}
            - Échantillons: {feature_info.get('model_stats', {}).get('train_samples', 'N/A'):,}
            
            **📈 Données d'Entraînement:**
            - Prix Min: ${feature_info.get('model_stats', {}).get('min_price', 'N/A'):,.0f}
            - Prix Max: ${feature_info.get('model_stats', {}).get('max_price', 'N/A'):,.0f}
            - Prix Moyen: ${feature_info.get('model_stats', {}).get('mean_price', 'N/A'):,.0f}
            """)
        
        # Paramètres
        st.markdown("## 🔧 Paramètres")
        unit_system = st.selectbox("🌍 Unités", ["Métrique (m²)", "Impérial (sq ft)"])
        currency = st.selectbox("💰 Devise", ["USD ($)", "EUR (€)"])
        advanced_mode = st.checkbox("🔬 Mode Avancé")
    
    # [Reste de l'interface - formulaire, etc.]
    # Interface principale avec colonnes responsives
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🏡 Caractéristiques de la Propriété")
        
        with st.form("house_prediction_form"):
            # Utiliser les ranges de VOTRE modèle si disponibles
            ranges = feature_info.get('feature_ranges', {}) if feature_info else {}
            
            # Surface
            col_a, col_b = st.columns(2)
            with col_a:
                if unit_system == "Métrique (m²)":
                    surface_m2 = st.number_input(
                        "🏠 Surface Habitable (m²)",
                        min_value=50.0,
                        max_value=500.0,
                        value=150.0,
                        step=5.0
                    )
                    surface_sqft = surface_m2 * 10.764
                else:
                    surface_sqft = st.number_input(
                        "🏠 Surface Habitable (sq ft)",
                        min_value=ranges.get('GrLivArea', {}).get('min', 500),
                        max_value=ranges.get('GrLivArea', {}).get('max', 4000),
                        value=1500,
                        step=50
                    )
                    surface_m2 = surface_sqft / 10.764
            
            with col_b:
                quality = st.selectbox(
                    "⭐ Qualité Générale",
                    options=list(range(1, 11)),
                    index=6
                )
            
            # Autres caractéristiques
            col_c, col_d = st.columns(2)
            with col_c:
                garage_cars = st.selectbox("🚗 Places Garage", list(range(0, 5)), index=2)
            with col_d:
                year_built = st.number_input(
                    "📅 Année Construction",
                    min_value=ranges.get('YearBuilt', {}).get('min', 1900),
                    max_value=2024,
                    value=2000
                )
            
            col_e, col_f = st.columns(2)
            with col_e:
                bathrooms = st.selectbox("🛁 Salles de Bain", [1, 1.5, 2, 2.5, 3, 3.5, 4], index=2)
            with col_f:
                fireplaces = st.selectbox("🔥 Cheminées", list(range(0, 4)), index=1)
            
            submitted = st.form_submit_button("🔮 Prédire le Prix", use_container_width=True)
    
    with col2:
        # Résumé et informations
        st.markdown("## 📊 Résumé")
        age = 2024 - year_built
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>🏠 Votre Propriété</h4>
            <p><strong>Surface:</strong> {surface_m2:.0f} m² ({surface_sqft:.0f} sq ft)</p>
            <p><strong>Qualité:</strong> {quality}/10 ⭐</p>
            <p><strong>Âge:</strong> {age} ans</p>
            <p><strong>Garage:</strong> {garage_cars} places</p>
            <p><strong>SdB:</strong> {bathrooms}</p>
            <p><strong>Cheminées:</strong> {fireplaces}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prédiction avec VOTRE modèle
    if submitted and model:
        # Préparer les données pour VOTRE modèle
        user_inputs = {
            'surface_sqft': surface_sqft,
            'quality': quality,
            'garage_cars': garage_cars,
            'year_built': year_built,
            'bathrooms': int(bathrooms),
            'fireplaces': fireplaces
        }
        
        # Faire la prédiction avec VOTRE modèle
        predicted_price_usd = predict_with_real_model(model, feature_info, user_inputs)
        
        if predicted_price_usd > 0:
            # Conversion de devise
            if currency == "EUR (€)":
                predicted_price = predicted_price_usd * 0.85
                currency_symbol = "€"
            else:
                predicted_price = predicted_price_usd
                currency_symbol = "$"
            
            # Afficher la prédiction
            st.markdown(f"""
            <div class="prediction-card fade-in-up">
                <h2>🎯 Prix Estimé (Votre Modèle)</h2>
                <div class="prediction-price">{currency_symbol}{predicted_price:,.0f}</div>
                <p>Prédiction basée sur votre modèle XGBoost • RMSE: ±${feature_info.get('rmse_score', 15000):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyses avec VOS données
            if advanced_mode and feature_info:
                st.markdown("## 📈 Analyses avec Vos Données")
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    # Comparaison avec VOS données d'entraînement
                    mean_price = feature_info.get('model_stats', {}).get('mean_price', 300000)
                    price_diff = predicted_price_usd - mean_price
                    price_diff_pct = (price_diff / mean_price) * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Vs Vos Données</h4>
                        <p><strong>Prix moyen dataset:</strong> ${mean_price:,.0f}</p>
                        <p><strong>Différence:</strong> ${price_diff:,.0f} ({price_diff_pct:+.1f}%)</p>
                        <p><strong>Percentile:</strong> {((predicted_price_usd / mean_price) * 50):.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_analysis2:
                    # Importance des features de VOTRE modèle
                    if 'feature_importance' in feature_info:
                        st.markdown("### 🎯 Top Features")
                        importance_df = pd.DataFrame(feature_info['feature_importance'])
                        top_features = importance_df.head(5)
                        
                        for _, row in top_features.iterrows():
                            st.write(f"• **{row['feature']}**: {row['importance']:.3f}")

if __name__ == "__main__":
    main()
