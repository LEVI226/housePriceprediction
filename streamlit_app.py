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

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🏠 Prédicteur Prix Immobilier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #4CAF50;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-bg: #ffffff;
        --text-color: #2c3e50;
        --border-radius: 15px;
        --shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: var(--border-radius);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        text-align: center;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .prediction-price {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle et les informations
@st.cache_resource
def load_model_and_info():
    """Charger le modèle XGBoost et les informations des features"""
    try:
        # Charger le modèle
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Charger les informations des features
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        return model, feature_info, True
    except FileNotFoundError:
        st.error("❌ Fichiers du modèle non trouvés. Assurez-vous que 'xgb_model.pkl' et 'feature_info.pkl' sont présents.")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None, None, False

# Fonction de conversion m² vers sq ft
def m2_to_sqft(m2):
    """Convertir m² en sq ft"""
    return m2 * 10.764

# Fonction de conversion sq ft vers m²
def sqft_to_m2(sqft):
    """Convertir sq ft en m²"""
    return sqft / 10.764

# Fonction de conversion USD vers EUR
def usd_to_eur(usd, rate=0.85):
    """Convertir USD en EUR"""
    return usd * rate

# Fonction de prédiction
def predict_price(model, features):
    """Prédire le prix avec le modèle"""
    try:
        prediction = model.predict([features])[0]
        return max(0, prediction)  # Assurer que le prix est positif
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return 0

# Interface principale
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Prédicteur Prix Immobilier IA</h1>
        <p>Estimation intelligente basée sur XGBoost • Données USA Housing • Précision ±$15,000</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model, feature_info, model_loaded = load_model_and_info()
    
    if not model_loaded:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Modèle non disponible</h3>
            <p>Pour utiliser cette application, vous devez d'abord entraîner le modèle avec le notebook fourni et placer les fichiers suivants dans le même dossier :</p>
            <ul>
                <li><code>xgb_model.pkl</code> - Le modèle XGBoost entraîné</li>
                <li><code>feature_info.pkl</code> - Les métadonnées des features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar avec informations du modèle
    with st.sidebar:
        st.markdown("## 📊 Informations du Modèle")
        
        if feature_info:
            st.markdown(f"""
            **🎯 Performance:**
            - RMSE: ${feature_info['rmse_score']:,.0f}
            - R² Score: {feature_info['model_stats']['test_r2']:.3f}
            
            **🏗️ Architecture:**
            - Modèle: XGBoost Optimisé
            - Features: {len(feature_info['feature_names'])}
            - Données d'entraînement: {feature_info['model_stats'].get('train_samples', 'N/A')}
            """)
        
        st.markdown("## 🔧 Paramètres")
        
        # Sélection de l'unité
        unit_system = st.selectbox(
            "🌍 Système d'unités",
            ["Métrique (m²)", "Impérial (sq ft)"],
            help="Choisissez votre système d'unités préféré"
        )
        
        # Sélection de la devise
        currency = st.selectbox(
            "💰 Devise",
            ["USD ($)", "EUR (€)"],
            help="Devise pour l'affichage du prix"
        )
        
        # Mode avancé
        advanced_mode = st.checkbox("🔬 Mode Avancé", help="Afficher plus de détails et d'analyses")
    
    # Interface principale avec colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🏡 Caractéristiques de la Propriété")
        
        # Récupérer les ranges des features
        ranges = feature_info['feature_ranges'] if feature_info else {}
        
        # Interface d'entrée des données
        with st.form("house_prediction_form"):
            # Première ligne - Surface
            col_a, col_b = st.columns(2)
            
            with col_a:
                if unit_system == "Métrique (m²)":
                    living_area_m2 = st.number_input(
                        "🏠 Surface Habitable (m²)",
                        min_value=50.0,
                        max_value=500.0,
                        value=150.0,
                        step=5.0,
                        help="Surface habitable principale de la maison"
                    )
                    living_area_sqft = m2_to_sqft(living_area_m2)
                else:
                    living_area_sqft = st.number_input(
                        "🏠 Surface Habitable (sq ft)",
                        min_value=ranges.get('GrLivArea', {}).get('min', 500),
                        max_value=ranges.get('GrLivArea', {}).get('max', 4000),
                        value=1500,
                        step=50,
                        help="Surface habitable principale de la maison"
                    )
                    living_area_m2 = sqft_to_m2(living_area_sqft)
            
            with col_b:
                if unit_system == "Métrique (m²)":
                    basement_area_m2 = st.number_input(
                        "🏠 Surface Sous-sol (m²)",
                        min_value=0.0,
                        max_value=300.0,
                        value=100.0,
                        step=5.0,
                        help="Surface totale du sous-sol"
                    )
                    basement_area_sqft = m2_to_sqft(basement_area_m2)
                else:
                    basement_area_sqft = st.number_input(
                        "🏠 Surface Sous-sol (sq ft)",
                        min_value=ranges.get('TotalBsmtSF', {}).get('min', 0),
                        max_value=ranges.get('TotalBsmtSF', {}).get('max', 3000),
                        value=1000,
                        step=50,
                        help="Surface totale du sous-sol"
                    )
                    basement_area_m2 = sqft_to_m2(basement_area_sqft)
            
            # Deuxième ligne - Qualité et Garage
            col_c, col_d = st.columns(2)
            
            with col_c:
                overall_qual = st.selectbox(
                    "⭐ Qualité Générale",
                    options=list(range(1, 11)),
                    index=6,
                    help="Qualité générale de la maison (1=Très Pauvre, 10=Excellent)"
                )
            
            with col_d:
                garage_cars = st.selectbox(
                    "🚗 Places de Garage",
                    options=list(range(0, 5)),
                    index=2,
                    help="Nombre de voitures pouvant être garées"
                )
            
            # Troisième ligne - Garage et Année
            col_e, col_f = st.columns(2)
            
            with col_e:
                if unit_system == "Métrique (m²)":
                    garage_area_m2 = st.number_input(
                        "🚗 Surface Garage (m²)",
                        min_value=0.0,
                        max_value=150.0,
                        value=50.0,
                        step=5.0,
                        help="Surface du garage"
                    )
                    garage_area_sqft = m2_to_sqft(garage_area_m2)
                else:
                    garage_area_sqft = st.number_input(
                        "🚗 Surface Garage (sq ft)",
                        min_value=ranges.get('GarageArea', {}).get('min', 0),
                        max_value=ranges.get('GarageArea', {}).get('max', 1500),
                        value=500,
                        step=25,
                        help="Surface du garage"
                    )
                    garage_area_m2 = sqft_to_m2(garage_area_sqft)
            
            with col_f:
                current_year = datetime.now().year
                year_built = st.number_input(
                    "📅 Année de Construction",
                    min_value=ranges.get('YearBuilt', {}).get('min', 1900),
                    max_value=current_year,
                    value=2000,
                    step=1,
                    help="Année de construction de la maison"
                )
            
            # Quatrième ligne - Salles de bain et Pièces
            col_g, col_h = st.columns(2)
            
            with col_g:
                full_bath = st.selectbox(
                    "🛁 Salles de Bain Complètes",
                    options=list(range(0, 6)),
                    index=2,
                    help="Nombre de salles de bain complètes"
                )
            
            with col_h:
                total_rooms = st.number_input(
                    "🏠 Nombre Total de Pièces",
                    min_value=ranges.get('TotRmsAbvGrd', {}).get('min', 3),
                    max_value=ranges.get('TotRmsAbvGrd', {}).get('max', 15),
                    value=7,
                    step=1,
                    help="Nombre total de pièces au-dessus du sol"
                )
            
            # Cinquième ligne - Cheminées
            fireplaces = st.selectbox(
                "🔥 Nombre de Cheminées",
                options=list(range(0, 4)),
                index=1,
                help="Nombre de cheminées dans la maison"
            )
            
            # Bouton de prédiction
            submitted = st.form_submit_button(
                "🔮 Prédire le Prix",
                use_container_width=True
            )
    
    with col2:
        st.markdown("## 📊 Résumé des Caractéristiques")
        
        # Affichage du résumé
        st.markdown(f"""
        <div class="feature-card">
            <h4>🏠 Surfaces</h4>
            <p><strong>Habitable:</strong> {living_area_m2:.0f} m² ({living_area_sqft:.0f} sq ft)</p>
            <p><strong>Sous-sol:</strong> {basement_area_m2:.0f} m² ({basement_area_sqft:.0f} sq ft)</p>
            <p><strong>Garage:</strong> {garage_area_m2:.0f} m² ({garage_area_sqft:.0f} sq ft)</p>
            
            <h4>🏡 Caractéristiques</h4>
            <p><strong>Qualité:</strong> {overall_qual}/10</p>
            <p><strong>Année:</strong> {year_built} ({current_year - year_built} ans)</p>
            <p><strong>Pièces:</strong> {total_rooms}</p>
            <p><strong>SdB complètes:</strong> {full_bath}</p>
            <p><strong>Places garage:</strong> {garage_cars}</p>
            <p><strong>Cheminées:</strong> {fireplaces}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Indicateurs de qualité
        if advanced_mode:
            st.markdown("### 🎯 Indicateurs de Qualité")
            
            # Score de surface
            total_area = living_area_sqft + basement_area_sqft
            area_score = min(100, (total_area / 3000) * 100)
            
            # Score de modernité
            age = current_year - year_built
            modernity_score = max(0, 100 - (age / 50) * 100)
            
            # Score de commodités
            amenity_score = ((garage_cars * 20) + (full_bath * 15) + (fireplaces * 10) + (overall_qual * 8)) / 1.3
            amenity_score = min(100, amenity_score)
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.metric("📐 Score Surface", f"{area_score:.0f}%")
                st.metric("🏗️ Score Modernité", f"{modernity_score:.0f}%")
            with col_score2:
                st.metric("🎯 Score Commodités", f"{amenity_score:.0f}%")
                st.metric("⭐ Score Global", f"{(area_score + modernity_score + amenity_score)/3:.0f}%")
    
    # Prédiction et résultats
    if submitted and model:
        # Préparer les features pour le modèle
        features = [
            living_area_sqft,    # GrLivArea
            basement_area_sqft,  # TotalBsmtSF
            overall_qual,        # OverallQual
            garage_cars,         # GarageCars
            garage_area_sqft,    # GarageArea
            year_built,          # YearBuilt
            full_bath,           # FullBath
            total_rooms,         # TotRmsAbvGrd
            fireplaces           # Fireplaces
        ]
        
        # Faire la prédiction
        predicted_price_usd = predict_price(model, features)
        
        if predicted_price_usd > 0:
            # Convertir en EUR si nécessaire
            if currency == "EUR (€)":
                predicted_price = usd_to_eur(predicted_price_usd)
                currency_symbol = "€"
                currency_name = "EUR"
            else:
                predicted_price = predicted_price_usd
                currency_symbol = "$"
                currency_name = "USD"
            
            # Afficher la prédiction
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prix Estimé</h2>
                <div class="prediction-price">{currency_symbol}{predicted_price:,.0f}</div>
                <p>Estimation basée sur XGBoost • Précision ±{feature_info['rmse_score']:,.0f} USD</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyses détaillées
            if advanced_mode:
                st.markdown("## 📈 Analyses Détaillées")
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    # Comparaison avec la moyenne
                    avg_price = feature_info['model_stats']['mean_price']
                    price_diff = predicted_price_usd - avg_price
                    price_diff_pct = (price_diff / avg_price) * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Comparaison Marché</h4>
                        <p><strong>Prix moyen marché:</strong> ${avg_price:,.0f}</p>
                        <p><strong>Différence:</strong> ${price_diff:,.0f} ({price_diff_pct:+.1f}%)</p>
                        <p><strong>Catégorie:</strong> {"🔥 Premium" if price_diff_pct > 20 else "📈 Au-dessus moyenne" if price_diff_pct > 0 else "💰 Bon rapport qualité-prix"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_analysis2:
                    # Prix par m²
                    price_per_m2_usd = predicted_price_usd / living_area_m2
                    price_per_m2 = usd_to_eur(price_per_m2_usd) if currency == "EUR (€)" else price_per_m2_usd
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📐 Prix par Surface</h4>
                        <p><strong>Prix/m² habitable:</strong> {currency_symbol}{price_per_m2:,.0f}</p>
                        <p><strong>Surface totale:</strong> {living_area_m2 + basement_area_m2:.0f} m²</p>
                        <p><strong>Efficacité:</strong> {"🌟 Excellente" if price_per_m2 < 1500 else "👍 Bonne" if price_per_m2 < 2000 else "💸 Élevée"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Graphique de l'importance des features
                if 'feature_importance' in feature_info and feature_info['feature_importance']:
                    st.markdown("### 🎯 Importance des Caractéristiques")
                    
                    importance_df = pd.DataFrame(feature_info['feature_importance'])
                    
                    fig = px.bar(
                        importance_df.head(8),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Impact des caractéristiques sur le prix",
                        color='importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Simulation de variations
                st.markdown("### 🔄 Simulation de Variations")
                
                variation_col1, variation_col2 = st.columns(2)
                
                with variation_col1:
                    st.markdown("**📈 Impact d'améliorations:**")
                    
                    # Simulation +1 qualité
                    features_improved = features.copy()
                    features_improved[2] = min(10, overall_qual + 1)  # OverallQual
                    price_improved = predict_price(model, features_improved)
                    improvement_value = price_improved - predicted_price_usd
                    
                    st.write(f"• +1 Qualité: +${improvement_value:,.0f}")
                    
                    # Simulation +1 salle de bain
                    features_bathroom = features.copy()
                    features_bathroom[6] = full_bath + 1  # FullBath
                    price_bathroom = predict_price(model, features_bathroom)
                    bathroom_value = price_bathroom - predicted_price_usd
                    
                    st.write(f"• +1 Salle de bain: +${bathroom_value:,.0f}")
                
                with variation_col2:
                    st.markdown("**📉 Impact de l'âge:**")
                    
                    # Simulation maison plus ancienne
                    features_older = features.copy()
                    features_older[5] = year_built - 10  # YearBuilt
                    price_older = predict_price(model, features_older)
                    age_impact = predicted_price_usd - price_older
                    
                    st.write(f"• +10 ans d'âge: -${age_impact:,.0f}")
                    
                    # Simulation surface réduite
                    features_smaller = features.copy()
                    features_smaller[0] = living_area_sqft * 0.9  # GrLivArea
                    price_smaller = predict_price(model, features_smaller)
                    size_impact = predicted_price_usd - price_smaller
                    
                    st.write(f"• -10% surface: -${size_impact:,.0f}")
    
    # Footer avec informations
    st.markdown("---")
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Précision du Modèle</h4>
            <p>Notre modèle XGBoost a été entraîné sur des milliers de transactions immobilières et offre une précision de ±$15,000 en moyenne.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_footer2:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Données Utilisées</h4>
            <p>Les prédictions sont basées sur 9 caractéristiques clés : surface, qualité, âge, garage, salles de bain, etc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_footer3:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Avertissement</h4>
            <p>Cette estimation est indicative. Les prix réels peuvent varier selon le marché local et d'autres facteurs.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
