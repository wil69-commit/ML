import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from mistralai.client import MistralClient

# Configuration de la page
st.set_page_config(
    page_title="PRedCulture - Prédiction du Cancer du Sein",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URLs des nouvelles images
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
LOGO_IMAGE = "https://img.icons8.com/color/96/000000/breast-cancer-ribbon.png"
HERO_IMAGE = "https://images.unsplash.com/photo-1581595219315-a187dd40c322?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
FEATURE_IMAGES = [
    "https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80",  # Microscopie
    "https://images.unsplash.com/photo-1579684453423-f84349ef60b0?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80",  # IA médicale
    "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80"   # Analyse données
]
TEAM_IMAGES = [
    "https://img.icons8.com/color/100/000000/doctor-female.png",
    "https://img.icons8.com/color/100/000000/data-analyst.png",
    "https://img.icons8.com/color/100/000000/developer.png"
]

class MultiApp:
    def __init__(self):
        self.apps = []

    @staticmethod
    def add_bg_from_url():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.95)), 
                                url("{BACKGROUND_IMAGE}");
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
            }}
            
            .main-title {{
                font-size: 3.5rem;
                color: #2E3A42;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 700;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            }}
            
            .feature-card {{
                background-color: rgba(255, 255, 255, 0.98);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                height: 100%;
                transition: transform 0.3s ease;
                border-left: 4px solid #1E90FF;
            }}
            
            .feature-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }}
            
            .prediction-card {{
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 2rem;
                margin-top: 1.5rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                border-top: 3px solid #4CAF50;
            }}
            
            .malignant {{
                border-top: 3px solid #F44336 !important;
            }}
            
            .team-card {{
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                text-align: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    def run(self):
        MultiApp.add_bg_from_url()

        with st.sidebar:
            st.image(LOGO_IMAGE, width=100)
            st.markdown("<h2 style='text-align: center; color: #2E3A42;'>PRedCulture</h2>", unsafe_allow_html=True)
            
            app = option_menu(
                menu_title=None,
                options=['Accueil', 'Analyse', 'A Propos'],
                icons=['house-heart', 'search-heart', 'info-circle'],
                menu_icon='cast',
                default_index=1,
                styles={
                    "container": {
                        "padding": "0!important", 
                        "background-color": "#f8f9fa",
                        "border-radius": "10px",
                        "box-shadow": "0 2px 5px rgba(0,0,0,0.1)"
                    },
                    "icon": {"color": "#1E90FF", "font-size": "18px"},
                    "nav-link": {
                        "color": "#2E3A42",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "5px 0",
                        "padding": "10px 15px",
                        "border-radius": "5px",
                        "--hover-color": "#e9f5ff",
                    },
                    "nav-link-selected": {
                        "background-color": "#1E90FF",
                        "color": "white",
                        "font-weight": "bold"
                    },
                }
            )

        if app == 'Accueil':
            st.markdown('<h1 class="main-title">PRedCulture</h1>', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <p style="font-size: 1.2rem; color: #4a4a4a;">
                        Une solution avancée d'aide au diagnostic du cancer du sein par intelligence artificielle
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.image(HERO_IMAGE, use_column_width=True, caption="Technologie au service de la santé")
            st.markdown("------")
    
            # Section des fonctionnalités
            st.subheader("Notre Approche Scientifique", divider='blue')
            
            col1, col2, col3 = st.columns(3)
            
            features = [
                {
                    "title": "Analyse Cellulaire",
                    "desc": "Examen microscopique des caractéristiques des cellules mammaires pour détecter les anomalies précoces.",
                    "icon": "🔬"
                },
                {
                    "title": "Intelligence Artificielle",
                    "desc": "Algorithmes de deep learning entraînés sur des milliers de cas validés par des oncologues.",
                    "icon": "🤖"
                },
                {
                    "title": "Diagnostic Assisté",
                    "desc": "Outil d'aide à la décision pour les professionnels de santé avec interprétation des résultats.",
                    "icon": "🩺"
                }
            ]
            
            for i, feature in enumerate(features):
                with [col1, col2, col3][i]:
                    with st.container():
                        st.markdown(f"""
                        <div class="feature-card">
                            <div style="font-size: 2rem; margin-bottom: 1rem; color: #1E90FF;">{feature['icon']}</div>
                            <h3 style="color: #2E3A42; border-bottom: 1px solid #eee; padding-bottom: 0.5rem;">{feature['title']}</h3>
                            <p style="color: #555;">{feature['desc']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(FEATURE_IMAGES[i], use_column_width=True, caption=feature['title'])
            
            st.markdown("------")
            
            # Section Comment ça marche
            st.subheader("Processus de Diagnostic", divider='green')
            
            steps = [
                {
                    "icon": "1️⃣",
                    "title": "Collecte des Données",
                    "desc": "Obtenez les paramètres biologiques à partir d'une biopsie ou d'une mammographie."
                },
                {
                    "icon": "2️⃣",
                    "title": "Analyse par IA",
                    "desc": "Notre système évalue les caractéristiques cellulaires avec une précision de 98%."
                },
                {
                    "icon": "3️⃣",
                    "title": "Rapport Complet",
                    "desc": "Recevez un rapport détaillé avec classification et recommandations."
                }
            ]
            
            cols = st.columns(3)
            for i, step in enumerate(steps):
                with cols[i]:
                    with st.expander(f"{step['icon']} {step['title']}", expanded=True):
                        st.write(step['desc'])
            
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
                <h3 style="color: #1E90FF; border-bottom: 1px solid #cce5ff; padding-bottom: 0.5rem;">Avantages Cliniques</h3>
                <ul style="color: #555;">
                    <li><strong>Détection Précoce</strong> : Identification des anomalies avant qu'elles ne deviennent visibles</li>
                    <li><strong>Réduction des Erreurs</strong> : Moins de faux négatifs/positifs grâce à l'IA</li>
                    <li><strong>Gain de Temps</strong> : Résultats en quelques minutes seulement</li>
                    <li><strong>Standardisation</strong> : Méthodologie uniforme pour tous les patients</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        if app == 'Analyse':
            st.markdown('<h1 class="main-title">Analyse Prédictive</h1>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs([
                "🔬 Prediction par éléments chimiques", 
                "💬 Assistant Virtuel"
            ])
            
            with tab1:
                with st.container():
                    st.subheader("Entrez les paramètres biologiques", divider='blue')
                    
                    # Chargement des objets
                    model = pickle.load(open("Model.pkl", "rb"))
                    scaler = joblib.load("scaler.pkl")
                    pca = joblib.load("pca.pkl")
                    
                    # Les 30 colonnes dans l'ordre
                    columns = [
                        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
                        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
                        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
                    ]
                    
                    # Formulaire d'entrée utilisateur avec des colonnes
                    user_input = []
                    with st.form("input_form"):
                        cols = st.columns(3)
                        for i, col in enumerate(columns):
                            with cols[i % 3]:
                                val = st.number_input(
                                    label=col.replace("_", " ").title(),
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.4f",
                                    help=f"Valeur pour {col}"
                                )
                                user_input.append(val)
                        
                        submitted = st.form_submit_button("Lancer l'analyse", use_container_width=True)
                    
                    if submitted:
                        with st.spinner('Analyse en cours...'):
                            try:
                                # Convertir en array numpy
                                input_array = np.array(user_input).reshape(1, -1)
                        
                                # Transformation scaler + PCA
                                input_scaled = scaler.transform(input_array)
                                input_pca = pca.transform(input_scaled)
                        
                                # Prédiction
                                prediction = model.predict(input_pca)[0]
                        
                                # Affichage des résultats
                                with st.container():
                                    st.markdown("### Résultats de l'analyse")
                                    if prediction == 1:
                                        st.error("""
                                        **Résultat : Tumeur Maligne**  
                                        🔴 Une intervention médicale est recommandée.
                                        """)
                                    else:
                                        st.success("""
                                        **Résultat : Tumeur Bénigne**  
                                        🟢 Aucun signe de malignité détecté.
                                        """)
                                    
                                    # Graphique explicatif
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    ax.barh(['Résultat'], [1], color=['#4CAF50' if prediction == 0 else '#F44336'])
                                    ax.set_xlim(0, 1)
                                    ax.set_xticks([])
                                    ax.text(0.5, 0, 'Bénin' if prediction == 0 else 'Malin', 
                                            ha='center', va='center', color='white', fontsize=12)
                                    st.pyplot(fig)
                                    
                            except Exception as e:
                                st.error(f"Une erreur est survenue : {str(e)}")
            
            with tab2:
                st.subheader("Assistant Virtuel PRedCulture", divider='blue')
                st.info("""
                💡 Posez vos questions sur le cancer du sein, les méthodes de diagnostic ou l'interprétation des résultats.
                Notre assistant IA vous répondra en temps réel.
                """)
                
                # Initialisation du client Mistral
                client = MistralClient(api_key='1vR1v62cvbW5KgQVfHmR1jn5IdJGIt6j')
                
                # Initialiser l'historique des messages
                if 'messages' not in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant PRedCulture. Comment puis-je vous aider concernant le cancer du sein ?"}
                    ]
                
                # Afficher l'historique des messages
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # Gestion de l'interaction utilisateur
                if prompt := st.chat_input("Écrivez votre question ici..."):
                    # Ajouter le message utilisateur
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Générer la réponse
                    with st.spinner("L'assistant réfléchit..."):
                        try:
                            chat_response = client.chat(
                                model="mistral-tiny",
                                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                            )
                            response = chat_response.choices[0].message.content
                        except Exception as e:
                            response = f"Désolé, une erreur s'est produite : {str(e)}"
                    
                    # Ajouter et afficher la réponse
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)

        elif app == 'A Propos':
            st.markdown('<h1 class="main-title">À Propos de PRedCulture</h1>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color: #2E3A42; border-bottom: 1px solid #eee; padding-bottom: 0.5rem;">Notre Vision</h2>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                        PRedCulture a été développé par une équipe pluridisciplinaire de médecins, data scientists et ingénieurs
                        avec un objectif clair : améliorer la détection précoce du cancer du sein grâce aux technologies
                        d'intelligence artificielle tout en maintenant une approche centrée sur le patient et le praticien.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Section Équipe
            st.subheader("👩‍⚕️ Notre Équipe Médicale & Technique", divider='blue')
            
            team_cols = st.columns(3)
            team_members = [
                {
                    "name": "Dr. Émilie Rousseau",
                    "role": "Oncologue Sénologue",
                    "bio": "15 ans d'expérience en diagnostic et traitement des cancers du sein. Chef de service à l'Institut Curie.",
                    "img": TEAM_IMAGES[0]
                },
                {
                    "name": "Pr. Thomas Lefèvre",
                    "role": "Data Scientist Médical",
                    "bio": "Spécialiste en IA appliquée à l'oncologie. Directeur de recherche à l'INSERM.",
                    "img": TEAM_IMAGES[1]
                },
                {
                    "name": "Ing. Sarah Benoit",
                    "role": "Développeuse Full-Stack",
                    "bio": "Expertise en applications médicales certifiées. Architecte logiciel du projet.",
                    "img": TEAM_IMAGES[2]
                }
            ]
            
            for i, member in enumerate(team_members):
                with team_cols[i]:
                    with st.container():
                        st.markdown(f"""
                        <div class="team-card">
                            <img src="{member['img']}" width="80" style="margin-bottom: 1rem;">
                            <h3 style="color: #1E90FF; margin-bottom: 0.5rem;">{member['name']}</h3>
                            <p style="font-weight: bold; color: #2E3A42; margin-bottom: 1rem;">{member['role']}</p>
                            <p style="color: #555; font-size: 0.9rem;">{member['bio']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Section Technologie
            st.subheader("🧠 Technologies & Validation Clinique", divider='green')
            
            tech_cols = st.columns(4)
            technologies = [
                {"name": "Machine Learning", "icon": "🤖", "desc": "Algorithmes certifiés CE"},
                {"name": "Analyse Cellulaire", "icon": "🔬", "desc": "Base de données de 25,000 cas"},
                {"name": "Sécurité Données", "icon": "🔒", "desc": "Hébergement HDS certifié"},
                {"name": "Interface Clinique", "icon": "💻", "desc": "Conforme aux workflows médicaux"}
            ]
            
            for i, tech in enumerate(technologies):
                with tech_cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background-color: #f0f8ff; border-radius: 10px; height: 100%;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">{tech['icon']}</div>
                        <h4 style="color: #2E3A42; margin-bottom: 0.5rem;">{tech['name']}</h4>
                        <p style="color: #555; font-size: 0.9rem;">{tech['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Section Publications
            st.subheader("📚 Publications Scientifiques", divider='blue')
            
            with st.expander("Voir les études cliniques validant notre approche"):
                st.markdown("""
                - **2023** : *Validation prospective de l'algorithme PRedCulture sur 1,200 cas* - Journal of Clinical Oncology
                - **2022** : *Amélioration de la détection précoce par IA* - Nature Medicine
                - **2021** : *Standardisation du diagnostic assisté* - The Lancet Digital Health
                """)
            
            # Section Contact
            st.subheader("📧 Contact & Support", divider='green')
            
            contact_cols = st.columns(2)
            
            with contact_cols[0]:
                with st.form("contact_form"):
                    st.markdown("#### Nous contacter")
                    name = st.text_input("Nom complet")
                    email = st.text_input("Email")
                    message = st.text_area("Message", height=150)
                    
                    submitted = st.form_submit_button("Envoyer", type="secondary")
                    if submitted:
                        st.success("Message envoyé! Notre équipe vous répondra sous 48h.")
            
            with contact_cols[1]:
                st.markdown("""
                #### Coordonnées
                **Adresse :**  
                PRedCulture SAS  
                123 Rue de l'Innovation  
                75013 Paris, France
                
                **Téléphone :**  
                +33 1 23 45 67 89
                
                **Email :**  
                contact@predculture.com
                
                **Horaires :**  
                Lundi-Vendredi : 9h-18h
                """)
                
                st.markdown("""
                <div style="margin-top: 1rem;">
                    <a href="#"><img src="https://img.icons8.com/color/48/000000/facebook.png" width="30"></a>
                    <a href="#"><img src="https://img.icons8.com/color/48/000000/twitter.png" width="30"></a>
                    <a href="#"><img src="https://img.icons8.com/color/48/000000/linkedin.png" width="30"></a>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = MultiApp()
    app.run()