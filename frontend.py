import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Détecteur de faux billets", page_icon="💵")

st.title("DetectorBills - Détection des faux billets avec IA")

uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant les données", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données envoyées :")
        st.dataframe(df.head())

        # Vérification du nombre de colonnes
        if len(df.columns) != 6:
            st.error(f"Le fichier CSV doit contenir exactement 6 colonnes. Ce fichier en contient {len(df.columns)}.")
        else:
            if st.button("Envoyer au modèle pour prédiction"):
                with st.spinner("Traitement en cours..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

                        response = requests.post("http://localhost:8000/predict", files=files)

                        if response.status_code != 200:
                            st.error(f"Erreur API : {response.status_code} - {response.text}")
                            st.stop()

                        data = response.json()

                        # Tableau prédictions
                        st.subheader("Table de prédictions")
                        df_pred = pd.DataFrame(data["table_predictions"])
                        st.dataframe(df_pred)

                        # Statistiques globales
                        st.subheader("Statistiques globales")
                        stats = data["statistiques"]
                        st.markdown(f"- Total d'entrées : {stats['total']}")
                        st.markdown(f"- Nombre de vrais billets : {stats['vrais']} ({stats['pourcentage_vrais']}%)")
                        st.markdown(f"- Nombre de faux billets : {stats['faux']} ({stats['pourcentage_faux']}%)")

                        # Graphique camembert
                        fig = go.Figure(
                            data=[go.Pie(
                                labels=["Vrais billets", "Faux billets"],
                                values=[stats["vrais"], stats["faux"]],
                                hole=0.4,
                                marker=dict(colors=["green", "red"])
                            )]
                        )
                        fig.update_layout(title_text="Répartition des vrais / faux billets")

                        st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Erreur lors de la requête : {e}")

    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")
        st.stop()
