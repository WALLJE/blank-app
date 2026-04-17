import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# --- Layout & Titel ---
st.set_page_config(page_title="SBERT Evaluation Dashboard", layout="wide")
st.title("🔬 SBERT NLP-Evaluations-Pipeline")
st.markdown("Analysieren Sie die Konsistenz und Variabilität von KI-Antworten pro Fall.")

# --- KI-Modell laden (gecached) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model = load_model()

# --- Datei-Upload ---
uploaded_file = st.file_uploader("Excel-Datei hochladen (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Check ob die nötigen Spalten da sind
    required_cols = ['Fall', 'Modell', 'Text']
    if not all(col in df.columns for col in required_cols):
        st.error(f"FEHLER: Die Tabelle muss die Spalten {required_cols} enthalten.")
        st.info(f"Gefundene Spalten: {list(df.columns)}")
    else:
        # --- Sidebar für Filter ---
        st.sidebar.header("Filter-Einstellungen")
        alle_faelle = ["Alle Fälle"] + list(df['Fall'].unique())
        ausgewaehlter_fall = st.sidebar.selectbox("Welchen Fall möchten Sie analysieren?", alle_faelle)

        # Daten filtern
        if ausgewaehlter_fall == "Alle Fälle":
            df_final = df
        else:
            df_final = df[df['Fall'] == ausgewaehlter_fall]

        st.success(f"Daten geladen: {len(df_final)} Antworten für '{ausgewaehlter_fall}' ausgewählt.")
        st.dataframe(df_final[['Fall', 'Modell', 'Text']].head(10))

        if st.button("Wissenschaftliche Analyse starten"):
            with st.spinner('Berechne SBERT-Vektoren...'):
                # Embeddings berechnen
                embeddings = model.encode(df_final['Text'].tolist())
                df_final['Vektor'] = list(embeddings)
                
                modelle = df_final['Modell'].unique()
                centroids = {}
                
                st.divider()
                st.subheader(f"📊 Analyse-Ergebnisse für: {ausgewaehlter_fall}")
                
                # Metriken in Kacheln anzeigen
                cols = st.columns(len(modelle))
                
                for idx, m in enumerate(modelle):
                    subset = df_final[df_final['Modell'] == m]
                    subset_embs = np.array(subset['Vektor'].tolist())
                    
                    if len(subset_embs) < 2:
                        cols[idx].warning(f"Modell '{m}': Zu wenig Daten (n={len(subset_embs)}).")
                        continue
                        
                    # 1. Konsistenz (Cosine Sim)
                    sim_matrix = cosine_similarity(subset_embs)
                    mean_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
                    
                    # 2. Variabilität (Euklidische Distanz zum Centroid)
                    centroid = np.mean(subset_embs, axis=0)
                    centroids[m] = centroid
                    mean_variance = np.mean([np.linalg.norm(vec - centroid) for vec in subset_embs])
                    
                    with cols[idx]:
                        st.markdown(f"### Modell: {m}")
                        st.write(f"Stichprobe: **n = {len(subset_embs)}**")
                        st.metric("Konsistenz (Ähnlichkeit)", f"{mean_sim:.4f}", 
                                  help="Näher an 1.0 bedeutet: Die KI sagt fast immer das Gleiche.")
                        st.metric("Variabilität (Streuung)", f"{mean_variance:.4f}", 
                                  help="Niedrigerer Wert bedeutet: Die KI ist stabiler/weniger 'halluzinativ'.")

                # --- Vergleichs-Metriken ---
                if len(centroids) >= 2:
                    st.divider()
                    st.subheader("📐 Direkter Vergleich der Systeme")
                    
                    m_list = list(centroids.keys())
                    # Wir vergleichen exemplarisch die ersten beiden Modelle
                    c1, c2 = centroids[m_list[0]], centroids[m_list[1]]
                    shift_sim = cosine_similarity([c1], [c2])[0][0]
                    shift_dist = 1 - shift_sim
                    
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.metric(f"Inhaltlicher Shift ({m_list[0]} vs. {m_list[1]})", f"{shift_dist:.4f}")
                        st.info("Dieser Wert misst, wie stark sich der Kern der Aussagen durch das System-Update (z.B. AMBOSS) verschoben hat.")
                    
                    with c_right:
                        if len(df_final) > 3:
                            sil_score = silhouette_score(embeddings, df_final['Modell'])
                            st.metric("Trennschärfe (Silhouette Score)", f"{sil_score:.4f}")
                            st.info("Ein positiver Wert zeigt an, dass die KI-Systeme klar unterscheidbare Antwort-Profile haben.")

st.sidebar.markdown("---")
st.sidebar.info("Tipp: Nutzen Sie pro Fall mindestens 5-10 Durchläufe für statistisch belastbare Werte.")
