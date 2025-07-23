import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import gzip
import io
import json 
import numpy as np

def set_dark_mode():
    st.markdown("""
        <style>
            .stApp {
                background-color: #000000;
                color: #ffffff;
            }
            section[data-testid="stSidebar"] {
                background-color: #111;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Decision AI Assistant ", page_icon='3d-ai-assistant-icon.avif', layout="wide")
st.title("Bem vindo ao Decision AI, nosso assistente de recrutamento")


st.sidebar.title('DecisionAI Hub')
st.sidebar.image("AI motion.gif", use_container_width=True)

with st.sidebar.expander("Sobre"):
        st.markdown("""
        **DecisionAI** é uma inteligência criada para auxiliar os recruitadores na missão de encontrar os talentos mais aderentes às vagas de forma ágil.
        
        Combina o poder do machine learning e das LLMs para criar um relatório completo sobre os candidatos.
        
        📧 [Contato para Suporte](mailto:suporte@decision-hr-analytics.com)
        """)

particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

# Securely Load API Key 
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if gemini_api_key is None:
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key is None:
    st.error("Gemini API key not found. Please set it in Streamlit Cloud secrets or in a local .env file.")
    st.info("Get your key from: https://aistudio.google.com/app/apikey")
    st.stop()

genai.configure(api_key=gemini_api_key)
model_gemini = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17") 

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_action" not in st.session_state:
    st.session_state.selected_action = None #'analyze_cv'

if "uploaded_cvs_data" not in st.session_state:
    st.session_state.uploaded_cvs_data = {} # Stores {filename: text}

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = [] # Stores list of dicts: {name, score, analysis, prediction_proba, recommended}

if "job_list" not in st.session_state:
    st.session_state.job_list = []

if "selected_job_title" not in st.session_state:
    st.session_state.selected_job_title = "Selecionar uma vaga"

if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)

# Loading ML Models and Data (Cached) 
@st.cache_resource # Caching the model loading, runs only once
def load_ml_models():
    try:
        with gzip.open('resume_matching_model_2.pkl.gz', 'rb') as f:
          dat = joblib.load(f)
      
        # Load StandardScaler
        scaler = dat['scaler'] 
        
        # Load RandomForestClassifier model
        clf_model = dat['model'] 
      
        # Load SentenceTransformer model
        sbert_model = dat['embedder'] 
        
        # Load ideal employee embeddings
        with open('job_ideal_embeddings.json', 'r') as f:
            job_ideal_embeds_json = json.load(f)
        
        # Convert list back to numpy arrays
        job_ideal_embeds = {
            job_id: np.array(embedding)
            for job_id, embedding in job_ideal_embeds_json.items()
        }
        
        return sbert_model, scaler, clf_model, job_ideal_embeds
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar arquivos de modelo: {e}. Certifique-se de que 'scaler.pkl', 'random_forest_model.pkl' e 'job_ideal_embeddings.json' estão no diretório raiz do seu projeto.")
        st.stop()
    except Exception as e:
        st.error(f"Erro inesperado ao carregar modelos: {e}")
        st.stop()

# Load all models and data upfront
sbert_model, scaler, clf_model, job_ideal_embeds = load_ml_models()


#Functions for Text Extraction 
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    document = Document(file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    text = ""
    try:
        if file_extension == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error(f"Arquivo '{uploaded_file.name}' possui tipo não suportado. Por favor, faça upload de arquivos PDF ou DOCX.")
            return None
    except Exception as e:
        st.error(f"Erro ao ler o arquivo '{uploaded_file.name}': {e}")
        return None
    return text

# Function for Cosine Similarity (using SBERT embeddings)
# This function takes embeddings, not raw text
def calculate_cosine_similarity_embeddings(embed1, embed2):
    # Ensuring embeddings are 2D arrays for sklearn's cosine_similarity
    if embed1.ndim == 1:
        embed1 = embed1.reshape(1, -1)
    if embed2.ndim == 1:
        embed2 = embed2.reshape(1, -1)
    
    if embed1.shape[1] != embed2.shape[1]:
        st.error("Dimensões dos embeddings não correspondem.")
        return 0.0
    
    return cosine_similarity(embed1, embed2)[0][0]

# Function for Gemini Chat History Formatting
def to_gemini_history(streamlit_messages):
    gemini_format = []
    for msg in streamlit_messages:
        role = 'user' if msg["role"] == "user" else 'model'
        gemini_format.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    return gemini_format

#Loading Job Descriptions from JSON 
@st.cache_data
def load_job_descriptions(json_path="vagas.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        seen = set()
        vagas_list = []
        for job_id, v in jobs.items():
            info = v.get('informacoes_basicas', {})
            profile = v.get('perfil_vaga', {})
            job_text = ' '.join([
                info.get('titulo_vaga', '') or '',
                info.get('objetivo_vaga', '') or '',
                profile.get('nivel profissional'),
                profile.get('areas_atuacao') or '',
                profile.get('principais_atividades') or '',
                profile.get('competencia_tecnicas_e_comportamentais') or '',
                profile.get('habilidades_comportamentais_necessarias') or '',
                profile.get('demais_observacoes') or ''
        
            ])
            if job_id not in seen:
              seen.add(job_id)
              vagas_list.append({'job_id': job_id,
                                  'titulo':f"{job_id} - {info.get('titulo_vaga', '')}",
                                 'descricao': job_text
                                })
        return vagas_list
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{json_path}' não encontrado. Por favor, verifique o caminho.")
        return []
    except json.JSONDecodeError:
        st.error(f"Erro: Não foi possível ler '{json_path}'. Verifique o formato do JSON.")
        return []

if not st.session_state.job_list:
    st.session_state.job_list = load_job_descriptions()


# Initial Welcome Message and Action Choice
if not st.session_state.messages and st.session_state.selected_action is None:
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Bem-vindo ao DecisionAI, o hub inteligente de recrutamento. O que gostaria de fazer?"})
    with st.chat_message("assistant", avatar= '3d-ai-assistant-icon.avif'):
        st.markdown(st.session_state.messages[0]["content"])

    options = [
                "Analisar CVs",
                "Conversar / Tirar dúvidas com a IA"
            ]

    choice = st.radio("Por favor, escolha uma opção:", options=options, index=None)
    
    if choice == "Analisar CVs":
        st.session_state.selected_action = 'analyze_cv'
        st.session_state.messages.append({"role": "user", "content": "Quero analisar CV(s)."})
        st.session_state.messages.append({"role": "assistant", "content": "Ok! Por favor, faça o upload de até 5 CVs na barra lateral e **selecione a vaga desejada**."})
    if choice == "Conversar / Tirar dúvidas com a IA":
        st.session_state.selected_action = 'ask_question'
        st.session_state.messages.append({"role": "user", "content": "Quero tirar uma dúvida."})
        st.session_state.messages.append({"role": "assistant", "content": "Certo! Pergunte o que quiser."})

#Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])

#Conditional UI for CV Analysis (Multiple Files + Job Selection)
if st.session_state.selected_action == 'analyze_cv':
    with st.spinner("Esperando o upload dos CVs"):
        st.sidebar.header("Upload de CVs")
        uploaded_files = st.sidebar.file_uploader(
            "Escolha arquivos PDF ou DOCX (até 5 CVs)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="cv_uploader"
        )
    
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.sidebar.warning("Por favor, selecione no máximo 5 CVs.")
                uploaded_files = uploaded_files[:5]
    
            current_uploaded_names = {f.name for f in uploaded_files}
            if current_uploaded_names != set(st.session_state.uploaded_cvs_data.keys()):
                st.session_state.uploaded_cvs_data = {}
                st.session_state.analysis_results = []
    
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Lendo CV: {uploaded_file.name}..."):
                        extracted_text = get_text_from_file(uploaded_file)
                        if extracted_text:
                            st.session_state.uploaded_cvs_data[uploaded_file.name] = extracted_text
                            st.sidebar.success(f"CV '{uploaded_file.name}' lido com sucesso!")
                        else:
                            st.sidebar.error(f"Não foi possível ler o CV: {uploaded_file.name}")
    
        if st.session_state.uploaded_cvs_data:
            st.sidebar.subheader("CVs Carregados:")
            for name in st.session_state.uploaded_cvs_data.keys():
                st.sidebar.write(f"- {name}")
        else:
            st.sidebar.info("Nenhum CV carregado ainda.")

    st.subheader("Escolha a Vaga ou Cole a Descrição")

    job_titles = ["Selecionar uma vaga"] + [job["titulo"] for job in st.session_state.job_list]

    selected_job_title = st.selectbox(
        "Selecione uma vaga da lista:",
        options=job_titles,
        key="job_title_selector",
        index=job_titles.index(st.session_state.selected_job_title) if st.session_state.selected_job_title in job_titles else 0
    )

    st.session_state.selected_job_title = selected_job_title

    job_id_selected = None # Initialize job_id for the selected job description
    if selected_job_title != "Selecionar uma vaga":
        for job in st.session_state.job_list:
            if job["titulo"] == selected_job_title:
                st.session_state.job_description = job["descricao"]
                job_id_selected = job.get("job_id") # Assuming 'job_id' key exists in vagas.json
                if job_id_selected is None:
                    st.warning("Job ID não encontrado para a vaga selecionada. A similaridade com o perfil ideal não será calculada.")
                break
    else:
        if st.session_state.job_description and st.session_state.selected_job_title == "Selecionar uma vaga":
            pass
        else:
            st.session_state.job_description = ""

    st.text_area(
        "Descrição da Vaga (auto-preenchido ou cole aqui)",
        key="job_description_input",
        value=st.session_state.job_description,
        height=200,
        help="A descrição da vaga será auto-preenchida ao selecionar uma vaga. Você também pode colar uma descrição aqui."
    )
    st.session_state.job_description = st.session_state.job_description_input


    if st.session_state.uploaded_cvs_data and st.session_state.job_description:
        if st.button("✨ Analisar Correspondência dos CVs", type="primary"):
            st.session_state.analysis_results = []
            with st.spinner("Calculando correspondências e gerando análises..."):
                # Embed job description once
                job_desc_embedding = sbert_model.encode(st.session_state.job_description)
              
                for cv_name, cv_text in st.session_state.uploaded_cvs_data.items():
                    if not cv_text:
                        st.session_state.analysis_results.append({
                            "name": cv_name,
                            "score": 0.0,
                            "analysis": "Não foi possível extrair texto deste CV.",
                            "prediction_proba": 0.0,
                            "recommended": False
                        })
                        continue

                    # Generate CV embedding
                    cv_embedding = sbert_model.encode(cv_text)

                    # Calculate cosine similarity to job description
                    cosine_to_job = calculate_cosine_similarity_embeddings(cv_embedding, job_desc_embedding) * 100

                    # Calculate cosine similarity to ideal employee profile
                    cosine_to_ideal = 0.0
                    if job_id_selected in job_ideal_embeds:
                        ideal_embedding = job_ideal_embeds[job_id_selected]
                        cosine_to_ideal = calculate_cosine_similarity_embeddings(cv_embedding, ideal_embedding) * 100
                    #elif job_id_selected:
                        #st.warning(f"Embedding do perfil ideal não encontrado para Job ID: {job_id_selected} (CV: {cv_name}). A similaridade com o perfil ideal não será calculada para este CV.")

                    # Prepare features for the ML model
                    features = np.array([[cosine_to_job, cosine_to_ideal]])
                    features_scaled = scaler.transform(features) # Transform with the loaded scaler

                    # Get prediction probability from the ML model
                    prediction_proba = clf_model.predict_proba(features_scaled)[0][1] * 100 # Probability of being hired

                    # Determine recommendation based on a threshold (e.g., > 50%)
                    recommended = prediction_proba >= 70 # You can adjust this threshold

                    # Formulate prompt for Gemini, including ML prediction
                    analysis_prompt = (
                        f"Você é um analista de recrutamento. A análise a seguir é para o CV de '{cv_name}' "
                        f"em relação à vaga com a descrição: '{st.session_state.job_description}'.\n\n"
                        f"**Métricas:**\n"
                        f"- Similaridade com a Descrição da Vaga (TF-IDF Cosine): {cosine_to_job:.2f}%\n"
                        f"- Similaridade com o Perfil de Funcionário Ideal (Embeddings Cosine): {cosine_to_ideal:.2f}%\n"
                        f"- **Probabilidade de Contratação (Modelo ML): {prediction_proba:.2f}%**\n\n"
                        f"--- Texto do CV: {cv_name} ---\n{cv_text}\n\n"
                        f"Com base nessas métricas e nos textos fornecidos, "
                        f"analise os pontos fortes e fracos do candidato em relação à vaga. "
                        f"Seja objetivo e profissional, e sugira áreas onde o CV poderia ser melhorado para a vaga. "
                        f"Inclua explicitamente a recomendação do modelo ML no final, informando se o candidato é 'Recomendado para Entrevista' ou 'Não Recomendado no momento' baseado na probabilidade de contratação.\n"
                        f"Formate a resposta com os seguintes tópicos: 'Pontos Fortes', 'Pontos a Melhorar', 'Recomendação Final do Modelo'."
                    )
                    try:
                        with st.spinner('DecisionAI está gerando os reports...'):
                            response = model_gemini.generate_content(analysis_prompt)
                            ai_analysis = response.text
                    except Exception as e:
                        ai_analysis = f"Ocorreu um erro ao gerar a análise para '{cv_name}': {e}"

                    st.session_state.analysis_results.append({
                        "name": cv_name,
                        "score": cosine_to_job, 
                        "cosine_to_job": cosine_to_job,
                        "cosine_to_ideal": cosine_to_ideal,
                        "prediction_proba": prediction_proba,
                        "recommended": recommended,
                        "analysis": ai_analysis
                    })

                # Sorting results by ML prediction probability in descending order
                st.session_state.analysis_results.sort(key=lambda x: x["prediction_proba"], reverse=True)

                ranking_message = "### Resultados da Análise de CVs (Ranking por Probabilidade de Contratação)\n\n"
                for i, result in enumerate(st.session_state.analysis_results):
                    recommendation_text = "🟢 **Recomendado para Entrevista**" if result['recommended'] else "🔴 **Não Recomendado no momento**"
                    ranking_message += (
                        f"**{i+1}. {result['name']}**\n"
                        f"- Probabilidade de Contratação (Modelo ML): {result['prediction_proba']:.2f}%\n"
                        f"- Similaridade com a Vaga (TF-IDF): {result['cosine_to_job']:.2f}%\n"
                        f"- Similaridade com o Perfil Ideal (Embeddings): {result['cosine_to_ideal']:.2f}%\n"
                        f"- Status: {recommendation_text}\n"
                        f"**Análise Detalhada:**\n{result['analysis']}\n\n"
                        f"---\n\n" 
                    )
                with st.spinner('Saindo do forno...'):
                    st.session_state.messages.append({"role": "assistant", "content": ranking_message})
                    st.markdown(st.session_state.messages[-1]["content"])
                    

#General Chat Input (for "Tirar uma dúvida" or follow-ups)
elif st.session_state.selected_action == 'ask_question':
    prompt = st.chat_input("Pergunte o que quiser...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        gemini_chat_history = to_gemini_history(st.session_state.messages[:-1])

        with st.chat_message("assistant", avatar='3d-ai-assistant-icon.avif'):
            with st.spinner("Pensando..."):
                try:
                    chat_session = model_gemini.start_chat(history=gemini_chat_history)
                    response = chat_session.send_message(prompt)
                    ai_response = response.text
                    st.markdown(ai_response)
                except Exception as e:
                    ai_response = f"Ocorreu um erro: {e}"
                    st.error(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

st.sidebar.subheader("Configurações")
theme = st.sidebar.selectbox("Tema", ["Escuro", "Claro"], index=0)
if theme == "Escuro":
    set_dark_mode()
    
# Clear Chat / Reset Options
if st.button("🏠 Início / Limpar Conversa"):
    st.session_state.messages = []
    st.session_state.selected_action = None
    st.session_state.uploaded_cvs_data = {}
    st.session_state.job_description = ""
    st.session_state.analysis_results = []
    st.session_state.job_list = []
    st.session_state.selected_job_title = "Selecionar uma vaga"
    st.rerun() 

# Add footer
st.markdown("---")
st.markdown("Copyrights @DecisionAI 2025") 
