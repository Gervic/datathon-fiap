import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Decision AI Assistant ", page_icon="🤖", layout="centered")
st.title("Bem vindo ao Decision AI, nosso assistente de recrutamento")

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

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
  
# --- 3. Initialize Session State Variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_action" not in st.session_state:
    st.session_state.selected_action = None

if "uploaded_cvs_data" not in st.session_state:
    st.session_state.uploaded_cvs_data = {} # Stores {filename: text}

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = [] # Stores list of dicts: {name, score, analysis}

if "job_list" not in st.session_state:
    st.session_state.job_list = [] # Stores the parsed job data from vagas.json

if "selected_job_title" not in st.session_state:
    st.session_state.selected_job_title = "Selecionar uma vaga" # Default option for selectbox
  
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)


# --- 4. Helper Functions for Text Extraction ---
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

# --- 5. Helper for Cosine Similarity ---
def calculate_cosine_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    documents = [text1, text2]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2]).flatten()[0]
    return cosine_sim

# --- 6. Helper for Gemini Chat History Formatting ---
def to_gemini_history(streamlit_messages):
    gemini_format = []
    for msg in streamlit_messages:
        role = 'user' if msg["role"] == "user" else 'model'
        gemini_format.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    return gemini_format

# --- 7. Load Job Descriptions from JSON ---
# This function will be called once per session or on app reload
@st.cache_data # Cache the data to avoid re-reading the file on every rerun
def load_job_descriptions(json_path="vagas.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        return jobs
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{json_path}' não encontrado. Por favor, verifique o caminho.")
        return []
    except json.JSONDecodeError:
        st.error(f"Erro: Não foi possível ler '{json_path}'. Verifique o formato do JSON.")
        return []

# Load jobs when the app starts
if not st.session_state.job_list:
    st.session_state.job_list = load_job_descriptions()


# --- 8. Initial Welcome Message and Action Choice ---
if not st.session_state.messages and st.session_state.selected_action is None:
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Bem-vindo ao Analisador de Recrutamento. O que gostaria de fazer?"})
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[0]["content"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Analisar CV(s)", use_container_width=True):
            st.session_state.selected_action = 'analyze_cv'
            st.session_state.messages.append({"role": "user", "content": "Quero analisar CV(s)."})
            st.session_state.messages.append({"role": "assistant", "content": "Ok! Por favor, faça o upload de até 5 CVs na barra lateral e **selecione a vaga desejada**."})
            st.experimental_rerun()
    with col2:
        if st.button("❓ Tirar uma dúvida", use_container_width=True):
            st.session_state.selected_action = 'ask_question'
            st.session_state.messages.append({"role": "user", "content": "Quero tirar uma dúvida."})
            st.session_state.messages.append({"role": "assistant", "content": "Certo! Pergunte o que quiser."})
            st.experimental_rerun()

# --- 9. Display Chat History ---
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# --- 10. Conditional UI for CV Analysis (Multiple Files + Job Selection) ---
if st.session_state.selected_action == 'analyze_cv':
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
                with st.sidebar.spinner(f"Lendo CV: {uploaded_file.name}..."):
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

    # Get job titles for selectbox
    job_titles = ["Selecionar uma vaga"] + [job["titulo"] for job in st.session_state.job_list]

    # Selectbox for job titles
    selected_job_title = st.selectbox(
        "Selecione uma vaga da lista:",
        options=job_titles,
        key="job_title_selector",
        index=job_titles.index(st.session_state.selected_job_title) if st.session_state.selected_job_title in job_titles else 0
    )

    # Update session state with selected title
    st.session_state.selected_job_title = selected_job_title

    # Update job description based on selection
    if selected_job_title != "Selecionar uma vaga":
        for job in st.session_state.job_list:
            if job["titulo"] == selected_job_title:
                st.session_state.job_description = job["descricao"]
                break
    else:
        # If "Selecionar uma vaga" is chosen, allow manual input or clear if previously selected
        if st.session_state.job_description and st.session_state.selected_job_title == "Selecionar uma vaga":
            pass # Keep current job_description if user typed it
        else:
            st.session_state.job_description = "" # Clear if it was from a previous selection

    # Text area for manual input or displaying selected job description
    st.text_area(
        "Descrição da Vaga (auto-preenchido ou cole aqui)",
        key="job_description_input",
        value=st.session_state.job_description,
        height=200,
        help="A descrição da vaga será auto-preenchida ao selecionar uma vaga. Você também pode colar uma descrição aqui."
    )
    # Update job_description session state if user manually edits the text_area
    st.session_state.job_description = st.session_state.job_description_input


    if st.session_state.uploaded_cvs_data and st.session_state.job_description:
        if st.button("✨ Analisar Correspondência dos CVs", type="primary"):
            st.session_state.analysis_results = []
            with st.spinner("Calculando correspondências e gerando análises..."):
                for cv_name, cv_text in st.session_state.uploaded_cvs_data.items():
                    if not cv_text:
                        st.session_state.analysis_results.append({
                            "name": cv_name,
                            "score": 0.0,
                            "analysis": "Não foi possível extrair texto deste CV."
                        })
                        continue

                    similarity_score = calculate_cosine_similarity(cv_text, st.session_state.job_description) * 100

                    analysis_prompt = (
                        f"Você é um analista de recrutamento. A pontuação de similaridade entre o CV de '{cv_name}' "
                        f"e a Descrição da Vaga é de {similarity_score:.2f}%. "
                        f"\n\n--- Texto do CV: {cv_name} ---\n{cv_text}\n\n"
                        f"--- Descrição da Vaga ---\n{st.session_state.job_description}\n\n"
                        f"Com base nesta pontuação e nos textos fornecidos, "
                        f"analise os pontos fortes e fracos do candidato em relação à vaga. "
                        f"Seja objetivo e profissional, e sugira áreas onde o CV poderia ser melhorado para a vaga. "
                        f"Formate a resposta com os seguintes tópicos: 'Pontos Fortes', 'Pontos a Melhorar', 'Recomendação Geral'. "
                        f"Mantenha a análise concisa, no máximo 3 parágrafos."
                    )
                    try:
                        response = model.generate_content(analysis_prompt)
                        ai_analysis = response.text
                    except Exception as e:
                        ai_analysis = f"Ocorreu um erro ao gerar a análise para '{cv_name}': {e}"

                    st.session_state.analysis_results.append({
                        "name": cv_name,
                        "score": similarity_score,
                        "analysis": ai_analysis
                    })

                st.session_state.analysis_results.sort(key=lambda x: x["score"], reverse=True)

                ranking_message = "### Resultados da Análise de CVs (Ranking)\n\n"
                for i, result in enumerate(st.session_state.analysis_results):
                    ranking_message += f"**{i+1}. {result['name']}** (Score: {result['score']:.2f}%)\n"
                    ranking_message += f"**Análise:**\n{result['analysis']}\n\n"
                    ranking_message += "---\n\n"

                st.session_state.messages.append({"role": "assistant", "content": ranking_message})
                st.experimental_rerun()

# --- 11. General Chat Input (for "Tirar uma dúvida" or follow-ups) ---
elif st.session_state.selected_action == 'ask_question':
    prompt = st.chat_input("Pergunte o que quiser...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        gemini_chat_history = to_gemini_history(st.session_state.messages[:-1])

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    chat_session = model.start_chat(history=gemini_chat_history)
                    response = chat_session.send_message(prompt)
                    ai_response = response.text
                    st.markdown(ai_response)
                except Exception as e:
                    ai_response = f"Ocorreu um erro: {e}"
                    st.error(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- 12. Clear Chat / Reset Options ---
if st.button("🏠 Início / Limpar Conversa"):
    st.session_state.messages = []
    st.session_state.selected_action = None
    st.session_state.uploaded_cvs_data = {}
    st.session_state.job_description = ""
    st.session_state.analysis_results = []
    st.session_state.job_list = [] # Reset job list so it's reloaded
    st.session_state.selected_job_title = "Selecionar uma vaga"
    st.experimental_rerun()
