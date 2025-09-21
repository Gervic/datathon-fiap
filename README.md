🤖 Datathon-FIAP – AI Recruitment App
-----

This project was developed as the final challenge of the FIAP Tech Postgraduate Program.
The goal is to build an AI-powered recruitment app, leveraging embeddings from resumes and job descriptions to provide insights about candidate suitability.


📂 Project Structure
------


├ decisionAI.py                 # App Script 

├ decisionAI.env                # Env variables

├ decisionAI_image.png          # App Image

├ resume_prediction_model.py    # Prediction model script

├ resume_prediction_model.ipynb # Prediction model notebook

├ resume_matching_model_2.pkl.gz# Model pickle file

├ job_texts_embeddings.json     # Embeddings of job descriptions

├ job_ideal_embeddings.json     # Embeddings of hired per job_id

├ vagas.json                    # Job descriptions

├ requirements.txt              # Dependencies 

└── README.md                     


⚙️ How It Works
-----


Embeddings:

job_texts_embeddings.json: embeddings of job descriptions.

job_ideal_embeddings.json: embeddings of ideal candidates per job_id.

Similarity Model:

A pre-trained ML model (resume_matching_model_2.pkl.gz) computes probabilities based on cosine similarity scores between:

Resume ↔ Job description

Resume ↔ Ideal candidates

Recruitment App:

The main script decisionAI.py runs the pipeline.

A Gemini model consumes these results and produces a textual analysis of the submitted resume.


🚀 Running the Project
------


Clone the repository

```bash
git clone https://github.com/seu-repo/datathon-fiap.git
cd datathon-fiap
```


Create a virtual environment (opcional)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```


Install the dependencies

```bash
pip install -r requirements.txt
```


Configure the variables
In the file decisionAI.env, insert your API key (ex.: Gemini).

Run the app

```bash
python decisionAI.py
``` 



📊 Expected Results
-------


Resume ↔ Job description similarity

Resume ↔ Ideal candidate similarity

AI-based analysis (Gemini) with recruitment insights
