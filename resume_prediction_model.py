# %% [markdown]
# ###Libraries

# %%
#%pip install sentence-transformers scikit-learn pandas numpy
#%pip install -U spacy
!python3 -m spacy download pt_core_news_md

# %%
import json
import pandas as pd
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher, Matcher
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ###Functions

# %%
#Function to read json files
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

#Functions to clean the resume and job text
def clean_text(text):
    if pd.isnull(text): return ''
    text = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()


def preprocess_text(text):
  if not isinstance(text, str):
      return ""

  # 1. Remove e-mails e phones numbers
  text = re.sub(r'\S+@\S+', ' ', text)  # e-mail
  text = re.sub(r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}', ' ', text)  # phones numbers

  # 2. Remove expressions like "32 anos", "age: 45 anos"
  text = re.sub(r'\b\d{1,2}\s?(anos|anos de idade)?\b', ' ', text, flags=re.IGNORECASE)
  text = re.sub(r'idade\s*[:\-]?\s*\d{1,2}', ' ', text, flags=re.IGNORECASE)

  # 3. Processing with spaCy
  doc = nlp(text)

  clean_tokens = []

  for token in doc:
      # Ignore names, localizations e irrelevants words
      if token.ent_type_ in ["PER", "LOC", "GPE"]:  # people, localization, city/state/country
          continue
      if token.is_stop or token.is_punct or not token.is_alpha:
          continue
      clean_tokens.append(token.lemma_.lower())

  return " ".join(clean_tokens)

# %% [markdown]
# ###Loading the data

# %%
prospects = load_json('Datathon Decision/prospects.json')
vagas = load_json('Datathon Decision/vagas.json')
applicants = load_json('Datathon Decision/applicants.json')

# %%
#Create the applicants dataframe
applicants_list = []
for cand_id, a in applicants.items():
    basic_infos = a.get('infos_basicas', {})
    pro_infos = a.get('informacoes_profissionais', {})
    area = pro_infos.get('area_atuacao', '')
    skills = pro_infos.get('conhecimentos_tecnicos', '')
    certifications = pro_infos.get('certificacoes', '')
    resume_text = ' '.join([
            a.get('cv_pt', '') or '',
            skills or '',
            certifications or ''
        ])

    applicants_list.append({
        'candidate_id': cand_id.strip(),
        'candidate_name': basic_infos.get('nome', ''),
        'area': area,
        'skills': skills,
        'certifications': certifications,
        'resume_text': resume_text

    })
applicants_df = pd.DataFrame(applicants_list)

# %%
#Table shape: lines and columns
applicants_df.shape

# %%
applicants_df[['candidate_id']].duplicated().sum()

# %%
applicants_df[['resume_text']].duplicated().sum()

# %%
applicants_df[['resume_text']].isnull().sum()

# %%
applicants_df[['candidate_id', 'resume_text']].duplicated().sum()

# %%
applicants_df[['candidate_name', 'resume_text']].duplicated().sum()

# %% [markdown]
# We probably have the same candidate applying with different IDs because we have duplicated resume text with the same candidate name.

# %%
applicants_df.head()

# %%
applicants_df.info()

# %%
#Create the jobs dataframe
vagas_list = []
for job_id, v in vagas.items():
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
    vagas_list.append({'job_id': job_id.strip(),
                       'job_title': info.get('titulo_vaga', ''),
                       'job_text': job_text
                       })
vagas_df = pd.DataFrame(vagas_list)

# %%
vagas_df.shape

# %%
vagas_df.job_id.duplicated().sum()

# %%
vagas_df.head(20)

# %% [markdown]
# We have 14k uniques job_id in the table but looking at the job title I've noticed that there are some duplicates.

# %%
# All prospects candidates (hired or not)
pairs = []
for job_id, entry in prospects.items():
    for p in entry.get('prospects', []):
        label = 1 if p.get('situacao_candidado', '').lower().startswith('contratado') else 0
        pairs.append({
            'job_id': job_id.strip(),
            'candidate_id': p['codigo'].strip(),
            'label': label
        })
positives_df = pd.DataFrame(pairs)

# %%
positives_df['job_id'].duplicated().sum()

# %%
positives_df['candidate_id'].duplicated().sum()

# %%
#Verifying duplicated for the set candidate and job
positives_df[['job_id', 'candidate_id']].duplicated().sum()

# %%
positives_df.groupby(['candidate_id', 'label']).count()

# %%
#Counting records per label
positives_df.label.value_counts()

# %% [markdown]
# As we can see, we have more records in prospects table than in applicants table because the same candidate apply to more than 1 job. For example candidate_id 9993 has applied to 10 jobs and has been rejected in all of them. But it could be an application to the same job which has differents job_ids. Let's see if we have records that don't exist in applicants table

# %%
#Let's see the difference between the 2 tables
l = set(positives_df['candidate_id'])
m = set(applicants_df['candidate_id'])
len(m.difference(l))

# %% [markdown]
# We have 19k candidates with no records in prospects. Almost the half of applicants

# %%
len(l.difference(m))

# %% [markdown]
# We have 5942 candidates with no records in applicants

# %%
positives_df

# %%
hired_df = applicants_df.merge(positives_df, on='candidate_id', how ='left')
hired_df = hired_df.dropna(subset=['label'])
full_hired_df = hired_df.merge(vagas_df, on='job_id', how ='left')
full_hired_df = full_hired_df.dropna(subset=['job_text'])
full_hired_df.label = full_hired_df.label.astype(int)

# %%
full_hired_df[['candidate_id', 'job_id']].duplicated().sum()

# %%
full_hired_df['label'].value_counts()

# %%
full_hired_df.head()

# %%
df_completed = full_hired_df.drop_duplicates(subset=['candidate_id', 'candidate_name','resume_text', 'job_title'])

# %%
df_completed.label.value_counts()

# %% [markdown]
# ###Embedding & Features

# %%
nlp = spacy.load("pt_core_news_md")

# %%
df_completed['resume_text_clean'] = df_completed['resume_text'].apply(preprocess_text)

# %%
df_completed['job_text_clean'] = df_completed['job_text'].apply(preprocess_text)

# %%
#Cleaning the texts
df_completed['resume_text_clean'] = df_completed['resume_text'].apply(preprocess_text)
df_completed['job_text_clean'] = df_completed['job_text'].apply(preprocess_text)

# %%
#Using a SentenceTransformer model to generate embeddings of text
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') #('paraphrase-multilingual-MiniLM-L12-v2') all-MiniLM-L6-v2
resume_embeds = model.encode(df_completed['resume_text_clean'].tolist(), show_progress_bar=True, batch_size=64)
job_embeds = model.encode(df_completed['job_text_clean'].tolist(), show_progress_bar=True, batch_size=64)

#For each job, calculate "ideal profile" embedding (mean of all hired resumes)
job_ideal_embeds = {}
for job_id in df_completed['job_id'].unique():
    hired_texts = df_completed[(df_completed['job_id']==job_id) & (df_completed['label']==1)]['resume_text_clean']
    cleaned = [x for x in hired_texts]
    if len(cleaned) > 0:
        hired_embeds = model.encode(cleaned, batch_size=16)
        job_ideal_embeds[job_id] = np.mean(hired_embeds, axis=0)

# %%
job_ideal_embeds_serializable = {
    job_id: embedding.tolist()
    for job_id, embedding in job_ideal_embeds.items()
}

with open('job_ideal_embeddings.json', 'w') as f:
    json.dump(job_ideal_embeds_serializable, f)

print("job_ideal_embeddings.json criado com sucesso!")

# %%
job_ids_seen = set()
jobs_ids_texts_embeddings = {}
for i, row in df_completed.iterrows():
    try:
      job_id = row['job_id']
      if job_id not in job_ids_seen:
          job_ids_seen.add(job_id)
          jobs_ids_texts_embeddings[job_id] = job_embeds[i]
    except:
      continue

# %%
job_embeds_serializable = {
    job_id: embedding.tolist()
    for job_id, embedding in jobs_ids_texts_embeddings.items()
}

with open('job_texts_embeddings.json', 'w') as f:
    json.dump(job_embeds_serializable, f)

print("job_texts_embeddings.json criado com sucesso!")

# %%
# Computing features -> cosine similarity between resume text and job description/hired resumes
cos_job = []
cos_ideal = []
for i, row in enumerate(df_completed.itertuples(index=False)):
  job_id = row.job_id
  e1 = resume_embeds[i].reshape(1, -1)
  e2 = job_embeds[i].reshape(1, -1)
  cos_job.append(cosine_similarity(e1, e2))
  if job_id in job_ideal_embeds:
    cos_ideal.append(cosine_similarity(e1, job_ideal_embeds[job_id].reshape(1, -1)))
  else:
    cos_ideal.append(0)

df_completed['cosine_to_job'] = cos_job
df_completed['cosine_to_ideal_employee'] = cos_ideal

X = df_completed[['cosine_to_job','cosine_to_ideal_employee']].values
y = df_completed['label'].values
X = StandardScaler().fit_transform(X)

# %% [markdown]
# ###Model training and evaluation

# %%
#Train Model & Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.33)
clf = RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print("Advanced Model Evaluation:", report)
print("ROC-AUC: {:.3f}".format(roc))

# %% [markdown]
# ###Model export

# %%
# Save model & features for app/GenAI integration
#import pickle
#pickle.dump({'model': clf, 'scaler': StandardScaler().fit(X), 'embedder': model}, open('resume_fit_prediction_model.pkl', 'wb'))

# %%
import gzip
import pickle
# Save compressed
with gzip.open('resume_fit_prediction_model.pkl.gz', 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler, 'embedder': embedder}, f)


