import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


# --- FUNÇÕES PARA COBERTURA TEMÁTICA ---

def extract_top_keywords_tfidf(text, vectorizer, top_n=10):
  """
  Extrai as top N palavras-chave de um texto usando TF-IDF.
  Retorna uma string com as palavras-chave separadas por espaço.
  """
  if not text or pd.isna(text):
      return ""
  
  tfidf_matrix = vectorizer.transform([text])
  
  feature_array = tfidf_matrix.toarray()
  tfidf_scores = feature_array[0]
  feature_names = vectorizer.get_feature_names_out()
  
  df_scores = pd.DataFrame({'word': feature_names, 'tfidf': tfidf_scores})
  
  top_keywords = df_scores.sort_values(by='tfidf', ascending=False).head(top_n)['word'].tolist()
  
  return " ".join(top_keywords)


def calculate_similarity_test(essay_keywords, prompt_id, model_sbert, prompts_df_for_lookup):
  """
  Calcula a similaridade de cosseno entre as palavras-chave da redação
  e o embedding do prompt correspondente.
  Função implementada para reaproveitar as embeddings do prompt já calculadas 
  na fase análise das abordagens.
  """    
  prompt_row = prompts_df_for_lookup[prompts_df_for_lookup['id'] == prompt_id]
  prompt_embedding = prompt_row['prompt_embeddings'].iloc[0]
  # prompt_embedding = torch.tensor(json.loads(prompt_embedding_str))

  essay_embedding = model_sbert.encode(essay_keywords, convert_to_tensor=True)
  similarity = util.cos_sim(essay_embedding, prompt_embedding)[0][0].item()
  return similarity


def calculate_similarity(essay_keywords, prompt_keywords, sbert_model):
  """
  Calcula a similaridade de cosseno entre as palavras-chave da redação
  e as palavras-chave prompt correspondente.
  """
  essay_embedding = sbert_model.encode(essay_keywords, convert_to_tensor=True)
  prompt_embedding = sbert_model.encode(prompt_keywords, convert_to_tensor=True)
  similarity = util.cos_sim(essay_embedding, prompt_embedding)[0][0].item()
  return similarity


# --- FUNÇÕES PARA ADEQUAÇÃO AO GÊNERO ---

def predict_prob(essay_text, vectorizer_model, classifier_model):
  """
  Função pra prever o gênero de uma redação.
  """
  essay_vec = vectorizer_model.transform([essay_text])

  # Previsão das probabilidades para ambas as classes
  probabilities = classifier_model.predict_proba(essay_vec)[0] # Retorna [proba_classe_0, proba_classe_1]

  # Retorna a probabilidade da classe 1 (Dissertativo-Argumentativo)
  return probabilities[1]


def extract_paragraph_features(df):
  """
  Extrai as features: Quantidade e Tamanho de Parágrafo
  Função implementada para a fase de análise das abordagens.
  """
  df['num_paragraphs'] = df['essay'].apply(len)  
  df['paragraph_lengths'] = df['essay'].apply(lambda x: [len(p.split()) for p in x])
  df['avg_paragraph_length'] = df['paragraph_lengths'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
  return df



# --- FUNÇÕES PARA REPERTÓRIO ---

def extract_named_entities_spacy(essay_text, nlp_model):
  """
  Função para extrair NER (scpaCy)
  """
  doc = nlp_model(essay_text)
  return [(ent.text, ent.label_)
    for ent in doc.ents
    if ent.label_ in {
      "PER",           # pessoa
      "ORG",           # organização
      "GPE",           # países, estados e cidades
      "LOC",           # locais não GPE (montanhas, corpos de água, etc)
      "LAW",           # doc normativos
      "EVENT",         # evento histórico ou cultural
      "WORK_OF_ART"    # obras artísticas e literárias
    }]



# --- Dicionários e Listas Customizadas para Repertório Sociocultural ---
# Ex de obras, livros, filmes, documentos importantes
custom_list = [
  # Documentos e Fundamentais (mantidos pela relevância universal/nacional)
  "Constituição Federal de 1988",
  "Declaração Universal dos Direitos Humanos",
  "O Príncipe", # Nicolau Maquiavel - Obra universal de grande impacto
  "O Contrato Social", # Jean-Jacques Rousseau - Obra universal de grande impacto

  # Literatura Brasileira Clássica 
  # Romantismo
  "Iracema", # José de Alencar
  "O Guarani", # José de Alencar
  "Lucíola", # José de Alencar
  "Memórias de um Sargento de Milícias", # Manuel Antônio de Almeida
  "A Moreninha", # Joaquim Manuel de Macedo
  "Noite na Taverna", # Álvares de Azevedo
  "Lira dos Vinte Anos", # Álvares de Azevedo
  "Macário", # Álvares de Azevedo
  "Escrava Isaura", # Bernardo Guimarães

  # Realismo/Naturalismo
  "Memórias Póstumas de Brás Cubas", # Machado de Assis
  "Dom Casmurro", # Machado de Assis
  "Quincas Borba", # Machado de Assis
  "O Alienista", # Machado de Assis
  "A Cidade e as Serras", # Eça de Queirós (Literatura Portuguesa, mas muito estudada no Brasil)
  "O Cortiço", # Aluísio Azevedo
  "O Mulato", # Aluísio Azevedo
  "Casa de Pensão", # Aluísio Azevedo
  "Vidas Secas", # Graciliano Ramos
  "São Bernardo", # Graciliano Ramos
  "Angústia", # Graciliano Ramos
  "Os Sertões", # Euclides da Cunha
  "Canudos", # Euclides da Cunha (referência alternativa a Os Sertões)

  # Pré-Modernismo
  "Triste Fim de Policarpo Quaresma", # Lima Barreto
  "O Triste Fim de Policarpo Quaresma", # Lima Barreto (variação comum)
  "Clara dos Anjos", # Lima Barreto
  "Recordações do Escrivão Isaías Caminha", # Lima Barreto

  # Modernismo (Primeira Fase)
  "Macunaíma", # Mário de Andrade
  "Pauliceia Desvairada", # Mário de Andrade
  "Memórias Sentimentais de João Miramar", # Oswald de Andrade
  "Serafim Ponte Grande", # Oswald de Andrade
  "Amar, Verbo Intransitivo", # Mário de Andrade
  "O Banquete", # Mário de Andrade

  # Modernismo (Segunda Fase)
  "A Bagaceira", # José Américo de Almeida
  "Capitães da Areia", # Jorge Amado
  "Dona Flor e Seus Dois Maridos", # Jorge Amado
  "Gabriela, Cravo e Canela", # Jorge Amado
  "Terras do Sem Fim", # Jorge Amado
  "Fogo Morto", # José Lins do Rego
  "Menino de Engenho", # José Lins do Rego
  "Graciliano Ramos", # (Em referência às obras do autor)
  "Riobaldo", # (Personagem de Grande Sertão: Veredas, para capturar menções)
  "Diadorim", # (Personagem de Grande Sertão: Veredas)
  "Grande Sertão: Veredas", # Guimarães Rosa
  "Primeiras Estórias", # Guimarães Rosa
  "Sagarana", # Guimarães Rosa
  "Sentimento do Mundo", # Carlos Drummond de Andrade
  "A Rosa do Povo", # Carlos Drummond de Andrade
  "Alguma Poesia", # Carlos Drummond de Andrade
  "Claro Enigma", # Carlos Drummond de Andrade
  "Morte e Vida Severina", # João Cabral de Melo Neto
  "O Auto da Compadecida", # Ariano Suassuna
  "A Pedra do Reino", # Ariano Suassuna
  "A Hora da Estrela", # Clarice Lispector
  "Laços de Família", # Clarice Lispector
  "Felicidade Clandestina", # Clarice Lispector
  "Quarto de Despejo", # Carolina Maria de Jesus
  "Perto do Coração Selvagem", # Clarice Lispector

  # Teatro Brasileiro
  "Auto da Barca do Inferno", # Gil Vicente (português, mas canônico no Brasil)
  "Vestido de Noiva", # Nelson Rodrigues
  "Boca de Ouro", # Nelson Rodrigues
  "O Pagador de Promessas", # Dias Gomes

  # Outras obras relevantes/universais que aparecem
  "1984", # George Orwell
  "Admirável Mundo Novo", # Aldous Huxley
  "A Origem das Espécies", # Charles Darwin
  "O Capital", # Karl Marx
  "Manifesto Comunista", # Karl Marx
  "O Mal-Estar da Civilização", # Sigmund Freud
  "A Arte da Guerra", # Sun Tzu
  "Pedagogia do Oprimido", # Paulo Freire
  "O Segundo Sexo", # Simone de Beauvoir
  "Vigiar e Punir", # Michel Foucault
  "Casa-Grande & Senzala", # Gilberto Freyre
  "Dom Quixote", # Miguel de Cervantes
  "A Divina Comédia", # Dante Alighieri
  "Os Lusíadas", # Luís Vaz de Camões
  "Ensaio sobre a Cegueira", # José Saramago
  "A Metamorfísica dos Costumes", # Immanuel Kant
  "A Peste", # Albert Camus
  "Crime e Castigo", # Fiódor Dostoiévski
  "O Pequeno Príncipe", # Antoine de Saint-Exupéry
  "A Metamorfose", # Franz Kafka
  "O Processo", # Franz Kafka
  "O Nome da Rosa", # Umberto Eco
  "A Desobediência Civil", # Henry David Thoreau
  "Meditações", # Marco Aurélio
  "Ética a Nicômaco", # Aristóteles
  "Sobre a Liberdade", # John Stuart Mill
  "O Mundo de Sofia", # Jostein Gaarder
  "Utopia", # Thomas More
  "Um Defeito de Cor", # Ana Maria Gonçalves
  "Torto Arado", # Itamar Vieira Junior
  "O Sol na Cabeça", # Ana Paula Maia
  "Eu Sei Por Que o Pássaro Canta na Gaiola", # Maya Angelou
  "As Intermitências da Morte", # José Saramago
  "Sapiens: Uma Breve História da Humanidade", # Yuval Noah Harari
  "A Revolução dos Bichos", # George Orwell
  "Cem Anos de Solidão", # Gabriel García Márquez
  "O Velho e o Mar", # Ernest Hemingway
  "Ensaio sobre a Lucidez", # José Saramago
  "O Canto da Sereia", # Lygia Fagundes Telles
  "Ciranda de Pedra", # Lygia Fagundes Telles
  "Faca de Dois Gumes", # Luís Fernando Veríssimo
  "Verissimo", # (Em referência às obras do autor)


  # Ex de conceitos, teorias, movimentos sociais/filosóficos/artísticos
  "Iluminismo", "Positivismo", "Capitalismo", "Socialismo", "Democracia",
  "Globalização", "Sustentabilidade", "Direitos Humanos", "Cidadania",
  "Meio Ambiente", "Justiça Social", "Cultura de Massa", "Sociedade Líquida",
  "Pós-modernidade", "Empatia", "Alteridade", "Estado de Bem-Estar Social",
  "Neoliberalismo", "Consumismo", "Alienacão", "Mais-valia", "Anomia",
  "Fato Social", "Darwinismo Social", "Efeito Estufa", "Crise Hídrica",
  "Revolução Industrial", "Guerra Fria", "Ditadura Militar", "Redemocratização",
  "Patriarcado", "Machismo", "Feminismo", "Racismo Estrutural",
  "Segurança Alimentar", "Inclusão Social", "Urbanização", "Gentrificação",
  "Ciberativismo", "Fake News", "Pós-Verdade", "Capitalismo de Vigilância",
  "Cultura do Cancelamento", "Economia Circular",
  "Objetivos de Desenvolvimento Sustentável",
  "Antropoceno", "Crise Migratória", "Xenofobia", "Populismo",
  "Polarização Política", "Direito à Cidade", "Consumo Consciente",
  "Mobilidade Urbana", "Direitos Digitais", "Inteligência Artificial",
  "Big Data", "Privacidade de Dados",
  "Reparação Histórica", "Minorias Sociais", "Diversidade Cultural",
  "Multiculturalismo", "Interseccionalidade",
  "Ação Afirmativa", "Cotas Raciais", "Lei Maria da Penha",
  "Estatuto da Criança e do Adolescente", "Estatuto do Idoso",
  "Estatuto da Pessoa com Deficiência", "Lei de Cotas",
  "Movimento Sem Terra", "Movimento Negro Unificado",
  "Movimento LGBTQIA+",
  "Colonialismo", "Neocolonialismo", "Desenvolvimentismo",
  "Populismo Econômico", "Estado Mínimo", "Estado de Exceção",
  "Teoria Crítica", "Escola de Frankfurt",
  "Existencialismo",
  "Racionalismo", "Empirismo", "Idealismo", "Materialismo",
  "Determinismo", "Subjetivismo", "Utilitarismo", "Deontologia",
  "Liberalismo", "Conservadorismo", "Anarquismo",
  "Renascimento", "Barroco", "Romantismo", "Realismo", "Naturalismo",
  "Modernismo", "Vanguardas Europeias", "Abstracionismo", "Surrealismo",
  "Tropicalismo", "Bossa Nova"
]


def extract_custom_repertoire(essay_text):
  """
  Função para extrair repertório customizado
  """
  found_repertoire = []
  text_lower = essay_text.lower() 
  for item in custom_list:
      # verificação simples de substring por enquanto
      if item.lower() in text_lower:
          found_repertoire.append(item)
  return list(set(found_repertoire)) # retorna itens únicos



# --- FUNÇÃO PARA AVALIAÇÃO DE MODELOS ---
def avaliar_modelos(models, param_grids, X_train, y_train, X_val, y_val):
  """
  Função para extrair NER (scpaCy)
  """
  from sklearn.model_selection import GridSearchCV, StratifiedKFold
  from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt
  results = {}

  # configurando o K-Folds
  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  for name, model in models.items():
      print(f"--- Avaliando {name} ---")
      
      # ajuste de hiperparâmetros, se aplicável
      if name in param_grids and param_grids[name]:
          grid_search = GridSearchCV(model, param_grid=param_grids[name], cv=kfold, scoring="accuracy")
          grid_search.fit(X_train, y_train)
          best_model = grid_search.best_estimator_
          print(f"Melhores parâmetros: {grid_search.best_params_}")
      else:
          best_model = model
          best_model.fit(X_train, y_train)
      
      # curvas de aprendizado
      train_sizes = np.linspace(0.1, 0.9, 10)
      train_errors = []
      val_errors = []
      
      for train_size in train_sizes:
          X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
          best_model.fit(X_partial, y_partial)
          train_errors.append(1 - best_model.score(X_partial, y_partial))
          val_errors.append(1 - best_model.score(X_val, y_val))
      
      plt.figure(figsize=(5, 3))
      plt.plot(train_sizes, train_errors, label="Treino", marker='o')
      plt.plot(train_sizes, val_errors, label="Validação", marker='s')
      plt.xlabel("Tamanho do Treinamento")
      plt.ylabel("Erro")
      plt.title(f"Curva de Aprendizado - {name}")
      plt.legend()
      plt.show()
      
      # predição
      y_pred = best_model.predict(X_val)

      # métricas
      acc = accuracy_score(y_val, y_pred)
      conf_matrix = confusion_matrix(y_val, y_pred)
      class_report = classification_report(y_val, y_pred)

      results[name] = {
          "Accuracy": acc,
          "Confusion Matrix": conf_matrix,
          "Classification Report": class_report
      }

      print(f"\nAcurácia: {acc:.4f}")
      print("\nMatriz de confusão:")
      print(conf_matrix)
      print("\nRelatório de classificação:")
      print(class_report)

  return results
