#region TREINANDO MODELO - SENTIMENTOS

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from torch.nn.utils.rnn import pad_sequence
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# # Dados de exemplo
# texts = [
#     "Este é um ótimo produto!",
#     "Não gostei da qualidade.",
#     "Adorei a experiência de uso.",
#     "O serviço ao cliente foi incrível!",
#     "A entrega foi rápida e eficiente.",
#     "Não recomendaria este produto a ninguém.",
#     "Este produto é muito ruim."
# ]
# labels = [1, 0, 1, 1, 1, 0, 0] 

# # Tokenizador BERT
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Classe para o conjunto de dados
# class SentimentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer):
#         """
#         Construtor da classe SentimentDataset.

#         Parâmetros:
#         - texts: Lista de strings contendo os textos.
#         - labels: Lista de rótulos associados aos textos.
#         - tokenizer: Objeto do tokenizador BERT.
#         """
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         """
#         Obtém um item do conjunto de dados.

#         Parâmetros:
#         - idx: Índice do item a ser recuperado.

#         Retorna:
#         Dicionário contendo o texto original, rótulo e IDs dos tokens.
#         """
#         # Tokenizar texto
#         tokenized_text = self.tokenizer(
#             self.texts[idx],
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "text": self.texts[idx],  
#             "label": torch.tensor(self.labels[idx], dtype=torch.long),
#             "input_ids": tokenized_text["input_ids"], 
#         }

# # Função para combinar batches no DataLoader
# def collate_fn(batch):
#     """
#     Função utilizada pelo DataLoader para combinar batches.

#     Parâmetros:
#     - batch: Lista de itens do conjunto de dados.

#     Retorna:
#     Dicionário contendo input_ids e labels.
#     """
#     # Preencher sequências para terem o mesmo comprimento
#     input_ids = pad_sequence([item["input_ids"].squeeze() for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
#     labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  

#     return {"input_ids": input_ids, "label": labels}

# # Criar conjunto de dados e DataLoader
# dataset = SentimentDataset(texts, labels, tokenizer)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# # Inicializar modelo BERT para classificação de sequências
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# # Configurar otimizador e função de perda
# optimizer = AdamW(model.parameters(), lr=2e-5)
# criterion = nn.CrossEntropyLoss()

# # Número de épocas de treinamento
# num_epochs = 3

# # Loop de treinamento
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # Limpar gradientes
#         optimizer.zero_grad()
        
#         # Obter saída do modelo
#         outputs = model(batch["input_ids"], labels=batch["label"])
#         loss = outputs.loss
        
#         # Retropropagar o erro e atualizar pesos
#         loss.backward()
#         optimizer.step()

# # Salvar modelo treinado
# model.save_pretrained("modelo_treinado")

# # Carregar modelo treinado
# model = BertForSequenceClassification.from_pretrained("sentiment_model")

# # Inferência com o modelo treinado
# input_text = "Este é um produto incrível!"
# input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(input_ids)
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits).item()

# # Determinar sentimento previsto
# sentiment = "positivo" if predicted_class == 1 else "negativo"
# print(f"Sentimento previsto: {sentiment}")

#region USANDO MODELO DE SENTIMENTOS 
#----------------------------------------------------------------------

#from transformers import pipeline

#sentiment_classifier = pipeline('sentiment-analysis')
#result = sentiment_classifier('Este é um exemplo de texto!')
#print(result)

#----------------------------------------------------------------------
#endregion

#endregion

#region USANDO MODELO - PREENCHIMENTO DE ESPAÇOS EM BRANCO
#from transformers import pipeline

# Carregando a pipeline para preenchimento de máscaras (Masked Language Model)
#fill_mask = pipeline("fill-mask")

# Texto com uma máscara indicando onde a palavra está faltando
#text = "A inteligência artificial está transformando a forma como <mask> interagimos."

# Obtendo previsões para preencher a máscara
#result = fill_mask(text)

# Exibindo as previsões
#for prediction in result:
    #print(f"Palavra: {prediction['token_str']}, Pontuação: {prediction['score']}")

#endregion

#region TREINANDO MODELO - CLASSIFICAÇÃO DE TEXTO
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Carregar os dados (substitua isso com seus próprios dados)
# texts = [...]  # Lista de textos
# labels = [...]  # Lista de rótulos (0 para negativo, 1 para positivo)

# # Dividir os dados em conjuntos de treinamento, validação e teste
# train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
# val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=0.5, random_state=42)

# # Definir uma classe Dataset personalizada para carregar os dados
# class CustomDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
#         encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# # Tokenizador BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Carregamento do tokenizador

# # Hiperparâmetros
# batch_size = 16
# max_length = 128
# learning_rate = 2e-5
# epochs = 3

# # Dataset e DataLoader
# train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)  # Criação do dataset
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Criação do DataLoader

# # Modelo BERT pré-treinado para classificação de sequência
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Carregamento do modelo

# # Otimizador
# optimizer = AdamW(model.parameters(), lr=learning_rate)  # Criação do otimizador

# # Loop de Treinamento
# for epoch in range(epochs):
#     model.train()  # Colocando o modelo no modo de treinamento
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Avaliação do Modelo
# model.eval()  # Colocando o modelo no modo de avaliação


#endregion

#region USANDO MODELO - SUMMARIZATION

# from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
# import time

# # Nomes dos modelos e diretório de cache
# token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
# model_name = 'recogna-nlp/ptt5-base-summ'
# cache_directory = "E:\\HuggingFaceCache"

# # Inicializando o tokenizador com o modelo de vocabulário pré-treinado
# tokenizer = T5Tokenizer.from_pretrained(token_name, cache_dir=cache_directory)

# # Inicializando o modelo de geração condicional T5 para PyTorch
# model_pt = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_directory)

# # Texto que será resumido
# text = '''
#    Adicione aqui o texto que você deseja resumir.
# '''

# # Medindo o tempo de execução
# tempo_inicio = time.time()

# # Tokenizando o texto e convertendo para tensores PyTorch
# inputs = tokenizer.encode(text, truncation=True, return_tensors='pt')

# # Gerando o resumo usando o modelo pré-treinado
# summary_ids = model_pt.generate(inputs,
#                                 max_length=200,      # comprimento máximo do resumo gerado
#                                 min_length=100,      # comprimento mínimo do resumo gerado
#                                 num_beams=10,        # número de feixes usados na geração
#                                 no_repeat_ngram_size=3,  # evita a repetição de trigramas no resumo
#                                 early_stopping=True   # interrompe a geração assim que todos os feixes terminam
#                                 )

# # Decodificando os IDs do resumo de volta para texto
# summary = tokenizer.decode(summary_ids[0])

# # Medindo o tempo de execução total
# tempo_fim = time.time()
# tempo_total = tempo_fim - tempo_inicio

# # Imprimindo o resumo gerado e o tempo de execução
# print(f"Tempo total de execução: {tempo_total} segundos")
# print(summary)


#endregion
 
#region USANDO MODELO - OUTRO SUMMARIZATION

# from transformers import T5Tokenizer

# # PyTorch model
# from transformers import T5Model, T5ForConditionalGeneration

# # Definindo os nomes dos modelos e o diretório de cache
# token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
# model_name = 'phpaiola/ptt5-base-summ-xlsum'
# cache_directory= "E:\\HuggingFaceCache"

# # Inicializando o tokenizador com o modelo de vocabulário pré-treinado
# tokenizer = T5Tokenizer.from_pretrained(token_name, cache_dir=cache_directory)

# # Inicializando o modelo de geração condicional T5 para PyTorch
# model_pt = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_directory)

# # Texto que será resumido
# text = '''
#    Adicione aqui o texto que você deseja resumir.
# '''

# # Tokenizando o texto e convertendo para tensores PyTorch
# inputs = tokenizer.encode(text, max_length=len(text), truncation=True, return_tensors='pt')

# # Gerando o resumo usando o modelo pré-treinado
# summary_ids = model_pt.generate(inputs, 
#                                 max_length=200,      # comprimento máximo do resumo gerado
#                                 min_length=100,      # comprimento mínimo do resumo gerado
#                                 num_beams=5,         # número de feixes usados na geração
#                                 no_repeat_ngram_size=3,  # evita a repetição de trigramas no resumo
#                                 early_stopping=True   # interrompe a geração assim que todos os feixes terminam
#                                 )

# # Decodificando os IDs do resumo de volta para texto
# summary = tokenizer.decode(summary_ids[0])

# # Imprimindo o resumo gerado
# print(summary)

#endregion

#region USANDO MODELO - TEXT EMBEDDINGS

# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F

# cache_directory= "E:\\HuggingFaceCache"
# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is converted']

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=cache_directory)
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=cache_directory)

# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# print(sentence_embeddings)

#endregion

#region USANDO MODELO - TRANSLATION

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# cache_directory = "E:\\HuggingFaceCache"
# tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5", cache_dir=cache_directory)

# model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5", cache_dir=cache_directory)


# text = """
# Uma equipe de pesquisadores liderada pelo Dr. Elena Rodriguez, astrobióloga da Universidade de Tecnologia Avançada, encontrou evidências intrigantes durante uma análise detalhada de dados coletados pelo telescópio espacial de última geração.
# Os dados, obtidos ao longo de meses de observação, revelam padrões espectrais incomuns em uma região distante da galáxia. Esses padrões, até então inexplicáveis, sugerem a presença de moléculas complexas que são, surpreendentemente, análogas a certos compostos orgânicos encontrados na Terra, fundamentais para a vida como a conhecemos.
# O Dr. Rodriguez enfatiza que é prematuro concluir a existência de vida extraterrestre, mas as descobertas oferecem um ponto de partida empolgante para investigações mais aprofundadas. A comunidade científica agora está mobilizando esforços para direcionar telescópios adicionais para a área em questão, na esperança de confirmar e entender melhor a origem desses sinais misteriosos.
# A notícia já gerou um grande entusiasmo entre os entusiastas da exploração espacial e astrobiologia, alimentando a esperança de que, finalmente, estamos mais perto do que nunca de responder à antiga pergunta: estamos sozinhos no universo? O mundo aguarda ansiosamente por mais desenvolvimentos conforme os cientistas continuam a desvendar os segredos desse fascinante mistério cósmico.
# """
# pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=len(text))

# result = pten_pipeline(f"""translate Portuguese to English: {text}""")
# print(result)
#endregion

