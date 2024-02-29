#region TREINANDO MODELO - SENTIMENTOS


# """
# Importamos as bibliotecas necessárias. A transformers é a biblioteca da Hugging Face, e torch é o PyTorch, uma biblioteca popular
# para aprendizado profundo em Python.
# """
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.utils.data import DataLoader, TensorDataset
# from torch import nn, optim
# import torch


# """
# Carregamos o tokenizador e o modelo BERT pré-treinado. bert-base-uncased é um modelo BERT pré-treinado com letras minúsculas.
# """
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# """
# Dados de treinamento.
# """
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


# """
# Tokenizamos os textos usando o tokenizador BERT, adicionando padding e truncando para garantir que todos os inputs tenham o
# mesmo tamanho.
# """
# tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


# """
# Criamos um conjunto de dados usando TensorDataset para conter os IDs dos tokens, máscaras de atenção e rótulos.
# """
# dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], torch.tensor(labels))

# """
# Usamos um DataLoader para iterar sobre os dados em lotes durante o treinamento.
# """
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# """
# Escolhemos o otimizador AdamW e a função de perda de entropia cruzada para treinamento supervisionado de classificação.
# """
# optimizer = optim.AdamW(model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()

# # Treinamento do modelo
# num_epochs = 3
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for batch in dataloader:
#         inputs, attention_mask, labels = batch
#         outputs = model(inputs, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     average_loss = total_loss / len(dataloader)
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# # Salvar o modelo treinado
# model.save_pretrained('sentiment_model')



# # Carregar modelo treinado
# model = BertForSequenceClassification.from_pretrained("sentiment_model")

# input_text = "Este é um produto incrível!"
# input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")

# # Inferência com o modelo treinado
# with torch.no_grad():
#     outputs = model(input_ids)
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits).item()
# # Determinar sentimento previsto
# sentiment = "positivo" if predicted_class == 1 else "negativo"
# print(f"Sentimento previsto: {sentiment}")

#endregion

#region USANDO MODELO - PREENCHIMENTO DE ESPAÇOS EM BRANCO
# from transformers import pipeline

#  #Carregando a pipeline para preenchimento de máscaras (Masked Language Model)
# fill_mask = pipeline("fill-mask")

#  #Texto com uma máscara indicando onde a palavra está faltando
# text = "A inteligência artificial está transformando a forma como <mask> interagimos."

#  #Obtendo previsões para preencher a máscara
# result = fill_mask(text)

#  #Exibindo as previsões
# for prediction in result:
#    print(f"Palavra: {prediction['token_str']}, Pontuação: {prediction['score']}")

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

#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#
#cache_directory = "E:\\HuggingFaceCache"
#tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5", cache_dir=cache_directory)
#
#model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5", cache_dir=cache_directory)
#
#
#text = """
#Uma equipe de pesquisadores liderada pelo Dr. Elena Rodriguez, astrobióloga da Universidade de Tecnologia Avançada, encontrou evidências intrigantes durante uma análise detalhada de dados coletados pelo telescópio espacial de última geração.
#Os dados, obtidos ao longo de meses de observação, revelam padrões espectrais incomuns em uma região distante da galáxia. Esses padrões, até então inexplicáveis, sugerem a presença de moléculas complexas que são, surpreendentemente, análogas a certos compostos orgânicos encontrados na Terra, fundamentais para a vida como a conhecemos.
#O Dr. Rodriguez enfatiza que é prematuro concluir a existência de vida extraterrestre, mas as descobertas oferecem um ponto de partida empolgante para investigações mais aprofundadas. A comunidade científica agora está mobilizando esforços para direcionar telescópios adicionais para a área em questão, na esperança de confirmar e entender melhor a origem desses sinais misteriosos.
#A notícia já gerou um grande entusiasmo entre os entusiastas da exploração espacial e astrobiologia, alimentando a esperança de que, finalmente, estamos mais perto do que nunca de responder à antiga pergunta: estamos sozinhos no universo? O mundo aguarda ansiosamente por mais desenvolvimentos conforme os cientistas continuam a desvendar os segredos desse fascinante mistério cósmico.
#"""
#pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=len(text))
#
#result = pten_pipeline(f"""translate Portuguese to English: {text}""")
#print(result)
#endregion