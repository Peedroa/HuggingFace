#region TREINANDO MODELO - SENTIMENTOS

#----------------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torch.nn.utils.rnn import pad_sequence
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# # Dados de treinamento (exemplo simples)
# texts = [
#     "Este é um ótimo produto!",
#     "Não gostei da qualidade.",
#     "Adorei a experiência de uso.",
#     "O serviço ao cliente foi incrível!",
#     "A entrega foi rápida e eficiente.",
#     "Não recomendaria este produto a ninguém.",
#     "Este produto é muito ruim."
# ]
# labels = [1, 0, 1, 1, 1, 0, 0] # 1 para positivo, 0 para negativo

# # Tokenização
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Criando dataset
# class SentimentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         # Tokenizando o texto usando o BERT Tokenizer
#         tokenized_text = self.tokenizer(
#             self.texts[idx],
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "text": self.texts[idx],  # Texto original
#             "label": torch.tensor(self.labels[idx], dtype=torch.long),  # Rótulo (1 para positivo, 0 para negativo)
#             "input_ids": tokenized_text["input_ids"],  # IDs de tokens gerados pelo Tokenizer
#         }

# def collate_fn(batch):
#     # Empacotando os input_ids, adicionando padding para lidar com sequências de comprimentos variáveis
#     input_ids = pad_sequence([item["input_ids"].squeeze() for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
#     labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  # Coletando os rótulos

#     return {"input_ids": input_ids, "label": labels}

# # Criando o DataLoader com a função collate_fn personalizada
# dataset = SentimentDataset(texts, labels, tokenizer)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# # Modelo BERT para classificação de sequências
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# # ajustar os pesos do modelo durante o treinamento para minimizar a função de perda.
# # lr=2e-5 define a taxa de aprendizado (learning rate) para o otimizador. A taxa de aprendizado controla o tamanho dos passos que o otimizador dá para atualizar os pesos do modelo. 

# # Configurando otimizador e função de perda
# optimizer = AdamW(model.parameters(), lr=2e-5)
# criterion = nn.CrossEntropyLoss()

# # Treinamento do modelo
    
# # num_epochs - é o número total de vezes que o conjunto de treinamento será percorrido durante o treinamento.
# # for epoch in range(num_epochs) - percorre cada época, permitindo que o modelo seja treinado em várias iterações.

# num_epochs = 3
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         optimizer.zero_grad()
#         outputs = model(batch["input_ids"], labels=batch["label"])
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Salvando o modelo treinado
# model.save_pretrained("modelo_treinado")

# # Carregando o modelo treinado
# model = BertForSequenceClassification.from_pretrained("modelo_treinado")

# # Tokenizando um novo texto de entrada para fazer uma previsão
# input_text = "Este é um produto incrível!"
# input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(input_ids)
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits).item()

# # Exibindo a previsão
# sentiment = "positivo" if predicted_class == 1 else "negativo"
# print(f"Sentimento previsto: {sentiment}")

#----------------------------------------------------------------------

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
# token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
# model_name = 'recogna-nlp/ptt5-base-summ'

# cache_directory= "E:\\HuggingFaceCache"

# tokenizer = T5Tokenizer.from_pretrained(token_name, cache_dir=cache_directory)
# model_pt = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_directory)

# text = '''

# Descoberta Arqueológica Revela Cidade Antiga Sob as Ruas de Metrópole Moderna

# Em uma surpreendente descoberta arqueológica, uma equipe de pesquisadores anunciou hoje a revelação de uma cidade antiga enterrada sob as ruas movimentadas da metrópole moderna de Urbis Prime. A descoberta ocorreu durante escavações de rotina para a construção de uma nova linha de metrô, quando os arqueólogos notaram vestígios de estruturas antigas.

# A cidade subterrânea, que remonta a aproximadamente 2.000 anos, revela uma complexa rede de ruas, edifícios e até mesmo uma praça central. Artefatos cuidadosamente preservados, incluindo cerâmicas, moedas antigas e fragmentos de murais, foram desenterrados, oferecendo uma visão fascinante da vida cotidiana na época.

# Os arqueólogos acreditam que a cidade pode ter sido um importante centro cultural e comercial em seu tempo, embora a sua existência tenha sido perdida ao longo dos séculos. A descoberta levanta questões intrigantes sobre a história da região e como a paisagem urbana evoluiu ao longo dos milênios.

# Os especialistas agora planejam realizar análises mais aprofundadas para datar com precisão os artefatos e entender melhor o papel dessa cidade antiga na história da civilização. A notícia já despertou o interesse de historiadores, arqueólogos e moradores locais, ansiosos para aprender mais sobre o passado enterrado sob seus próprios pés.

# um fato importante é que acharam corpos.
# '''

# tempo_inicio = time.time()


# inputs = tokenizer.encode(text, truncation=True, return_tensors='pt')
# summary_ids = model_pt.generate(inputs ,max_length=200, min_length=100, num_beams=10, no_repeat_ngram_size=3, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0])
# tempo_fim = time.time()
# tempo_total = tempo_fim - tempo_inicio
# print(f"Tempo total de execução: {tempo_total} segundos")

# print(summary)

#endregion
 
