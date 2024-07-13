# Verificar e instalar pacotes necessários
if (!require('randomForest')) install.packages('randomForest', dependencies = TRUE)
if (!require('caret')) install.packages('caret', dependencies = TRUE)
if (!require('nnet')) install.packages('nnet', dependencies = TRUE)

# Carregar as bibliotecas necessárias
library(randomForest)
library(caret)
library(nnet)
library(readr)
library(dplyr)
library(tidyr)
library(class)

# Função para carregar e ajustar os tipos de dados
load_data <- function(file_path, value_name) {
  data <- read_csv(file_path, show_col_types = FALSE)
  data <- data %>%
    mutate(across(-Anos, as.character)) %>%
    pivot_longer(cols = -Anos, names_to = "country", values_to = value_name)
  data[[value_name]] <- as.numeric(data[[value_name]])
  return(data)
}

# Carregar os dados
internet_data <- load_data('C:/Users/nokas/Desktop/UAb 23-24/2º Semestre/Raciocínio e Representação do Conhecimento/e-folio B/final/AssinaturasAcessoInternet.csv', 'internet_access')
wage_gap_data <- load_data('C:/Users/nokas/Desktop/UAb 23-24/2º Semestre/Raciocínio e Representação do Conhecimento/e-folio B/final/DisparidadeSalarialHomensMulheres.csv', 'wage_gap')
learning_data <- load_data('C:/Users/nokas/Desktop/UAb 23-24/2º Semestre/Raciocínio e Representação do Conhecimento/e-folio B/final/ParticipacaoAdultosAprendizagem.csv', 'adult_learning_participation')
unemployment_data <- load_data('C:/Users/nokas/Desktop/UAb 23-24/2º Semestre/Raciocínio e Representação do Conhecimento/e-folio B/final/TaxaDesempregoLongaDuracao.csv', 'unemployment_rate')

# Verificar os dados carregados
print("Dados de Acesso à Internet")
print(head(internet_data))
print("Dados de Disparidade Salarial")
print(head(wage_gap_data))
print("Dados de Participação de Adultos na Aprendizagem")
print(head(learning_data))
print("Dados de Taxa de Desemprego de Longa Duração")
print(head(unemployment_data))

# Unir os dados por um identificador comum (Anos e country)
data <- merge(merge(merge(internet_data, wage_gap_data, by = c("Anos", "country")),
                    learning_data, by = c("Anos", "country")),
              unemployment_data, by = c("Anos", "country"))

# Verificar o conjunto de dados combinado
print("Dados Combinados")
print(head(data))

# Selecionar as colunas relevantes e remover anos consecutivos
data <- data %>%
  select(Anos, country, internet_access, wage_gap, unemployment_rate, adult_learning_participation) %>%
  arrange(country, Anos)

# Eliminar anos consecutivos (mantendo apenas anos pares para simplificação)
data <- data %>% filter(as.integer(Anos) %% 2 == 0)

# Tratar valores em falta
print("Dados Antes de Remover NAs")
print(data)
data <- na.omit(data)
print("Dados Depois de Remover NAs")
print(data)

# Verificar que países estão incluídos
print("Países Incluídos")
print(unique(data$country))

# Discretizar a variável target para classificação
data$adult_learning_participation <- cut(data$adult_learning_participation, 
                                         breaks = c(-Inf, median(data$adult_learning_participation, na.rm = TRUE), Inf), 
                                         labels = c("low", "high"))

# Converter variáveis em fatores
data[, 3:5] <- lapply(data[, 3:5], factor)

# Subtrair 1 da variável de resposta para ter valores binários 0 e 1
data$adult_learning_participation <- as.numeric(data$adult_learning_participation) - 1

# Dividir os dados em conjuntos de treino e teste
set.seed(42)
trainIndex <- createDataPartition(data$adult_learning_participation, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Normalizar os dados
preProcValues <- preProcess(dataTrain[, -c(1, 2, 6)], method = c("center", "scale"))
dataTrain_norm <- predict(preProcValues, dataTrain[, -c(1, 2, 6)])
dataTest_norm <- predict(preProcValues, dataTest[, -c(1, 2, 6)])

# Verificar se há valores NA ou NaN nos dados normalizados
sum(is.na(dataTrain_norm))
sum(is.na(dataTest_norm))

# Garantir que todas as colunas de input para knn são numéricas
str(dataTrain_norm)
str(dataTest_norm)

# Modelos de aprendizagem
###### Árvores de Decisão ######
rf_model <- randomForest(x = dataTrain[, 3:5], y = factor(dataTrain$adult_learning_participation, levels = c(0, 1)), ntree = 1, importance = TRUE)
print(rf_model)

# Mostrar a árvore de decisão da Random Forest
tree_info <- getTree(rf_model, 1, labelVar = TRUE)
print(tree_info)

# Verificar o tipo de modelo Random Forest para validar que foi uma classificação e não uma regressão
print(rf_model$type)

# Prever usando o modelo Random Forest
rf_predictions <- predict(rf_model, dataTest[, 3:5])
rf_accuracy <- confusionMatrix(factor(rf_predictions, levels = c(0, 1)), 
                               factor(dataTest$adult_learning_participation, levels = c(0, 1)))$overall['Accuracy']

# Verificar o valor de OOB error rate
cat("OOB Error Rate:", rf_model$err.rate[nrow(rf_model$err.rate), "OOB"], "\n")

###### K Vizinhos Mais Próximos usando knn3 ######
knn_model <- knn3(x = dataTrain_norm, y = factor(dataTrain$adult_learning_participation, levels = c(0, 1)), k = 5)
knn_predictions <- predict(knn_model, dataTest_norm)

# Verificar a estrutura da previsão
str(knn_predictions)

# Converter as probabilidades em classes binárias 0 e 1
knn_predictions_class <- ifelse(knn_predictions[, 2] > 0.5, 1, 0)

# Avaliar a precisão do modelo KNN
knn_accuracy <- confusionMatrix(factor(knn_predictions_class, levels = c(0, 1)), 
                                factor(dataTest$adult_learning_participation, levels = c(0, 1)))$overall['Accuracy']

###### Redes Neuronais ######
nn_model <- nnet(x = dataTrain_norm, y = dataTrain$adult_learning_participation, size = 1, maxit = 500)
print(nn_model)

# Mostrar os pesos da rede neural
print(nn_model$wts)

# Prever usando o modelo de Redes Neuronais
nn_predictions <- predict(nn_model, dataTest_norm, type = "raw")
nn_predictions_class <- ifelse(nn_predictions > 0.5, 1, 0)

# Avaliar a precisão do modelo de Redes Neuronais
nn_accuracy <- confusionMatrix(factor(nn_predictions_class, levels = c(0, 1)), 
                               factor(dataTest$adult_learning_participation, levels = c(0, 1)))$overall['Accuracy']

# Resultados
results <- data.frame(
  Model = c("Decision Tree", "K-Nearest Neighbors", "Neural Network"),
  Accuracy = c(rf_accuracy, knn_accuracy, nn_accuracy)
)

# Mostrar resultados
print(results)
