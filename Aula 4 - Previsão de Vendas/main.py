
""" 
Projeto Ciência de Dados - Previsão de Vendas
Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
--
Data Science Project - Sales Prediction

Passo a Passo de um Projeto de Ciência de Dados / Steps of a Data Science Project
Passo 1: Entendimento do Desafio / Understanding the challenge
Passo 2: Entendimento da Área/Empresa / Understanding the Area/Company
Passo 3: Extração/Obtenção de Dados / Extraction/Gathering of data
Passo 4: Ajuste de Dados (Tratamento/Limpeza) / Pre-processing/cleaning data
Passo 5: Análise Exploratória - Exploratory Analysis
Passo 6: Modelagem + Algoritmos / Modeling + ALgorithms (AI, if needed)
Passo 7: Interpretação de Resultados - Interpreting results
 
 """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

chart = pd.read_csv("advertising.csv")
print(chart)
# Note: investments in tv, radio and journal are in thousands of reais and sales are in millions of reais

# Identifying possible problems in the chart for pre-processing
print(chart.info())
# All clean, let's proceed to exploratory analysys

#print(chart.corr())
sns.heatmap(chart.corr(), cmap="Wistia", annot=True)
#plt.show()
plt.savefig("Heatmap of correlation.png")
plt.clf() #clear

# Conclusion:
# High correlation between investments in TV and sales
# Low correlation between Radio and sales and even lower correlation between Journal and sales

# Using AI for prediction

# x = values used to make the prediction (the rest)
# y = values to be predicted
x = chart[["TV","Radio","Jornal"]]
y = chart["Vendas"]

# Separating train set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)

# Initializing models
linear_regression = LinearRegression()
random_forest = RandomForestRegressor()

# Training models
linear_regression.fit(x_train,y_train)
random_forest.fit(x_train,y_train)

# Testing models
LR_pred = linear_regression.predict(x_test)
RF_pred = random_forest.predict(x_test)

# Evaluation in terms of R^2 score
print(r2_score(y_test, LR_pred))
print(r2_score(y_test, RF_pred))

# Plotting predictions
aux_table = pd.DataFrame()
aux_table["Real Sale Values"] = y_test
aux_table["Linear Regression Predictions"] = LR_pred
aux_table["Random Forest Predictions"] = RF_pred

plt.figure(figsize=(15,6))
sns.lineplot(data=aux_table)
#plt.show()
plt.savefig("Comparision between real sale values and predictions made by Linear Regression model and Random Forest model.png")
plt.clf()

# Predictions of possible future investments to define how investment should be carried in the future
possibilities_chart = pd.read_csv("novos.csv")

future_pred = random_forest.predict(possibilities_chart)
print(future_pred)

# Plotting relevance of each type of investment for predictions using the Random Forest model
sns.barplot(x=x_train.columns, y=random_forest.feature_importances_)
#plt.show()
plt.savefig("Relevance of each type of investment used by Random Forest model.png")
plt.clf() #clear