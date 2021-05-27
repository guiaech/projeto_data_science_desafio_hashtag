#!/usr/bin/env python
# coding: utf-8

# Projeto de ciência de dados com previsão de vendas baseado nos investimentos em meios de comunicação. Desafio proposto pela hashtag com dados fornecidos por eles.

# In[1]:


import pandas as pd
#importei pandas para manoseio das tabelas
df = pd.read_csv("advertising.csv")
#defini uma variavel para receber a tabela
display(df)
#comando para vizualização da tabela


# In[2]:


print(df.info())
#comando usado para ter uma visão geral das informações da lista 


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
#importei as bibliotecas graficas para gerar os graficos
sns.pairplot(df)
#indica tipo de grafico usado na biblioteca seaborn
plt.show()
#exibe o grafico criado


# In[4]:


sns.heatmap(df.corr() , cmap = 'Wistia' , annot = True)
#um novo grafico para melhor analise. Foi usado heatmap os argumentos usados foram "df.corr()" para definir quais parametros,"cmap = 'Wistia'" apos a cor e posteriormente "annot = True" para os numeros que apareceram no grafico
plt.show()
#exibe o grafico criado


# In[9]:


from sklearn.model_selection import train_test_split
#importei a biblioteca 
x = df.drop('Vendas', axis = 1)
#nomei a variavel x com todos valores menos o Vendas 
y = df['Vendas']
#nomei a variavel y com as vendas
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3 , random_state = 1)
#defini as variaveis de treino e de teste usando o metodo "train_test_split" determinando o tamanho de 30% 


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
#Importei as bibliotecas nescessarias

#Treino AI
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

#Teste AI
test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
rms_lin = np.sqrt(metrics.mean_squared_error(y_test, test_pred_lin) )
print(f"R2 da regressão linear: {r2_lin}")
print(f"RMS da regressão linear: {rms_lin}")

r2_rf = metrics.r2_score(y_test, test_pred_rf)
rms_rf = np.sqrt(metrics.mean_squared_error(y_test, test_pred_rf))
print(f"R2 da Random Forest:{r2_rf}")
print(f"RMS da Radom Forest:{rms_rf}")
      


# In[ ]:




