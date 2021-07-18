import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("Dados_Aeronautica_3.csv",sep=";")

# Visualização da base inserida.
data.loc[1]

# Investigação dos dados de quando há fatalidades
data.query('aeronave_fatalidades_total > 0').head(15)

# 1ª hipótese: Há um fator predominante quando há fatalidades.

data.query('Houve_fatalidades == 1 & fator_area != "#N/D" & fator_area != "***"').groupby(['fator_area','Houve_fatalidades']).agg({'fator_area': np.size, 'aeronave_tipo_operacao': np.size})

# Resultado: Não parece ter correlações fortes na análise superficial. Pode ser revisitado depois.

# 2ª hipótese: Algum ou alguns estados tem maior número de ocorrências.


data.groupby('ocorrencia_uf').agg({'ocorrencia_uf':np.size})

# Resultado: Nada que chame atenção. 
# SP tenderia a ter maior número de ocorrências por causa do tráfego aéreo da área.

# 3ª hipótese: As ocorrências tem sazonalidade. 

data['ocorrencia_ano'] = pd.DatetimeIndex(data['ocorrencia_dia']).year

data['ocorrencia_mes'] = pd.DatetimeIndex(data['ocorrencia_dia']).month

data.groupby('ocorrencia_ano').agg({'ocorrencia_ano':np.size})

a = data.groupby('ocorrencia_mes').agg({'ocorrencia_mes':np.size})
A = a/np.sum(a)
print(A)

# Resultado: Aparentemente não há % uma sazonalidade observada (nem de anos, nem de meses). 

# No entanto, iremos observar os anos 2011, 2012 e 2013 que obtiveram maior participação no número de ocorrências.

b = data.groupby('ocorrencia_ano').agg({'ocorrencia_ano':np.size})
B = b/np.sum(b)
print(B)

c = data.query('ocorrencia_ano == 2012').groupby('ocorrencia_mes').agg({'ocorrencia_mes':np.size})
C = c/np.sum(c)
print(C)

c = data.query('ocorrencia_ano == 2011').groupby('ocorrencia_mes').agg({'ocorrencia_mes':np.size})
C = c/np.sum(c)
print(C)

c = data.query('ocorrencia_ano == 2013').groupby('ocorrencia_mes').agg({'ocorrencia_mes':np.size})
C = c/np.sum(c)
print(C)

# Resultado: Porcentualmente, novamente, não há nada que chame atenção na análise preliminar. 

# Investigação de correlação Pearson.

# Transformar variáveis qualitativas para correlação. str para category.

for col in ['ocorrencia_classificacao', 'ocorrencia_cidade', 'ocorrencia_uf', 'ocorrencia_aerodromo','ocorrencia_tipo','ocorrencia_tipo_categoria','aeronave_operador_categoria','aeronave_tipo_veiculo','aeronave_fabricante','aeronave_motor_tipo','aeronave_motor_quantidade',
'aeronave_pais_fabricante','aeronave_registro_segmento','aeronave_fase_operacao','aeronave_tipo_operacao','aeronave_nivel_dano','fator_nome','fator_aspecto','fator_condicionante','fator_area']:
    data[col] = data[col].astype('category').cat.codes

correlationP = data.corr(method='pearson')
f, ax = plt.subplots(figsize=(40, 20))
mask = np.triu(np.ones_like(correlationP, dtype=bool))
cmap = sn.diverging_palette(230, 20, as_cmap=True)
sn.heatmap(correlationP, annot=True, mask = mask, cmap=cmap)

# Heatmap acima não mostra nenhuma correlação forte entre variáveis diferentes. 
# Um detalhe no entanto, é a correlação fraca entre fator_aspecto e o total_recomendacoes.

# Nesse caso, vale a pena utilizar o método Spearman, que é mais indicado para dados em cross-section. 

correlationS = data.corr(method='spearman')
f, ax = plt.subplots(figsize=(40, 20))
mask = np.triu(np.ones_like(correlationS, dtype=bool))
cmap = sn.diverging_palette(230, 20, as_cmap=True)
sn.heatmap(correlationS, annot=True, mask = mask, cmap=cmap)

# Para o método Spearman, é encontrado correlação forte entre fator_aspecto e total_recomendacoes.


# Desse ponto de partida, a análise da variável é recomendada. Na nossa base de dados original, a variável foi modificada para category.
# Sendo assim, voltaremos com a base original para investigar as variáveis.

# Hipótese 4: Um aumento na variável total_recomendacoes está diretamente ligada com um aumento de ocorrências de um único tipo em Fator_aspecto

data1 = pd.read_csv("Dados_Aeronautica_3.csv",sep=";")
data1.groupby('fator_aspecto').agg({'fator_aspecto':np.size})

# Para ver por mês de todos os dados (variável 'mês/ano' feita diretamente no excel). 
# O fator_aspecto de maior número é o Desempenho Humano. 

data1.query('fator_aspecto == "DESEMPENHO DO SER HUMANO"').groupby('mês/ano').agg({'fator_aspecto':np.size, 'total_recomendacoes':np.sum})

# Para ver a correlação, é necessário utilizar um gráfico de pontos

matriz = np.array(data1.query('fator_aspecto == "DESEMPENHO DO SER HUMANO"').groupby('mês/ano').agg({'mês/ano':np.size, 'total_recomendacoes':np.sum}))
y = matriz[:,0]
X = matriz[:,1]
print(X)
print(y)


plt.plot(X,y,'bo')
plt.xlabel('Total Recomendações')
plt.ylabel('Número de Ocorrências_Desempenho do Ser Humano')
plt.legend(['Correlação'])
plt.title('Total Recomendações X Número de Ocorrências')
plt.grid()
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lm_model = LinearRegression()
lm_model.fit(X.reshape(-1,1),y)
slope = lm_model.coef_
intercept = lm_model.intercept_
print("b0: \t{}".format(intercept))
print("b1: \t{}".format(slope[0]))

plt.scatter(X, y, s=3)
plt.plot(X, (X * slope + intercept), color='g')
plt.xlabel('Total Recomendações')
plt.ylabel('Número de Ocorrências_Desempenho do Ser Humano')
plt.legend(['Correlação'])
plt.title('Total Recomendações X Número de Ocorrências')

plt.show()

# Observando o gráfico acima, poderia-se dizer que a hipótese 4 não foi rejeitada. No entanto, é necessário uma análise mais aprofundada e, 
# inclusive, uma investigação se o modelo de regressão acima é o mais indicado para os dados.

# Desse modo, não é rejeitado que o aumento de recomendações está diretamente ligado ao aumento de ocorrências com fatores de desempenho humano.
