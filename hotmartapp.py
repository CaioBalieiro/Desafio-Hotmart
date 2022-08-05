import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("hotmart.csv", delimiter=",")
df = df.drop(columns='Unnamed: 0')

# Transformando os ID em números inteiros
df['producer_id'] = df['producer_id'].astype(int)
df['product_id'] = df['product_id'].astype(int)
df['buyer_id'] = df['buyer_id'].astype(int)
df['affiliate_id'] = df['affiliate_id'].astype(int)

col = df.columns

st.sidebar.image('img/hotmart.png', width=140)

paginas = ['Home', 'Conjunto de dados', 'Análise Descritiva', 'Pergunta_n1', 'Pergunta_n2','Pergunta_n3', 'Pergunta_n4', 'Pergunta_n5', 'Modelo de Classificação']

pagina = st.sidebar.radio('Escolha a página:', paginas)

if pagina == 'Home':
    st.title('Aplicativo para análise de vendas Hotmart')
    '''
    # Descrição do APP:
    ### Este aplicativo foi criado para facilitar a apresentação das análises do conjunto de dados de vendas da Hotmart. 
    '''
    st.image('img/hotmart.png', use_column_width='always')

if pagina == 'Conjunto de dados':
    st.title('Conjunto de dados vendas hotmart 2016')
    st.subheader('Cabeçalho do conjunto de dados')
    st.write(df.head())

    st.subheader('Estatísticas de resumo do conjunto de dados')
    st.write(df.describe())

if pagina == 'Análise Descritiva':
    st.subheader('Descrição do conjunto de dados')
    st.write('O conjunto de dados é referente a dados de vendas da Hotmart no ano de 2016, uma base com mais de 1,5 \
             milhões de linhas e avaliados no mês de janeiro a junho de 2016')
    st.subheader('Quantidade de outliers no conjunto de dados:')
    st.write('Cerca de', 38192 , 'linhas, resultando em', 1561437, 'linhas para a análise')
    st.subheader('Quantidade de compras realizadas neste período:')
    st.write('Foram realizadas', 1561437,  'compras')
    st.subheader('Quantidade de produtos vendidos neste período:')
    st.write('Foram vendidos',17557, 'produtos')
    st.subheader('Quantidade de afiliados neste período:')
    st.write('Foram ', 22751, 'afiliados')
    st.subheader('Quantidade de  produtores neste período:')
    st.write('Foram', 7956, 'produtores')
    st.subheader('Quantidade de usuários  neste período:')
    st.write('Foram', 1074738,' usuários')



if pagina == 'Pergunta_n1':
    st.subheader('A empresa depende dos maiores produtores da plataforma? Ou seja, os produtores que mais vendem são \
    responsáveis pela maior parte do faturamento da empresa?')
    bt = st.number_input('Qual o top de produtores queres utilizar', value=100)
    # bt = st.selectbox('Qual o top de produtores queres utilizar?', [100, 200, 300, 400, 500])

    col1 = df['producer_id'].value_counts().head(bt).index
    col2 = df['producer_id'].value_counts().head(bt)
    consulta1 = pd.DataFrame(zip(col1, col2), columns=['producer_id', 'n_compras'])
    cons = df.drop_duplicates('producer_id')

    st.write('O número de produtores no conjunto de dados é:', cons.shape[0])
    # Tabela dos 100 maiores produtores da Hotmart
    tabela_100prod = pd.merge(consulta1, df, how='inner', on='producer_id')[
        ['producer_id', 'n_compras', 'purchase_value', 'score_faturamento', 'class_faturamento', 'peso_faturamento']]

    st.write('A porcetagem de faturamento referente ao top', bt, "produtores da Hotmart correspondem a", \
             (tabela_100prod['score_faturamento'].sum() / df['score_faturamento'].sum()).round(3) * 100)
    #plt.figure(figsize=(10, 6))
    #sns.barplot(x='class_faturamento', data=tabela_100prod)
    #f = sns.countplot(x='class_faturamento', data=tabela_100prod)
    #st.pyplot(f)
    st.write(tabela_100prod['class_faturamento'].value_counts(normalize=True).head(bt)*100)
if pagina == 'Pergunta_n2':
    st.subheader('Existe algum padrão ou tendência relevante nos dados?')
    add_selectbox = st.selectbox('Qual variável vai utilizar?', df['product_category'].unique())
    consulta_book = df[df['product_category'] == add_selectbox]
    consulta_book['purchase_date'] = pd.to_datetime(consulta_book['purchase_date'], format="%Y-%m-%d")
    consulta_book['purchase_date'] = consulta_book['purchase_date'].dt.date
    data = pd.DataFrame(consulta_book['purchase_date'].value_counts().sort_index())
    data.rename(columns={'purchase_date': 'Frequência de compras'}, inplace=True)
    fig = px.line(data, y='Frequência de compras', labels={
        "index": "Meses 2022"
    })
    st.subheader('Gráfico de séries')
    st.plotly_chart(fig)

    st.subheader('Gráfico de autocorrelação')
    g = plot_acf(data)
    st.pyplot(g)

    st.subheader('Aplicação do teste de hipótese de Dickey-Fuller Aumentado')
    result = adfuller(data)
    # print('Dickey-Fuller Aumentado')
    # print('Teste Estatístico: {:.4f}'.format(result[0]))
    st.write('Valor-p: {:.4f}'.format(result[1]))

    if result[1] > 0.05:
        st.write('Com base no p-valor acima, há indícios de que a série temporal da categoria', add_selectbox, 'tenha componentes de tedência ou sazonalidade.')
    else:
        st.write('A série da categoria', add_selectbox,'é estacionária, ou seja, não tem tedência nem sazonalidade.')

    # print('Valores Críticos:')
    # for key, value in result[4].items():
    #    print('\t{}: {:.4f}'.format(key, value))
if pagina == 'Pergunta_n3':

    segmento = pd.read_csv('segmento.csv')

    st.subheader('É possível segmentar os usuários com base em suas características(faturamento, nicho de produto, etc)?')
    st.write('A resposta é sim! Basta utilizar o tratamento adequado aos id dos usuários e utilizando técnicas, tais como  \
    (One Hot encoding) e utilizando um algoritmo de segmentação. Para este desafio foi adotado o modelo \
             Kmeans. No entanto, vale ressaltar que um modelo de segmentação de RFM também poderia ser aplicado, \
             utilizando as features buyer_id, purchase_date, purchase_value (sem z-score!) e construindo uma nova feature \
             relacionada a número de transações de cada buyer_id.')
    st.subheader('Descritiva dos cluster encontrados pelo Kmeans:')
    st.subheader('Quantidade de usuários em cada cluster:')
    st.write(segmento['cluster'].value_counts().sort_index())
    st.subheader('Faturamento (score de faturamento) da Hotmart de usuários em cada cluster:')
    st.write(segmento.groupby('cluster')['score_faturamento'].sum())

if pagina == 'Pergunta_n4':
    st.subheader(
        "Quais características mais impactam no sucesso de um produto? Ou seja, o que faz um produto vender mais?")

    prod_venda = pd.DataFrame(df['product_id'].value_counts().reset_index())
    prod_venda.rename(columns={'index': 'product_id', 'product_id': 'n_compras'}, inplace=True)

    # Motivos de vendas
    motivos = prod_venda.merge(df, how='inner', on='product_id')
    motivos = motivos.drop_duplicates('product_id')

    bt1 = st.number_input('Qual o top de produtos queres utilizar', value=500)
    st.subheader('Quais product_category mais vendem na hotmart e impactam no sucesso de vendas?')
    st.write(motivos['product_category'].head(bt1).value_counts(normalize=True) * 100)

    st.subheader('Quais product_niche mais vendem na hotmart e impactam no sucesso de vendas?')
    st.write(motivos['product_niche'].head(500).value_counts(normalize=True) * 100)

    st.subheader('Em quais purchase_device ocorrem as maiores compras de produto da hotmart e impactam no sucesso de vendas?')
    st.write(motivos['purchase_device'].head(500).value_counts(normalize=True) * 100)


    motivos['purchase_date'] = pd.to_datetime(motivos['purchase_date'], format="%Y-%m-%d")
    motivos['product_creation_date'] = pd.to_datetime(motivos['product_creation_date'], format="%Y-%m-%d")


    st.subheader('Os cursos com menor tempo de criação tendem a ter mais compras?')
    st.write((motivos['purchase_date'] - motivos['product_creation_date']).head(bt1).median())

    st.write('Enquanto que a mediana para os', bt1,'produtos que menos vendem tem uma mediana de:')
    st.write((motivos['purchase_date'] - motivos['product_creation_date']).tail(bt1).median())

if pagina == 'Pergunta_n5':
    st.subheader(
        "É possível estimar quanto de faturamento a Hotmart irá fazer nos próximos três meses a partir do último mês mostrado no dataset?")

    st.write('A reposta para essa pergunta é sim (do ponto de vista de codar é possível), entretanto, é \
     necessário avaliar se a quantidade de dados é adequada, \
     para este desafio foi dada apenas uma amostra cujo a variação era de janeiro a junho de 2016 (6 valores do passado), são poucos meses \
     observados para que o modelo possa aprender (como as observações eram apenas 6, nem deu para separar em \
      treino e teste). Por essa razão o modelo apresentou previsões semelhantes para os demais três meses.')

    st.write('Solução para isto: 1 - Aumentar a quantidade de anos para que a previsão do modelo faça sentido, \
             e a Hotmart possa usufruir dos resultados do modelo. Isto é fácil de resolver pois a Hotmart tem dados suficiente para melhorar\
             e aumentar a quantidade de meses observados. 2 - Utilizar o auxílio do Pyspark para trabalhar com esse \
             aumento de dados')
    data_prev = pd.read_csv('previsao_faturamento.csv')

    col2 = ['score_faturamento', 'purchase_value']

    add = st.selectbox('Qual variável vai utilizar?', col2)
    mes = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho']
    fig1 = px.line(data_prev, x=mes, y=add, labels={
        "x": "Meses 2016"
    })

    st.subheader('Gráfico de séries')
    st.plotly_chart(fig1)

    model = auto_arima(data_prev[add])

    n = st.number_input('Quantos meses quer estimar?', value=3)

    forecast = model.predict(n)
    if n == 1:
        meses = ['Julho']
    if n == 2:
        meses = ['Julho', 'Agosto']
    if n == 3:
        meses = ['Julho', 'Agosto', 'Setembro']
    if n == 4:
        meses = ['Julho', 'Agosto', 'Setembro', 'Outubro']
    if n == 5:
        meses = ['Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro']
    if n == 6:
        meses = ['Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

    tabela = pd.DataFrame(zip(meses, forecast), columns=['Mes', 'Estimado'])

    st.write(tabela)
if pagina == 'Modelo de Classificação':
    st.subheader('Modelo de Random Forest para predizer se o valor da compra está abaixo ou acima da média dos valores de todas as compras')
    st.subheader('Métricas do modelo escolhido')
    st.write('Acurácia:', 0.923)
    st.write('F1-Score:', 0.920)
    st.subheader('Matriz de confusão')
    st.image('img/confusion.png',  width=800)

    X = pd.read_csv('test.csv')
    X = X.drop(columns='Unnamed: 0')
    # load the model from disk
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    buttom = st.number_input('Quantos usuários quer predizer?', value=1)

    result = loaded_model.predict(X.iloc[0:buttom])

    st.write('Se o usuário da Hotmart fez uma compra cujo o valor é maior que a média geral dos valores:',1 )
    st.write('Se o usuário da Hotmart fez uma compra cujo o valor é menor que a média geral dos valores:', 0)
    st.write(result)
