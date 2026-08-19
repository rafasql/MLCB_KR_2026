# LAB 01 - AULA 02 (MLCB): Classificador de Intenções

## Resultados do código

```text
--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]
```

## 1 - Avaliação dos resultados

Ao analisar os resultados, foi possível identificar que algumas mensagens foram classificadas corretamente e outras incorretamente.

A mensagem **"Quero consultar quanto dinheiro tenho"** foi classificada como `fazer_pix`, porém o resultado correto seria `consultar_saldo`. Portanto, essa classificação está incorreta.

A mensagem **"Pode me ajudar a fazer um pix?"** foi classificada como `fazer_pix`, que corresponde corretamente à intenção da mensagem.

A mensagem **"Gostaria de cancelar meu cartão de crédito"** foi classificada como `cancelar_conta`. Esse resultado não corresponde exatamente à mensagem, pois o usuário deseja cancelar o cartão de crédito, enquanto a intenção disponível no treinamento é cancelar a conta.

Portanto, dos três testes realizados, apenas um apresentou uma classificação totalmente correta.

## 2 - Como melhorar o resultado do algoritmo?

A principal maneira de melhorar o resultado seria aumentar e diversificar o dataset utilizado para o treinamento.

O dataset possui poucos exemplos para cada intenção, fazendo com que o algoritmo tenha dificuldade para identificar corretamente novas frases. Seria necessário adicionar mais exemplos de mensagens para cada intenção, utilizando diferentes formas de escrever a mesma solicitação.

Também seria importante criar uma intenção específica para o cancelamento de cartão, como `cancelar_cartao`, separada da intenção `cancelar_conta`.

Com mais dados de treinamento e intenções melhor definidas, o modelo teria mais informações para aprender as diferenças entre as solicitações e poderia apresentar resultados mais precisos.

## 3 - Função do LogisticRegression no algoritmo

O `LogisticRegression` é o modelo de Machine Learning responsável por realizar a classificação das mensagens.

Primeiro, o `CountVectorizer` transforma as mensagens de texto em números, representando a quantidade de vezes que determinadas palavras aparecem nas frases. Depois, o `LogisticRegression` utiliza esses dados para aprender a relação entre as palavras das mensagens e suas respectivas intenções.

Após o treinamento, quando uma nova mensagem é apresentada ao modelo, ele analisa as características dessa mensagem e prevê qual das intenções conhecidas é a mais provável.

Neste laboratório, as intenções utilizadas foram `consultar_saldo`, `fazer_pix` e `cancelar_conta`.


laboratorio 02

# ============================================================
# LAB 02 - AULA 02 (MLCB): Naive Bayes e Probabilidades
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset de Atendimento de E-Commerce
dados_ecommerce = {
    'mensagem': [
…print("--- RESULTADOS DO LAB 02 ---")
print(f"Mensagem de Teste: '{mensagem_teste[0]}'")
print(f"Intenção Predita: {predicao}\n")
print("--- Distribuição de Probabilidades por Classe ---")
for classe, prob in zip(classes, probabilidades):
    print(f"Classe [{classe}]: {prob * 100:.2f}%")

#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB02 você precisa copiar a saída do código (output) e adicionar as repostas das perguntas abaixo:
# 1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório

--- RESULTADOS DO LAB 02 ---
Mensagem de Teste: 'Gostaria de devolver o produto que comprei'
Intenção Predita: troca_devolucao

--- Distribuição de Probabilidades por Classe ---
Classe [duvida_frete]: 27.99%
Classe [rastrear_pedido]: 24.54%
Classe [troca_devolucao]: 47.46%

#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB02 você precisa copiar a saída do código (output) e adicionar as repostas das perguntas abaixo:

# 1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório

# 2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?

# 3 - Detalhe a função do Naive Bayes no algorítmo.


# LAB 02 - AULA 02 (MLCB): Classificador de Intenções

## Resultados do código

```text
--- RESULTADOS DO LAB 02 ---
Mensagem de Teste: 'Gostaria de devolver o produto que comprei'
Intenção Predita: troca_devolucao

--- Distribuição de Probabilidades por Classe ---
Classe [duvida_frete]: 27.99%
Classe [rastrear_pedido]: 24.54%
Classe [troca_devolucao]: 47.46%
```

## 1 - Avaliação dos resultados

O resultado apresentado foi correto. A mensagem **"Gostaria de devolver o produto que comprei"** foi classificada como `troca_devolucao`, que corresponde à intenção expressada pelo usuário.

A classe `troca_devolucao` também apresentou a maior probabilidade, com **47,46%**, enquanto `duvida_frete` apresentou **27,99%** e `rastrear_pedido` apresentou **24,54%**.

Apesar de a classificação ter sido correta, a probabilidade de `troca_devolucao` não foi muito alta, indicando que o modelo ainda apresenta certa incerteza na classificação. Isso pode estar relacionado à quantidade e à variedade dos exemplos utilizados no treinamento.

## 2 - Como melhorar o resultado do algoritmo?

Caso sejam identificados erros nas classificações, uma das principais formas de melhorar o resultado seria aumentar e diversificar o conjunto de dados utilizado no treinamento.

É importante fornecer ao algoritmo uma quantidade maior de exemplos para cada intenção, utilizando diferentes formas de escrever uma mesma solicitação. Dessa forma, o modelo consegue aprender melhor as características de cada classe.

Também é importante revisar os dados utilizados no treinamento, garantindo que as mensagens estejam corretamente associadas às suas respectivas intenções. Com um dataset maior, mais diversificado e bem organizado, o algoritmo tende a apresentar resultados mais precisos.

## 3 - Função do Naive Bayes no algoritmo

O Naive Bayes é um algoritmo de Machine Learning utilizado para classificação. Ele calcula a probabilidade de uma determinada mensagem pertencer a cada uma das categorias disponíveis e escolhe a categoria com maior probabilidade.

Neste laboratório, o algoritmo utiliza as palavras presentes nas mensagens para aprender quais termos estão mais relacionados a cada intenção. Quando uma nova mensagem é apresentada, o Naive Bayes analisa as palavras encontradas e calcula qual intenção possui maior probabilidade de corresponder àquela mensagem.

O modelo é chamado de "Naive" porque assume, de forma simplificada, que as características utilizadas para a classificação são independentes entre si. Mesmo sendo uma suposição simplificada, o Naive Bayes costuma funcionar bem em problemas de classificação de textos.
