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
