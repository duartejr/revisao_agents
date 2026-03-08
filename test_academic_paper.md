# Previsão de Vazão em Rios com Redes Neurais

## Introdução

Este artigo apresenta um estudo comparativo de modelos de redes neurais para previsão de vazão fluvial. A previsão precisa de vazão é crítica para o gerenciamento de recursos hídricos e prevenção de enchentes.

## Metodologia

Utilizamos bases de dados de vazão histórica de três rios brasileiros. Os modelos testados foram LSTM (Long Short-Term Memory) e o recente Chronos-2, um modelo de previsão temporal desenvolvido pela Amazon.

### Coleta de Dados

Os dados foram coletados de estações de monitoramento hidrometeorológico mantidas pela Agência Nacional de Águas. O período de coleta variou de 5 a 10 anos, dependendo da disponibilidade de dados para cada rio.

### Modelos Testados

1. **LSTM**: Implementação clássica com 2 camadas recorrentes de 128 neurônios
2. **Chronos-2**: Modelo pré-treinado disponibilizado pela Amazon Web Services

## Resultados

Os resultados preliminares indicam que o modelo Chronos-2 superou o LSTM em 15% em termos de erro quadrático médio (RMSE) para previsões de 7 dias.

## Conclusão

Este trabalho demonstra o potencial do Chronos-2 para aplicações de hidrologia operacional. Futuras pesquisas devem investigar o desempenho em diferentes escalas temporais e bacias hidrográficas.

## Referências

- Amazon Web Services (2024). Chronos: Pretrained Language Models for Time Series Forecasting
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation
