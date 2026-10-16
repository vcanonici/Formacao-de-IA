# Exemplo didático de PR — Classificador MNIST básico em Python

**Objetivo:** demonstrar como estruturar um Pull Request pequeno, revisável e verificável, com código, testes e evidências. **Não é a solução da rede neural do Exercício 3.**

Este exemplo implementa o classificador **nearest-centroid**: calcula uma imagem média para cada classe com exemplos de treino e atribui à nova imagem a classe cujo centroide está mais próximo. **Não é uma rede neural**, não usa gradiente, não fornece modelo para navegador e não cria um site.

## Execução

Use os quatro arquivos do [MNIST do curso](../../Dataset-MNIST/README.md). Com Python 3.10 ou superior, execute na raiz do repositório:

    python Dia2/Exemplos/PR-Classificador-Simples/mnist_centroid.py --data-dir Dia2/Dataset-MNIST/data --train-limit 2000 --test-limit 200

O resultado é um JSON contendo método, quantidade de imagens de treinamento, total de imagens avaliadas, acertos e **acurácia realmente calculada**.

Os limites usam subconjuntos para uma demonstração rápida. **Não apresentar essas métricas como resultado da avaliação do MNIST inteiro.** Para usar os conjuntos completos, executar com --train-limit 0 --test-limit 0. O algoritmo utiliza apenas a biblioteca padrão e lê os arquivos originais IDX compactados diretamente.

O método por centroides é uma referência didática, **não é arquitetura recomendada para o projeto final**. As decisões de rede neural devem ser investigadas pelos participantes.

## Testes

Os testes criam pequenos arquivos IDX artificiais em diretórios temporários. **Não é preciso baixar o MNIST para testá-los.**

    python -m unittest discover -s Dia2/Exemplos/PR-Classificador-Simples -p 'test_*.py' -v

Cobertura básica: leitura IDX, limite de amostras, treinamento, previsão baseada somente nos pixels, avaliação, entrada inválida e execução completa da CLI.

**Limitação:** esses testes verificam o funcionamento básico, mas não medem a acurácia real no MNIST. Para avaliá-la, execute o programa com os arquivos reais e registre o resultado observado. O PR não presume um valor de acurácia.

## Critérios para revisar este PR

- **Escopo:** somente o classificador de exemplo, sua documentação, testes e workflow de CI.
- **Integridade:** rótulos são usados para treinar e comparar previsões, mas não entram na função de previsão.
- **Reprodutibilidade:** comandos e dependências documentados; testes locais independem de rede.
- **Rastreabilidade:** evidências objetivas de validação no corpo do PR e alterações concentradas nos arquivos pertinentes.
- **Limitações:** sem arquitetura neural, sem treinamento por gradiente, sem pesos exportados, sem visualização e sem GitHub Pages.

Como referência de organização das tarefas, consulte a [Issue didática #1](https://github.com/vcanonici/Formacao-de-IA/issues/1). **Este PR não fecha a Issue #1**, cujo objetivo é especificar uma rede neural. Os alunos deverão relacionar seus próprios PRs às Issues correspondentes nos seus repositórios.

**Não copiar automaticamente este baseline como solução dos exercícios.** O objetivo é observar o ciclo Issue → alteração pequena → testes → descrição do PR → revisão → correção.
