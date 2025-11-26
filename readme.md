# GeneticPromptLab (Local Edition) 🧬🤖

> Otimização Evolutiva de Prompts para Geração de Código Pandas usando LLMs Locais.

Este projeto implementa um **Algoritmo Genético (AG)** para evoluir e otimizar automaticamente prompts de sistema (System Prompts). O objetivo é encontrar a melhor instrução possível para que um LLM local (via **Ollama**) traduza perguntas em linguagem natural para código **Python Pandas** correto.
Base de Dados utilizada: https://dadosabertos.bcb.gov.br/dataset/desenrola-brasil

## 🏗️ Arquitetura do Projeto

O projeto utiliza a biblioteca **DEAP** para o motor evolucionário e o **Ollama** para inferência local de LLMs (como Llama 3 ou Gemma), eliminando custos de API e latência de rede.

### Fluxo de Funcionamento:
1.  **População Inicial:** O AG cria prompts aleatórios combinando "genes" (Persona, Exemplos Few-Shot, Regras de Formato, etc.).
2.  **Avaliação (Fitness):** Cada prompt é testado contra um **Dataset de Treino** (perguntas + queries gabarito).
3.  **Inferência Local:** O prompt + pergunta são enviados para o LLM local via Ollama.
4.  **Cálculo de Score:** O código gerado é comparado com o gabarito usando uma métrica de similaridade flexível (lógica + sintaxe).
5.  **Evolução:** Os melhores prompts se reproduzem (Crossover) e sofrem mutações para criar a próxima geração.
6.  **Validação:** O melhor prompt final é testado contra um **Dataset de Teste** (inédito) para medir a generalização.

---

## ⚙️ Pré-requisitos do Sistema

Como o projeto lida com processamento pesado de PDF e IA Local, você precisará instalar:

### 1. Python 3.8+
* **Python**:
    * [Download para Python](https://www.python.org/downloads/)

### 2. Ollama (LLM Local)
Este projeto roda 100% localmente para garantir privacidade dos dados.
1.  Baixe e instale o [Ollama](https://ollama.com/).
2.  No terminal, baixe os modelos necessários:
    ```bash
    ollama pull llama3
    ```
---

# 🖥️ Como Usar
Execute o orquestrador principal:
```bash
    python -m genetic_lab
```

## 📂 Estrutura de Pastas

```text
lab_session_genetic_algorithms/
├── data/                          # Datasets (CSV) e Gabaritos (JSON)
│   ├── dados_desenrola_train.csv
│   ├── perguntas_gabarito_train.json  (Dataset de Treino - O AG estuda isso)
│   ├── dados_desenrola_test.csv
│   └── perguntas_gabarito_test.json   (Dataset de Teste - Prova Final)
│
├── genetic_lab/                   # Pacote Principal (Código Fonte)
│   ├── __main__.py                # Ponto de entrada (Motor do AG)
│   ├── config.py                  # Configurações (Modelo, Hyperparâmetros)
│   ├── dataset.py                 # Gerenciador de dados
│   ├── llm_local.py               # Interface com Ollama
│   ├── utils.py                   # Normalização e cálculo de similaridade
│   └── genetic_algorithm/         # Módulo do DEAP
│       ├── toolbox.py             # Definição do DNA (Gene Pool)
│       └── evaluation.py          # Função de Fitness e Debug
│
└── README.md