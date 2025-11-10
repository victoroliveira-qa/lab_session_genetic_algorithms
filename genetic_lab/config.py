# genetic_lab/config.py
"""
Configurações Globais para o GeneticPromptLab
"""

# --- Configuração do LLM ---
# (Não usamos mais a API do Gemini, mas deixamos aqui para referência)
# GEMINI_API_KEY = "coloque sua api key"

# (ATUALIZADO) Nome do modelo local do Ollama que você baixou
LLM_MODEL_NAME = "gemma:2b"
LLM_TEMPERATURE = 0.5

# --- Configuração do Dataset ---
# (ATUALIZADO) Caminhos agora apontam para a pasta 'data'
DATA_CSV_TRAIN = "data/dados_desenrola_train.csv"
GABARITO_JSON_TRAIN = "data/perguntas_gabarito_train.json"

DATA_CSV_TEST = "data/dados_desenrola_test.csv"
GABARITO_JSON_TEST = "data/perguntas_gabarito_test.json"

# --- Configuração do Algoritmo Genético ---
POPULATION_SIZE = 30
NUM_GENERATIONS = 20
FITNESS_BATCH_SIZE = 3
MUTATION_RATE = 0.1 # (Não é usado pelo DEAP, que usa MUTPB no main)
ELITE_SIZE = 2      # (Não é usado pelo DEAP, que usa HallOfFame)
TOURNAMENT_SIZE = 3 # (Usado pelo DEAP)