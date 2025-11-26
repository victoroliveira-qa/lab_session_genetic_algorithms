import pandas as pd
import json
import random
from . import config


class DatasetManager:
    def __init__(self):
        # 1. Carrega o CSV de TREINO para análise
        self.dataframe_train = self._load_csv(config.DATA_CSV_TRAIN)

        # 2. Gera o "schema" (a string que o LLM verá durante o TREINO)
        self.schema_string = self._generate_schema_string(self.dataframe_train)
        print("--- Schema do DataFrame (detectado de TREINO) ---")
        print(self.schema_string)
        print("--------------------------------------------------")

        # 3. Carrega o gabarito de TREINO (perguntas/queries)
        self.gabarito_data_train = self._load_gabarito(config.GABARITO_JSON_TRAIN)
        print(f"Dataset Manager: {len(self.gabarito_data_train)} pares (pergunta, query) de TREINO carregados.")

        # 4. Carrega os dados de teste para a validação final
        self.gabarito_data_test = self._load_gabarito(config.GABARITO_JSON_TEST)
        print(f"Dataset Manager: {len(self.gabarito_data_test)} pares (pergunta, query) de TESTE carregados.")

    def _load_csv(self, path):
        """ Carrega o CSV usando pandas. """
        try:
            # (MUDANÇA) Adicionado sep=';'
            return pd.read_csv(path, encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            print(f"AVISO: Falha no UTF-8 para {path}, tentando 'latin1'.")
            # (MUDANÇA) Adicionado sep=';'
            return pd.read_csv(path, encoding='latin1', sep=';')
        except FileNotFoundError:
            print(f"ERRO: Arquivo CSV não encontrado em '{path}'.")
            return None

    def _load_gabarito(self, path):
        """ Carrega o JSON de perguntas e queries. """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERRO: Arquivo de gabarito JSON não encontrado em '{path}'.")
            return []
        except json.JSONDecodeError:
            print(f"ERRO: Arquivo JSON mal formatado em '{path}'.")
            return []

    def _generate_schema_string(self, df):
        """ Cria uma string de schema legível para o LLM. """
        if df is None:
            return "ERRO: DataFrame de treino não carregado."
        s = "Schema do DataFrame (o DataFrame se chama 'df'):\n"
        for col, dtype in df.dtypes.items():
            s += f"- {col} (tipo: {dtype})\n"
        return s

    def get_fitness_batch(self, batch_size):
        """ Retorna um lote (batch) aleatório do GABARITO DE TREINO. """
        actual_batch_size = min(batch_size, len(self.gabarito_data_train))
        if actual_batch_size == 0:
            return []
        return random.sample(self.gabarito_data_train, actual_batch_size)

    def get_schema(self):
        """ Retorna a string do schema de TREINO. """
        return self.schema_string

    def get_test_data(self):
        """ Retorna os dados de gabarito de TESTE. """
        return self.gabarito_data_test