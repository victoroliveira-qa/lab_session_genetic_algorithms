# genetic_lab/__main__.py
"""
O Motor do Algoritmo Genético, usando DEAP.
Este é o ponto de entrada principal, executado com 'python -m genetic_lab'
"""
from . import config
from . import dataset
from .genetic_algorithm import toolbox as ag_toolbox
from .genetic_algorithm import evaluation
from . import llm_local

from deap import algorithms, tools
import numpy as np
import matplotlib.pyplot as plt  # <--- 1. IMPORTAR MATPLOTLIB
import time  # <--- Para salvar o gráfico com nome único


def plot_evolution(logbook):
    """
    (NOVO) Plota as estatísticas de evolução (avg, max, min).
    """
    # Pega os dados do logbook
    gen = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")
    min_fitness = logbook.select("min")

    # Cria a figura do gráfico
    plt.figure(figsize=(10, 6))

    # Plota as linhas
    plt.plot(gen, avg_fitness, label="Fitness Médio (avg)")
    plt.plot(gen, max_fitness, label="Melhor Fitness (max)", color="green", linestyle="--")
    plt.plot(gen, min_fitness, label="Pior Fitness (min)", color="red", linestyle=":")

    # Configurações do gráfico
    plt.xlabel("Geração")
    plt.ylabel("Fitness (Acurácia)")
    plt.title("Evolução do Fitness ao Longo das Gerações")
    plt.legend(loc="best")  # Adiciona a legenda
    plt.grid(True)  # Adiciona grade

    # Salva o gráfico em um arquivo
    filename = f"evolution_plot_{int(time.time())}.png"
    plt.savefig(filename)
    print(f"\n--- Gráfico da evolução salvo como '{filename}' ---")


def run_evolution_deap():
    print("--- INICIANDO GENETIC-PROMPT-LAB (com DEAP + LLM Local) ---")

    dm = dataset.DatasetManager()
    if dm.dataframe_train is None or not dm.gabarito_data_train:
        print("ERRO: Falha ao carregar dados de treino. Verifique os arquivos em /data/")
        return

    print(f"AVISO: Certifique-se que o Ollama está rodando e o modelo '{llm_local.MODEL_NAME}' foi baixado.")

    ag_toolbox.toolbox.register("evaluate", evaluation.evaluate_fitness,
                                dataset_manager=dm,
                                batch_size=config.FITNESS_BATCH_SIZE)

    POP_SIZE = config.POPULATION_SIZE
    N_GEN = config.NUM_GENERATIONS
    CXPB = 0.7
    MUTPB = 0.2

    print(f"Criando população inicial de {POP_SIZE} indivíduos...")
    pop = ag_toolbox.toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)

    # (MUDANÇA) Nós precisamos de um 'Logbook' para registrar os dados
    logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"Iniciando evolução para {N_GEN} gerações...")

    # (MUDANÇA) 'algorithms.eaSimple' retorna a população final e o logbook
    pop, log = algorithms.eaSimple(pop,
                                   ag_toolbox.toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=N_GEN,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    # (MUDANÇA) Passa o logbook para nossa nova função de plot
    plot_evolution(log)

    print("\n--- EVOLUÇÃO (TREINO) CONCLUÍDA ---")

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"Melhor Fitness (Acurácia no TREINO): {best_fitness:.4f}")
    print(f"Melhor Indivíduo (cromossomo): {best_individual}")

    print("\n--- MELHOR PROMPT (TREINADO) ---")
    schema = dm.get_schema()
    best_prompt_text = ag_toolbox.individual_to_prompt(best_individual, schema)
    print(best_prompt_text)

    # Etapa de Validação Final (no conjunto de TESTE)
    evaluation.validate_on_test_set(best_individual, dm)


if __name__ == "__main__":
    run_evolution_deap()