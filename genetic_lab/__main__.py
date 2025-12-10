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
import matplotlib.pyplot as plt
import time
import json
import os

BASELINE_FILE = "data/baseline_results.json"


def plot_evolution(logbook):
    """
    Plota as estatísticas de evolução (avg, max, min).
    """
    gen = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")
    min_fitness = logbook.select("min")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg_fitness, label="Fitness Médio (avg)")
    plt.plot(gen, max_fitness, label="Melhor Fitness (max)", color="green", linestyle="--")
    plt.plot(gen, min_fitness, label="Pior Fitness (min)", color="red", linestyle=":")

    plt.xlabel("Geração")
    plt.ylabel("Fitness (Acurácia)")
    plt.title("Evolução do Fitness ao Longo das Gerações")
    plt.legend(loc="best")
    plt.grid(True)

    filename = f"evolution_plot_{int(time.time())}.png"
    plt.savefig(filename)
    print(f"\n--- Gráfico da evolução salvo como '{filename}' ---")
    plt.close()


def plot_final_comparison(ag_accuracy):
    """
    Lê os baselines do JSON e plota comparando com o AG.
    """
    # 1. Tenta carregar os resultados do baseline
    if not os.path.exists(BASELINE_FILE):
        print(
            "AVISO: Arquivo de baselines não encontrado. Rode 'python -m genetic_lab.run_baseline' primeiro para gerar o comparativo.")
        return

    try:
        with open(BASELINE_FILE, 'r') as f:
            baselines = json.load(f)
    except Exception as e:
        print(f"Erro ao ler arquivo de baselines: {e}")
        return

    # 2. Prepara os dados
    labels = list(baselines.keys()) + ["GeneticPromptLab (AG)"]
    values = list(baselines.values()) + [ag_accuracy]

    # Cores: Vermelho (Zero), Azul (Few), Verde (AG)
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # 3. Plota
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom',
                 fontweight='bold', fontsize=12)

    plt.title("Comparação Final: Baselines vs. Algoritmo Genético")
    plt.ylabel("Acurácia no Teste (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    filename = "final_comparison_ag_vs_baselines.png"
    plt.savefig(filename)
    print(f"\n--- 🏆 Gráfico Comparativo Final salvo como '{filename}' ---")
    plt.close()


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
    logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"Iniciando evolução para {N_GEN} gerações...")

    pop, log = algorithms.eaSimple(pop,
                                   ag_toolbox.toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=N_GEN,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    # 1. Gera o Gráfico de Evolução (Padrão)
    plot_evolution(log)

    print("\n--- EVOLUÇÃO (TREINO) CONCLUÍDA ---")

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"Melhor Fitness (Acurácia no TREINO): {best_fitness:.4f}")

    print("\n--- MELHOR PROMPT (TREINADO) ---")
    schema = dm.get_schema()
    best_prompt_text = ag_toolbox.individual_to_prompt(best_individual, schema)
    print(best_prompt_text)

    # 2. Validação Final e Captura do Score
    final_accuracy = evaluation.validate_on_test_set(best_individual, dm)

    # 3. Gera o Gráfico Comparativo (AG vs Baselines)
    plot_final_comparison(final_accuracy)


if __name__ == "__main__":
    run_evolution_deap()