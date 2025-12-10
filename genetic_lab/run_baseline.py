# genetic_lab/run_baseline.py
"""
Script de Baseline para Comparação Científica.
Mede a performance do Llama 3 SEM o Algoritmo Genético.
Gera gráficos e salva resultados para comparação futura.
"""
import json
import os
import matplotlib.pyplot as plt
from . import dataset
from . import llm_local
from . import utils

RESULTS_FILE = "data/baseline_results.json"
PLOT_FILE = "baseline_comparison.png"


def plot_baseline_results(results):
    """
    Gera um gráfico de barras comparando Zero-Shot vs Few-Shot.
    """
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#ff9999', '#66b3ff'])

    # Adiciona o valor no topo da barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom',
                 fontweight='bold')

    plt.title("Performance das Linhas de Base (Baselines)")
    plt.ylabel("Acurácia (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(PLOT_FILE)
    print(f"\n--- Gráfico de baseline salvo como '{PLOT_FILE}' ---")
    plt.close()


def save_results_to_json(results):
    """
    Salva os resultados numéricos para o AG ler depois.
    """
    # Garante que a pasta data existe
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"--- Resultados numéricos salvos em '{RESULTS_FILE}' ---")


def run_baseline_test():
    print("--- INICIANDO TESTE DE BASELINE (SEM AG) ---")

    dm = dataset.DatasetManager()
    test_data = dm.get_test_data()
    schema = dm.get_schema()

    if not test_data:
        print("Erro: Sem dados de teste.")
        return

    # Definição dos Prompts
    prompt_zero_shot = (
        "Você é um assistente de código. Escreva um código Python Pandas para responder à pergunta.\n"
        f"Considere este schema:\n{schema}\n"
        "Retorne apenas o código."
    )

    prompt_manual = (
        "Atue como expert em Pandas. Converta a pergunta para código.\n"
        f"Schema:\n{schema}\n"
        "Exemplo: 'Total de vendas' -> df['vendas'].sum()\n"
        "Regra: Retorne apenas o código executável, sem markdown."
    )

    baselines = [
        ("Zero-Shot", prompt_zero_shot),
        ("Few-Shot (Manual)", prompt_manual)
    ]

    final_results = {}

    for nome, prompt_template in baselines:
        print(f"\n>>> Testando Baseline: {nome} <<<")
        score_total = 0

        for item in test_data:
            pergunta = item['pergunta']
            gabarito = item['query_pandas']

            full_prompt = f"{prompt_template}\n\nPERGUNTA: {pergunta}\nCÓDIGO:"

            resp = llm_local.query_local_llm(full_prompt)
            codigo = utils.clean_llm_code(resp)

            score = utils.calculate_string_similarity(codigo, gabarito)
            score_total += score

        acuracia = (score_total / len(test_data)) * 100
        print(f"RESULTADO {nome}: {acuracia:.2f}%")
        final_results[nome] = acuracia

    # Salva e Plota
    save_results_to_json(final_results)
    plot_baseline_results(final_results)


if __name__ == "__main__":
    run_baseline_test()