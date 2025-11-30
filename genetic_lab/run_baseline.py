# genetic_lab/run_baseline.py
"""
Script de Baseline para Comparação Científica.
Mede a performance do Llama 3 SEM o Algoritmo Genético.
"""
from . import dataset
from . import llm_local
from . import utils


def run_baseline_test():
    print("--- INICIANDO TESTE DE BASELINE (SEM AG) ---")

    # 1. Carregar Dados
    dm = dataset.DatasetManager()
    test_data = dm.get_test_data()
    schema = dm.get_schema()

    if not test_data:
        print("Erro: Sem dados de teste.")
        return

    # --- DEFINIÇÃO DOS PROMPTS DE CONTROLE ---

    # BASELINE 1: Prompt "Ingênuo" (Zero-Shot)
    # O que uma pessoa comum perguntaria.
    prompt_zero_shot = (
        "Você é um assistente de código. Escreva um código Python Pandas para responder à pergunta.\n"
        f"Considere este schema:\n{schema}\n"
        "Retorne apenas o código."
    )

    # BASELINE 2: Prompt "Manual" (Few-Shot Básico)
    # O que um programador faria manualmente.
    prompt_manual = (
        "Atue como expert em Pandas. Converta a pergunta para código.\n"
        f"Schema:\n{schema}\n"
        "Exemplo: 'Total de vendas' -> df['vendas'].sum()\n"
        "Regra: Retorne apenas o código executável, sem markdown."
    )

    # Lista de baselines para testar
    baselines = [
        ("Zero-Shot (Padrão)", prompt_zero_shot),
        ("Few-Shot (Manual)", prompt_manual)
    ]

    for nome, prompt_template in baselines:
        print(f"\n>>> Testando Baseline: {nome} <<<")
        score_total = 0
        acertos = 0

        for i, item in enumerate(test_data):
            pergunta = item['pergunta']
            gabarito = item['query_pandas']

            # Monta o prompt final
            full_prompt = f"{prompt_template}\n\nPERGUNTA: {pergunta}\nCÓDIGO:"

            # Chama o LLM
            resp = llm_local.query_local_llm(full_prompt)
            codigo = utils.clean_llm_code(resp)

            # Calcula score (usando sua normalização)
            score = utils.calculate_string_similarity(codigo, gabarito)

            score_total += score
            if score == 1.0: acertos += 1
            print(f"   Item {i}: {score}") # Descomente se quiser ver detalhe

        acuracia = (score_total / len(test_data)) * 100
        print(f"RESULTADO {nome}:")
        print(f"   Acurácia Ponderada: {acuracia:.2f}%")
        print(f"   Acertos Exatos: {acertos}/{len(test_data)}")


if __name__ == "__main__":
    run_baseline_test()