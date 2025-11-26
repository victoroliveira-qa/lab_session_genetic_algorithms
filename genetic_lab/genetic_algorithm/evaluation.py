from .. import llm_local
from .. import utils
from . import toolbox as ag_toolbox

def evaluate_fitness(individual, dataset_manager, batch_size):
    """
    Função de Fitness (Treino). SÓ USA DADOS DE TREINO.
    """
    schema = dataset_manager.get_schema()
    prompt_text = ag_toolbox.individual_to_prompt(individual, schema)

    if prompt_text == "ERRO":
        return (0.0,)

    batch = dataset_manager.get_fitness_batch(batch_size)
    if not batch:
        return (0.0,)

    total_score = 0

    for item in batch:
        pergunta = item['pergunta']
        query_gabarito = item['query_pandas']

        full_llm_prompt = f"{prompt_text}\n\nPERGUNTA DO USUÁRIO:\n{pergunta}\n\nCÓDIGO PANDAS:"
        llm_response = llm_local.query_local_llm(full_llm_prompt)
        codigo_gerado = utils.clean_llm_code(llm_response)

        score = utils.calculate_string_similarity(codigo_gerado, query_gabarito)
        total_score += score

    average_fitness = total_score / len(batch)
    return (average_fitness,)


def validate_on_test_set(best_individual, dataset_manager):
    """
    Validação Final (Teste) com DEBUG DETALHADO.
    """
    print("\n--- INICIANDO VALIDAÇÃO FINAL NO CONJUNTO DE TESTE ---")

    test_data = dataset_manager.get_test_data()
    if not test_data:
        print("AVISO: Nenhum dado de teste encontrado.")
        return 0.0

    schema = dataset_manager.get_schema()
    prompt_text = ag_toolbox.individual_to_prompt(best_individual, schema)

    total_score_sum = 0
    perfect_count = 0

    for i, item in enumerate(test_data):
        pergunta = item['pergunta']
        query_gabarito = item['query_pandas']

        full_llm_prompt = f"{prompt_text}\n\nPERGUNTA DO USUÁRIO:\n{pergunta}\n\nCÓDIGO PANDAS:"

        llm_response = llm_local.query_local_llm(full_llm_prompt)
        codigo_gerado = utils.clean_llm_code(llm_response)

        score = utils.calculate_string_similarity(codigo_gerado, query_gabarito)

        # --- LÓGICA DE DEBUG ---
        if score == 1.0:
            perfect_count += 1
            print(f"  ✅ Teste {i + 1}: Acertou (1.0)")
        elif score >= 0.8:
            print(f"  ⚠️ Teste {i + 1}: Quase (0.8) - '{codigo_gerado}'")
        else:
            print(f"  ❌ Teste {i + 1}: Errou (0.0)")
            print(f"     | Pergunta: {pergunta}")
            print(f"     | Esperado: {query_gabarito}")
            print(f"     | Gerado  : {codigo_gerado}")

        total_score_sum += score

    final_weighted_accuracy = (total_score_sum / len(test_data)) * 100

    print(f"\n--- RESULTADO DA VALIDAÇÃO (TESTE) ---")
    print(f"Acertos Perfeitos: {perfect_count} de {len(test_data)}")
    print(f"Pontuação Total: {total_score_sum:.1f} de {len(test_data)}")
    print(f"Acurácia Final Ponderada: {final_weighted_accuracy:.2f}%")
    return final_weighted_accuracy