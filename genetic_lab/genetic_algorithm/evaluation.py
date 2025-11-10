# genetic_lab/genetic_algorithm/evaluation.py
"""
Funções de Fitness (Treino) e Validação (Teste).
"""
# (ATUALIZADO) Imports relativos
from .. import config  # (..) sobe um nível (para genetic_lab)
from .. import llm_local  # (..) sobe um nível
from .. import utils  # (..) sobe um nível
from . import toolbox as ag_toolbox  # (.) mesmo nível (genetic_algorithm)


def evaluate_fitness(individual, dataset_manager, batch_size):
    """
    Função de Fitness (Avaliação) para o DEAP.
    Recebe um indivíduo (ex: [0, 1, 2, 0]) e retorna seu fitness.
    SÓ USA DADOS DE TREINO.
    """
    # 1. Converte o indivíduo (lista de int) em um prompt de texto
    schema = dataset_manager.get_schema()
    prompt_text = ag_toolbox.individual_to_prompt(individual, schema)

    if prompt_text == "ERRO":
        return (0.0,)  # Fitness zero para indivíduos inválidos

    # 2. Pega um lote de dados de TREINO
    batch = dataset_manager.get_fitness_batch(batch_size)
    if not batch:
        return (0.0,)  # Fitness zero se não houver dados

    total_score = 0

    for item in batch:
        pergunta = item['pergunta']
        query_gabarito = item['query_pandas']

        # 3. Formata o prompt final (Instrução + Pergunta)
        full_llm_prompt = f"{prompt_text}\n\nPERGUNTA DO USUÁRIO:\n{pergunta}\n\nCÓDIGO PANDAS:"

        # 4. Executa a consulta ao LLM (LOCALMENTE)
        llm_response = llm_local.query_local_llm(full_llm_prompt)

        # 5. Limpa e calcula o score
        codigo_gerado = utils.clean_llm_code(llm_response)
        score = utils.calculate_string_similarity(codigo_gerado, query_gabarito)

        total_score += score

    # O fitness é a acurácia média do lote
    average_fitness = total_score / len(batch)

    # DEAP espera que o fitness seja uma tupla
    return (average_fitness,)


def validate_on_test_set(best_individual, dataset_manager):
    """
    Roda o prompt "campeão" contra o conjunto de TESTE (30%).
    """
    print("\n--- INICIANDO VALIDAÇÃO FINAL NO CONJUNTO DE TESTE ---")

    test_data = dataset_manager.get_test_data()
    if not test_data:
        print("AVISO: Nenhum dado de teste encontrado. Pulando validação final.")
        return 0.0

    # Renderiza o prompt campeão
    schema = dataset_manager.get_schema()
    prompt_text = ag_toolbox.individual_to_prompt(best_individual, schema)

    total_score = 0
    correct_count = 0

    for i, item in enumerate(test_data):
        pergunta = item['pergunta']
        query_gabarito = item['query_pandas']

        full_llm_prompt = f"{prompt_text}\n\nPERGUNTA DO USUÁRIO:\n{pergunta}\n\nCÓDIGO PANDAS:"

        llm_response = llm_local.query_local_llm(full_llm_prompt)
        codigo_gerado = utils.clean_llm_code(llm_response)

        score = utils.calculate_string_similarity(codigo_gerado, query_gabarito)
        if score == 1.0:
            correct_count += 1

        total_score += score
        print(f"  Teste {i + 1}/{len(test_data)}: Pergunta: '{pergunta[:40]}...' -> Score: {score}")

    final_accuracy = (correct_count / len(test_data)) * 100
    print(f"\n--- RESULTADO DA VALIDAÇÃO (TESTE) ---")
    print(f"Acertos: {correct_count} de {len(test_data)}")
    print(f"Acurácia Final (no Teste): {final_accuracy:.2f}%")
    return final_accuracy