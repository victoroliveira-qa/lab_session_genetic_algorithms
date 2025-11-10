# genetic_lab/utils.py
"""
Funções de ajuda para limpar e avaliar as saídas do LLM.
"""
import re


def clean_llm_code(response_text):
    """
    Limpa o código gerado pelo LLM.
    Remove markdown, explicações e espaços extras.
    """
    if response_text is None:
        return ""

    # Remove blocos de markdown
    response_text = re.sub(r'```python\s*', '', response_text)
    response_text = re.sub(r'```', '', response_text)

    # Pega apenas a primeira linha (se for multilinhas)
    response_text = response_text.split('\n')[0]

    # Remove espaços em branco extras no início/fim
    return response_text.strip()


def calculate_string_similarity(generated, ground_truth):
    """
    Função de fitness simples (acurácia exata).
    Compara as strings após normalizar espaços em branco.
    """
    # Normaliza espaços (ex: "a  b" vira "a b")
    gen_norm = " ".join(generated.split())
    gt_norm = " ".join(ground_truth.split())

    if gen_norm == gt_norm:
        return 1.0  # Acerto perfeito
    else:
        return 0.0  # Erro