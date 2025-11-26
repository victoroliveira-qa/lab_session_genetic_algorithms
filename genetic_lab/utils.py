import re

def clean_llm_code(response_text):
    if response_text is None:
        return ""
    # Remove markdown e preâmbulos comuns do Llama 3
    response_text = re.sub(r'```python\s*', '', response_text)
    response_text = re.sub(r'```', '', response_text)
    # Remove frases como "Here is the code:"
    response_text = re.sub(r'Here is the code.*?:', '', response_text, flags=re.IGNORECASE)

    # Pega apenas a primeira linha não vazia
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return ""

def calculate_string_similarity(generated, ground_truth):
    def normalize_pandas(s):
        s = s.replace(" ", "").lower()
        s = s.replace("'", '"')
        s = re.sub(r'df\.([a-z0-9_]+)', r'df["\1"]', s)
        return s

    gen_norm = normalize_pandas(generated)
    gt_norm = normalize_pandas(ground_truth)

    # Comparação exata após normalização
    if gen_norm == gt_norm:
        return 1.0

    if len(gen_norm) > 5 and (gen_norm in gt_norm or gt_norm in gen_norm):
        return 0.8

    return 0.0