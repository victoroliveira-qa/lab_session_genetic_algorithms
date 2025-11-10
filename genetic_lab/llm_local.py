# genetic_lab/llm_local.py
"""
Interface para o LLM rodando 100% localmente via Ollama.
"""
import ollama
from . import config # <--- MUDANÇA (Adicionado import relativo)

# O nome do modelo que você baixou (ex: 'gemma:2b' ou 'llama3:8b')
MODEL_NAME = config.LLM_MODEL_NAME

def query_local_llm(prompt_text):
    """
    Envia um prompt para o LLM local (Ollama) e retorna a resposta.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text}],
            options={
                'temperature': config.LLM_TEMPERATURE
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"ERRO: Não foi possível conectar ao Ollama. Você o instalou e rodou 'ollama pull {MODEL_NAME}'?")
        print(f"Detalhe: {e}")
        return "" # Retorna vazio em caso de erro