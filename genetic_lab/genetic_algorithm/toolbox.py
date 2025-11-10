# genetic_lab/genetic_algorithm/toolbox.py
"""
Configuração da "Caixa de Ferramentas" (Toolbox) do DEAP.
Aqui definimos o "DNA" dos nossos prompts.
"""
import random
from deap import base, creator, tools

# O "DNA" do nosso AG: O que ele pode escolher?
GENE_POOL = {
    'persona': [
        "Você é um especialista sênior em Python Pandas.",
        "Você é um assistente de ciência de dados.",
        "Você é um tradutor de linguagem natural para código Pandas."
    ],
    'schema_info': [
        "O DataFrame 'df' tem o seguinte schema:\n{schema_string}",
        "Considere o seguinte schema para o DataFrame 'df':\n{schema_string}",
        ""  # Opção de não informar o schema
    ],
    'output_rule': [
        "Retorne APENAS o código Python, sem explicações.",
        "Não use markdown (```python ... ```), apenas o código puro.",
        "Responda apenas com a linha de código Pandas solicitada."
    ],
    'error_handling': [
        "O DataFrame sempre se chama 'df'.",
        "Se a pergunta for ambígua, retorne 'ERRO: Ambiguidade'.",
        "Assuma que 'df' já está carregado."
    ]
}

# 1. Definir os limites para nossos genes (índices)
GENE_BOUNDS = [len(GENE_POOL[key]) - 1 for key in GENE_POOL]
# GENE_BOUNDS será: [2, 2, 2, 2] (índices de 0 a 2 para cada gene)

# 2. Criar a "Fitness" e o "Indivíduo" no DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 3. Registrar as ferramentas de criação
toolbox = base.Toolbox()

toolbox.register("attr_gene", lambda i: random.randint(0, GENE_BOUNDS[i]))
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 lambda: toolbox.attr_gene(random.randint(0, len(GENE_BOUNDS) - 1)), n=len(GENE_BOUNDS))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 4. Registrar os Operadores Genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[0] * 4, up=GENE_BOUNDS, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)  # tournsize=3 é o padrão


# --- Funções de "Tradução" ---

def individual_to_prompt(individual, schema_string):
    """
    Converte um indivíduo do DEAP (ex: [0, 1, 2, 0]) em um prompt de texto.
    """
    try:
        genes = {
            'persona': GENE_POOL['persona'][individual[0]],
            'schema_info': GENE_POOL['schema_info'][individual[1]],
            'output_rule': GENE_POOL['output_rule'][individual[2]],
            'error_handling': GENE_POOL['error_handling'][individual[3]],
        }

        prompt_text = (
            f"{genes['persona']}\n\n"
            f"{genes['schema_info']}\n\n"
            f"Regras de Saída:\n"
            f"- {genes['output_rule']}\n"
            f"- {genes['error_handling']}"
        )

        return prompt_text.format(schema_string=schema_string)

    except IndexError:
        print(f"ERRO: Indivíduo inválido detectado: {individual}")
        return "ERRO"