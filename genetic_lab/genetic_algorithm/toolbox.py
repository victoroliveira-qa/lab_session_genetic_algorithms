"""
Configuração da "Caixa de Ferramentas" (Toolbox) do DEAP.
Aqui definimos o "DNA" dos nossos prompts.
"""
import random
from deap import base, creator, tools

# --- O "DNA" DO PROMPT ---
GENE_POOL = {
    'persona': [
        "Aja como um expert em Pandas Python.",
        "Você é um gerador de código SQL/Pandas estrito.",
        "Traduza a pergunta natural para código Python Pandas eficiente.",
        "Você é um assistente sênior de Data Science."
    ],
    'schema_info': [
        "Use estritamente as colunas deste schema:\n{schema_string}",
        "Considere que o dataframe 'df' possui as colunas:\n{schema_string}",
        "Baseie-se na estrutura da tabela abaixo:\n{schema_string}"
    ],
    'examples': [
        # Opção 0: Sem exemplos (Zero-shot)
        "",
        # Opção 1: Exemplos genéricos de soma/média
        "Exemplo: 'Total de vendas' -> df['vendas'].sum()\nExemplo: 'Média de idade' -> df['idade'].mean()",
        # Opção 2: Exemplos de filtro e agrupamento (Mais complexos)
        "Exemplo: 'Vendas em SP' -> df[df['UF']=='SP']['vendas'].sum()\nExemplo: 'Maior valor por loja' -> df.groupby('loja')['valor'].max()",
        # Opção 3: Exemplo de contagem
        "Exemplo: 'Quantos clientes?' -> df['id_cliente'].nunique()"
    ],
    'output_rule': [
        "RESPOSTA APENAS EM CÓDIGO. NUNCA use ```python ou explicações.",
        "Retorne somente a linha de comando Python. Sem markdown.",
        "Sua saída deve ser executável diretamente com eval(). Nada de texto extra."
    ],
    'error_handling': [
        "O DataFrame é 'df'. Não importe pandas.",
        "Assuma que 'df' já está carregado na memória.",
        "Trate nomes de colunas com sensibilidade a maiúsculas/minúsculas."
    ]
}

# 1. Definir os limites para nossos genes (índices)
# O AG vai escolher um índice para cada chave do dicionário
GENE_KEYS = list(GENE_POOL.keys())  # Garante a ordem: ['persona', 'schema_info', 'examples', ...]
GENE_BOUNDS = [len(GENE_POOL[key]) - 1 for key in GENE_KEYS]

# 2. Criar a "Fitness" e o "Indivíduo" no DEAP
# Se o creator já foi criado na sessão, o DEAP reclama, então usamos try/except ou checagem
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass  # Já foram criados

# 3. Registrar as ferramentas de criação
toolbox = base.Toolbox()


# Função auxiliar para gerar um gene aleatório baseado no índice da chave
def random_gene(index):
    return random.randint(0, GENE_BOUNDS[index])

toolbox.register("attr_gene", random_gene)

# Função para criar um indivíduo completo (um valor para cada chave do pool)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 lambda: toolbox.attr_gene(random.randint(0, len(GENE_BOUNDS) - 1)), n=len(GENE_BOUNDS))


def create_random_individual():
    ind = []
    for i in range(len(GENE_BOUNDS)):
        ind.append(random.randint(0, GENE_BOUNDS[i]))
    return creator.Individual(ind)


toolbox.register("individual_correct", create_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual_correct)

# 4. Registrar os Operadores Genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[0] * len(GENE_BOUNDS), up=GENE_BOUNDS, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- Funções de "Tradução" ---

def individual_to_prompt(individual, schema_string):
    """
    Converte um indivíduo do DEAP (lista de índices) em um prompt de texto.
    """
    try:
        # Mapeia os índices do indivíduo para as strings reais
        # A ordem de GENE_KEYS é crucial aqui
        genes = {}
        for i, key in enumerate(GENE_KEYS):
            genes[key] = GENE_POOL[key][individual[i]]

        # Monta o texto final
        prompt_text = (
            f"{genes['persona']}\n\n"
            f"{genes['schema_info']}\n\n"
            f"{genes['examples']}\n\n"  # <--- AQUI ESTÁ O PULO DO GATO (FEW-SHOT)
            f"Regras de Saída:\n"
            f"- {genes['output_rule']}\n"
            f"- {genes['error_handling']}"
        )

        return prompt_text.format(schema_string=schema_string)

    except IndexError:
        print(f"ERRO: Indivíduo inválido detectado (índice fora do limite): {individual}")
        return "ERRO"
    except Exception as e:
        print(f"ERRO na renderização do prompt: {e}")
        return "ERRO"