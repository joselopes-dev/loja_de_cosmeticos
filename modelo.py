"""
Chatbot de Farmácia com Machine Learning - Versão Melhorada
Baseado na Aula 02 - PLN + NLTK
Curso Técnico em Inteligência Artificial - CETAM / IBC
"""

import os
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ============================================================
# 1. Inicialização
# ============================================================
os.system('cls' if os.name == 'nt' else 'clear')

print("=" * 70)
print("💊  CHATBOT FARMACÊUTICO COM MACHINE LEARNING (v2)")
print("=" * 70)
print("--> Carregando recursos NLTK e treinando o modelo...")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_pt = set(stopwords.words('portuguese'))

# ============================================================
# 2. Base de conhecimento da farmácia
# ============================================================
medicamentos = {
    "dipirona": {
        "preco": "R$ 8,90 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },
    "paracetamol": {
        "preco": "R$ 7,50 (750mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },
    "ibuprofeno": {
        "preco": "R$ 12,90 (400mg - 20 comprimidos)",
        "indicacao": "Dor, inflamação e febre",
        "estoque": True
    },
    "omeprazol": {
        "preco": "R$ 15,90 (20mg - 28 cápsulas)",
        "indicacao": "Azia, refluxo e proteção gástrica",
        "estoque": True
    },
    "losartana": {
        "preco": "R$ 18,50 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },
    "vitamina c": {
        "preco": "R$ 22,00 (1g - 30 comprimidos efervescentes)",
        "indicacao": "Imunidade e prevenção de gripes",
        "estoque": True
    },
    "rivotril": {
        "preco": "Medicamento controlado - necessário receita",
        "indicacao": "Ansiedade e convulsões",
        "estoque": True
    },
    "amoxicilina": {
        "preco": "R$ 28,90 (500mg - 21 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },
    "loratadina": {
        "preco": "R$ 11,90 (10mg - 12 comprimidos)",
        "indicacao": "Alergia e coriza",
        "estoque": True
    },
    "buscopan": {
        "preco": "R$ 14,50 (10mg - 20 comprimidos)",
        "indicacao": "Cólica e dor abdominal",
        "estoque": True
    }
}

# ============================================================
# 3. Dados de treinamento ampliados
# ============================================================
treinamentos = [
    # Saudação
    ("oi", "saudacao"), ("olá", "saudacao"), ("ola", "saudacao"),
    ("bom dia", "saudacao"), ("boa tarde", "saudacao"), ("boa noite", "saudacao"),
    ("tudo bem", "saudacao"), ("e aí", "saudacao"), ("fala", "saudacao"),
    ("oi tudo bem", "saudacao"), ("olá bom dia", "saudacao"),

    # Despedida
    ("tchau", "despedida"), ("até logo", "despedida"), ("até mais", "despedida"),
    ("obrigado", "despedida"), ("obrigada", "despedida"), ("valeu", "despedida"),
    ("falou", "despedida"), ("até breve", "despedida"), ("flw", "despedida"),

    # Horário
    ("qual o horário", "horario"), ("que horas abre", "horario"),
    ("que horas fecha", "horario"), ("horário de funcionamento", "horario"),
    ("vocês abrem aos domingos", "horario"), ("está aberto agora", "horario"),
    ("funciona no sábado", "horario"), ("abre domingo", "horario"),
    ("horário de hoje", "horario"), ("quando fecha", "horario"),

    # Medicamentos / Estoque
    ("vocês tem dipirona", "medicamento"), ("tem paracetamol", "medicamento"),
    ("tem ibuprofeno", "medicamento"), ("tem remédio para dor de cabeça", "medicamento"),
    ("tem antibiótico", "medicamento"), ("tem omeprazol", "medicamento"),
    ("tem losartana", "medicamento"), ("tem rivotril", "medicamento"),
    ("vocês vendem vitamina c", "medicamento"), ("tem pomada", "medicamento"),
    ("tem remédio para gripe", "medicamento"), ("tem algo para dor de garganta", "medicamento"),
    ("tem buscopan", "medicamento"), ("tem loratadina", "medicamento"),
    ("tem amoxicilina", "medicamento"), ("vocês tem esse remédio", "medicamento"),
    ("tem em estoque", "medicamento"),

    # Preços
    ("quanto custa a dipirona", "preco"), ("qual o preço do paracetamol", "preco"),
    ("quanto é o ibuprofeno", "preco"), ("preço da vitamina c", "preco"),
    ("quanto custa o omeprazol", "preco"), ("tem promoção", "preco"),
    ("está barato", "preco"), ("qual o valor", "preco"), ("quanto fica", "preco"),
    ("preço do losartana", "preco"), ("quanto custa", "preco"),

    # Sintomas
    ("estou com dor de cabeça", "sintoma"), ("tenho febre", "sintoma"),
    ("estou gripado", "sintoma"), ("dor de garganta", "sintoma"),
    ("estou com azia", "sintoma"), ("tenho pressão alta", "sintoma"),
    ("o que tomar para dor", "sintoma"), ("recomenda algo para gripe", "sintoma"),
    ("estou com tosse", "sintoma"), ("tenho alergia", "sintoma"),
    ("dor de barriga", "sintoma"), ("estou com cólica", "sintoma"),
    ("o que é bom para dor", "sintoma"), ("remédio para febre", "sintoma"),

    # Localização
    ("onde fica a farmácia", "localizacao"), ("qual o endereço", "localizacao"),
    ("vocês fazem entrega", "localizacao"), ("tem delivery", "localizacao"),
    ("como chegar", "localizacao"), ("fazem entrega", "localizacao"),
    ("onde fica", "localizacao"),

    # Pagamento
    ("aceita cartão", "pagamento"), ("pode pagar no pix", "pagamento"),
    ("aceita dinheiro", "pagamento"), ("parcelam", "pagamento"),
    ("tem máquina de cartão", "pagamento"), ("aceita pix", "pagamento"),
    ("forma de pagamento", "pagamento"),
]

# ============================================================
# 4. Pré-processamento (igual ao seu código original)
# ============================================================
def preprocessar(texto):
    texto = texto.lower()
    tokens = word_tokenize(texto, language='portuguese')
    tokens_limpos = [t for t in tokens if t.isalnum() and t not in stop_pt]
    return " ".join(tokens_limpos)

# ============================================================
# 5. Treinamento do modelo
# ============================================================
X_treino = [preprocessar(frase) for frase, _ in treinamentos]
y_treino = [intencao for _, intencao in treinamentos]

modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # unigramas + bigramas
    ('clf', MultinomialNB(alpha=0.1))
])
modelo.fit(X_treino, y_treino)

print("--> Modelo treinado com sucesso!")
print("--> Sistema pronto! Digite suas perguntas.")
print("--> Digite 'sair', 'exit' ou 'quit' para encerrar.")
print("-" * 70)

# ============================================================
# 6. Funções auxiliares
# ============================================================
def encontrar_medicamento(texto):
    """Procura se o usuário mencionou algum medicamento conhecido"""
    texto = texto.lower()
    for nome in medicamentos.keys():
        if nome in texto:
            return nome
    return None

def responder(mensagem):
    mensagem_lower = mensagem.lower()
    texto_limpo = preprocessar(mensagem)

    # 1. Verifica se mencionou um medicamento específico
    med = encontrar_medicamento(mensagem_lower)

    # 2. Classifica a intenção
    if texto_limpo:
        intencao = modelo.predict([texto_limpo])[0]
        proba = max(modelo.predict_proba([texto_limpo])[0])
        if proba < 0.28:
            intencao = "fallback"
    else:
        intencao = "fallback"

    # 3. Respostas específicas e inteligentes
    if med:
        info = medicamentos[med]
        if intencao == "preco" or "quanto" in mensagem_lower or "preço" in mensagem_lower or "custa" in mensagem_lower:
            return f"O {med.title()} custa {info['preco']}. Indicação: {info['indicacao']}."
        elif intencao == "medicamento" or "tem" in mensagem_lower:
            status = "Sim, temos em estoque!" if info['estoque'] else "No momento está em falta."
            return f"{status} {med.title()} → {info['preco']}. Indicação: {info['indicacao']}."
        else:
            return f"Sobre o {med.title()}: {info['preco']}. Indicação: {info['indicacao']}. Deseja mais informações?"

    # 4. Respostas por intenção
    if intencao == "saudacao":
        return random.choice([
            "Olá! Seja bem-vindo(a) à Farmácia Saúde Total. Como posso te ajudar hoje?",
            "Oi! É um prazer atender você. Em que posso ser útil?",
            "Olá! Sou o assistente da farmácia. Pode me contar o que você precisa?"
        ])

    elif intencao == "despedida":
        return random.choice([
            "Foi um prazer atender você! Cuide-se e volte sempre.",
            "Até logo! Qualquer coisa é só chamar. Tenha um ótimo dia!",
            "Obrigado pela preferência. Até a próxima e se cuide!"
        ])

    elif intencao == "horario":
        return "Nosso horário: Segunda a Sábado das 08h às 20h | Domingos e feriados das 09h às 13h."

    elif intencao == "medicamento":
        return ("Temos vários medicamentos disponíveis. "
                "Me diga o nome do produto (ex: dipirona, paracetamol, omeprazol...) "
                "que eu verifico o preço e a disponibilidade.")

    elif intencao == "preco":
        return ("Posso te informar o preço sim! Me diga o nome do medicamento "
                "(ex: dipirona, ibuprofeno, vitamina c...)")

    elif intencao == "sintoma":
        # Respostas mais úteis por sintoma
        if "cabeça" in mensagem_lower or "dor de cabeça" in mensagem_lower:
            return "Para dor de cabeça, costumamos indicar Dipirona (R$ 8,90) ou Paracetamol (R$ 7,50). Ambos são de venda livre."
        elif "febre" in mensagem_lower:
            return "Para febre, Dipirona ou Paracetamol são as opções mais comuns. Dipirona: R$ 8,90 | Paracetamol: R$ 7,50."
        elif "azia" in mensagem_lower or "estômago" in mensagem_lower:
            return "Para azia e proteção do estômago, o Omeprazol (R$ 15,90) é muito indicado."
        elif "gripe" in mensagem_lower or "resfriado" in mensagem_lower:
            return "Para gripe e resfriado, além de repouso, Vitamina C (R$ 22,00) e Paracetamol ajudam bastante."
        elif "alergia" in mensagem_lower or "coriza" in mensagem_lower:
            return "Para alergia e coriza, a Loratadina (R$ 11,90) é uma boa opção de venda livre."
        elif "cólica" in mensagem_lower or "barriga" in mensagem_lower:
            return "Para cólica e dor abdominal, o Buscopan (R$ 14,50) costuma aliviar bem."
        else:
            return ("Entendi o seu sintoma. Temos opções de venda livre, mas o ideal é sempre "
                    "consultar um farmacêutico ou médico. Me diga o sintoma com mais detalhes "
                    "que eu tento te indicar algo adequado.")

    elif intencao == "localizacao":
        return ("Estamos na Rua das Flores, 123 - Centro. "
                "Fazemos delivery na região (taxa a partir de R$ 5,00). "
                "Quer o número do WhatsApp para pedir?")

    elif intencao == "pagamento":
        return "Aceitamos Dinheiro, Pix, Débito e Crédito (até 3x sem juros)."

    else:  # fallback
        return random.choice([
            "Desculpe, não entendi bem. Pode me explicar de outra forma? Posso ajudar com medicamentos, preços, horários, sintomas ou localização.",
            "Não consegui identificar exatamente o que você precisa. Pode reformular? Estou aqui para ajudar com remédios, preços e orientações.",
            "Pode me dar mais detalhes? Posso informar sobre estoque, preços, horários de funcionamento e recomendações de venda livre."
        ])

# ============================================================
# 7. Loop principal
# ============================================================
while True:
    try:
        mensagem = input("\nVocê: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n💊 Bot: Encerrando o atendimento. Cuide-se!")
        break

    if not mensagem:
        print("💊 Bot: Por favor, digite alguma coisa para eu poder te ajudar.")
        print("Posso ajudar com mais alguma coisa?")
        continue

    if mensagem.lower() in ("sair", "exit", "quit"):
        print("\n💊 Bot: Obrigado por conversar conosco! Até logo e cuide-se bem!")
        break

    resposta = responder(mensagem)
    print(f"\n💊 Bot: {resposta}")
    print("Posso ajudar com mais alguma coisa?")

print("\n" + "=" * 70)
print("Atendimento finalizado. Obrigado por usar o Chatbot Farmacêutico!")
print("=" * 70)