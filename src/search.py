import os
from dotenv import load_dotenv

from langchain.schema import Document, StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def similarity_search(query: str) -> list[tuple[Document, float]]:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True
    )

    results = store.similarity_search_with_score(query, k=1)
    output = []
    for i, (doc, score) in enumerate(results, start=1):
        output.append(doc.page_content)

    return "\n".join(output)


def search_prompt(question=None):
    results = similarity_search(question)

    system = ("system", PROMPT_TEMPLATE.format(contexto=results, pergunta="{question}"))
    user = ("user", "{question}")

    chat_prompt = ChatPromptTemplate.from_messages([system, user])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

    chain = chat_prompt | llm | StrOutputParser()

    return chain.invoke({"question": question})
