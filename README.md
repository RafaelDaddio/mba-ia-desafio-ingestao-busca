# Desafio MBA Engenharia de Software com IA - Full Cycle

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave da OpenAI com acesso ao modelo configurado em `OPENAI_MODEL`

## 1) Criar ambiente virtual e instalar dependências

No diretório do projeto:

```bash
python -m venv .venv
```

### Windows (PowerShell)

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Subir PostgreSQL com pgvector

```bash
docker compose up -d
```

## 3) Configurar variáveis de ambiente

Copie o arquivo `.env.example` para `.env` na raiz do projeto.

### Windows (PowerShell)

```powershell
copy .env.example .env
```

### Linux/macOS

```bash
cp .env.example .env
```

Depois, abra o arquivo `.env` e preencha as variáveis necessárias, por exemplo:

```env
OPENAI_API_KEY=coloque_sua_chave_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-5-nano
PG_VECTOR_COLLECTION_NAME=docs
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/rag
PDF_PATH=docs/seu_arquivo.pdf
```

> Se receber erro `403` (`not allowed to sample from this model`), troque `OPENAI_MODEL` por um modelo habilitado para sua conta.

## 4) Ingerir o PDF no banco vetorial

```bash
python .\src\ingest.py
```

## 5) Executar o chat

```bash
python .\src\chat.py
```

No terminal, faça perguntas sobre o conteúdo do PDF.
Digite `sair` para encerrar.