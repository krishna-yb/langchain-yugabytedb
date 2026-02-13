# langchain-yugabytedb

The `langchain-yugabytedb` package implementations of core LangChain abstractions using `YugabyteDB` Distributed SQL Database.

The package is released under the MIT license.

Feel free to use the abstraction as provided or else modify them / extend them as appropriate for your own application.

## Requirements

The package supports the [asyncpg](https://github.com/MagicStack/asyncpg) and [psycopg3](https://www.psycopg.org/psycopg3/) drivers.

## Installation

```bash
pip install -U langchain-yugabytedb
```

### Documentation

* [Quickstart](https://docs.yugabyte.com/preview/explore/ysql-language-features/pg-extensions/extension-pgvector/#hnsw)

### Example

```python
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_yugabytedb import YBEngine, YugabyteDBVectorStore

# Replace the connection string with your own YugabyteDB connection string
CONNECTION_STRING = "postgresql+psycopg://yugabyte:@localhost:5433/yugabyte"
engine = YBEngine.from_connection_string(url=CONNECTION_STRING)

# Replace the vector size with your own vector size
VECTOR_SIZE = 768
embedding = DeterministicFakeEmbedding(size=VECTOR_SIZE)

TABLE_NAME = "my_doc_collection"

engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
)

store = YugabyteDBVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embedding,
)

docs = [
    Document(page_content="Apples and oranges"),
    Document(page_content="Cars and airplanes"),
    Document(page_content="Train")
]

store.add_documents(docs)

query = "I'd like a fruit."
docs = store.similarity_search(query)
print(docs)
```

> [!TIP]
> All synchronous functions have corresponding asynchronous functions

## PgDistRagRetriever

Use the `PgDistRagRetriever` for vector similarity search with YugabyteDB.
It supports two modes:

### Mode 1: With pg_dist_rag Extension (Default)

**🔄 Workflow Overview:**

```
1. Start pg_dist_rag           ← YugabyteDB with RAG extension
          ↓
2. Create vector index         ← pg_dist_rag processes documents from S3
          ↓
3. Build index                 ← Generates embeddings, stores in YugabyteDB
          ↓
4. Use LangChain on top! ✨    ← Query with PgDistRagRetriever
```

This mode requires the `pg_dist_rag` extension to be running **first**. LangChain then 
queries the vector indices created by pg_dist_rag, joining with `dist_rag.documents` 
and `dist_rag.sources` tables for rich metadata.

#### STEP 1: Start pg_dist_rag (do this first!)

Start YugabyteDB with the RAG service enabled:

```bash
# Set required environment variables
export AWS_S3_BUCKET_NAME="your-bucket-name"
export AWS_REGION="us-east-2"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export OPENAI_API_KEY="your-openai-key"

# Start yugabyted with RAG service enabled
./bin/yugabyted start \
  --base_dir /tmp/ybd_rag_test \
  --advertise_address 127.0.0.1 \
  --tserver_flags="enable_dist_rag_service=true,dist_rag_conf_csv={AWS_S3_BUCKET_NAME=${AWS_S3_BUCKET_NAME},AWS_REGION=${AWS_REGION},AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID},AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},OPENAI_API_KEY=${OPENAI_API_KEY}}"
```

**Expected output:**
```
Setting up Distributed RAG environment...
✅ Created extension vector
✅ Created extension pg_dist_rag
✅ Distributed RAG environment ready
🚀 YugabyteDB started successfully!
```

#### STEP 2: Create and build your vector index in pg_dist_rag
```sql
-- Connect to the database
./bin/ysqlsh -h 127.0.0.1 -U yugabyte -d yugabyte

-- Create source and initialize index
DO $$
DECLARE
    v_source_id UUID;
    v_index_id UUID;
BEGIN
    v_source_id := dist_rag.create_source('s3://your-bucket/documents/');
    v_index_id := dist_rag.init_vector_index(
        r_index_name := 'my_index',
        r_sources := ARRAY[v_source_id]::UUID[],
        r_ai_provider := 'OPENAI',
        r_embedding_model_params := '{"model": "text-embedding-3-large", "dimensions": 1536}',
        r_chunk_params := '{"chunk_size": 1024, "chunk_overlap": 256}'
    );
END $$;

-- Build the index (processes documents and generates embeddings)
SELECT dist_rag.build_index(r_index_name := 'my_index');

-- Monitor progress (wait until documents are processed)
SELECT index_name, document_name, document_status, chunks_processed
FROM dist_rag.vector_index_pipeline_details
WHERE index_name = 'my_index';

-- Wait until you see document_status = 'COMPLETED' for your documents
```

#### STEP 3: Now use LangChain on top of pg_dist_rag! 🚀

Once pg_dist_rag is running and your index is built, you can use LangChain's 
`PgDistRagRetriever` to query it:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_yugabytedb import PgDistRagRetriever

# Create retriever pointing to your pg_dist_rag index
retriever = PgDistRagRetriever(
    connection_string="postgresql+psycopg://yugabyte:@localhost:5433/yugabyte",
    index_name="my_index",  # Must match the index name from Step 2
    embeddings=OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536
    ),
    k=5,  # Number of chunks to retrieve
)

# Use with RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

result = qa("What are the best drills for improving dribbling?")
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('source_name', 'Unknown')}: {doc.page_content[:100]}...")
```

**Complete LangChain example (pg_dist_rag must be running!):**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_yugabytedb import PgDistRagRetriever

# ⚠️ Prerequisites: 
# - YugabyteDB with pg_dist_rag extension is running (Step 1)
# - Vector index "soccer_drills_test" is built and documents processed (Step 2)

# Now connect LangChain to your pg_dist_rag index
retriever = PgDistRagRetriever(
    connection_string="postgresql+psycopg://yugabyte:@localhost:5433/yugabyte",
    index_name="soccer_drills_test",  # Index created via dist_rag.init_vector_index
    embeddings=OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536,
        openai_api_key="your-openai-key"
    ),
    k=5,  # Retrieve top 5 most relevant chunks
)

# Create a QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key="your-openai-key"
    ),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Ask questions
questions = [
    "What are effective dribbling drills for youth players?",
    "How can I improve passing accuracy in training?",
    "What defensive positioning strategies should I teach?"
]

for question in questions:
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print('='*80)
    
    result = qa(question)
    
    print(f"\nAnswer:\n{result['result']}")
    
    print(f"\n\nSource Documents ({len(result['source_documents'])}):")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n{i}. Document ID: {doc.metadata.get('document_id')}")
        print(f"   Source: {doc.metadata.get('source_name', 'Unknown')}")
        print(f"   Chunk: {doc.page_content[:200]}...")
```

**What metadata is available?**

Each retrieved document includes rich metadata from pg_dist_rag:

```python
docs = retriever.get_relevant_documents("your query")

for doc in docs:
    print("Chunk text:", doc.page_content)
    print("Metadata:", doc.metadata)
    # {
    #   'document_id': 'uuid-of-document',
    #   'metadata_filters': {...},  # Filters from pg_dist_rag
    #   'document_metadata': {...},  # Document-level metadata
    #   'source_id': 'uuid-of-source',
    #   'source_metadata': {...},   # Source-level metadata
    #   'source_name': 's3://bucket/path'  # Original source URI
    # }
```

### For Normal pgvector (Without pg_dist_rag)

**PgDistRagRetriever is exclusively for pg_dist_rag.** If you're using standard pgvector without pg_dist_rag, use vanilla LangChain instead:

```python
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

# For normal pgvector tables (no pg_dist_rag)
vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
    collection_name="my_docs",
    connection="postgresql://yugabyte:@localhost:5433/yugabyte",
    use_jsonb=True
)

# Add documents
vectorstore.add_texts(["doc1", "doc2", "doc3"])

# Use as retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents("your query")
```

---

### When to use what?

| Feature | PgDistRagRetriever | PGVector (vanilla LangChain) |
|---------|------------------|-------------|
| **Setup** | Requires pg_dist_rag extension | Standard pgvector |
| **Data ingestion** | Automatic from S3/sources | Manual via `add_texts()` |
| **Metadata** | Rich (documents + sources) | Basic (JSONB column) |
| **Use case** | Production RAG pipelines with S3 | Simple vector search |
| **Tables** | Managed by pg_dist_rag | Collection-based |

**Recommendation:** 
- Use **PgDistRagRetriever** if you're building production RAG with pg_dist_rag extension and S3 sources
- Use **PGVector** (vanilla LangChain) for standard pgvector without pg_dist_rag

## ChatMessageHistory

The chat message history abstraction helps to persist chat message history
in a YugabyteDB table.

YugabyteDBChatMessageHistory is parameterized using a `table_name` and a `session_id`.

The `table_name` is the name of the table in the database where
the chat messages will be stored.

The `session_id` is a unique identifier for the chat session. It can be assigned
by the caller using `uuid.uuid4()`.

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_yugabytedb import YugabyteDBChatMessageHistory
import psycopg

# Establish a synchronous connection to the database
# (or use psycopg.AsyncConnection for async)
conn_info = "dbname=yugabyte user=yugabyte host=localhost port=5433" # Fill in with your connection info
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
YugabyteDBChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = YugabyteDBChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```
