# YugabyteDB Documentation Chatbot

An intelligent documentation assistant powered by LangChain and pg_dist_rag.

## Features

✨ **Natural Language Q&A** - Ask questions in plain English
🧠 **Context-Aware** - Remembers conversation history
📚 **Source Citations** - Shows where answers come from
⚡ **Fast & Accurate** - Powered by vector search + LLM
🎨 **Beautiful CLI** - Rich terminal interface

## Quick Start

### 1. Install Dependencies

```bash
cd ~/code/langchain-yugabytedb/docs_chatbot
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export INDEX_NAME="docs_index"  # Optional, defaults to "docs_index"
```

### 3. Prepare Your Documentation Index

```sql
-- Connect to YugabyteDB
~/code/yugabyte-db/bin/ysqlsh -h 127.0.0.1 -p 5433 -U yugabyte

-- Create a source (S3 bucket with your docs)
SELECT dist_rag.create_source(
    's3://your-bucket/docs/',
    '{"description": "Documentation files"}'
);

-- Create vector index
SELECT dist_rag.init_vector_index(
    'docs_index',
    ARRAY(SELECT id FROM dist_rag.sources)::uuid[],
    '{"chunk_size": 500}',
    'OPENAI',
    '{"model": "text-embedding-3-small", "dimensions": 1536}'
);

-- Build the index (this may take a few minutes)
SELECT dist_rag.build_index('docs_index');

-- Check status
SELECT index_name, index_creation_status, 
       (SELECT COUNT(*) FROM dist_rag.documents WHERE index_id = v.id) as doc_count
FROM dist_rag.vector_indexes v
WHERE index_name = 'docs_index';
```

### 4. Run the Chatbot

```bash
python cli.py
```

## Usage

### Commands

- **help** - Show help message
- **status** - Check index status
- **info** - Show index configuration
- **clear** - Clear conversation history
- **history** - Show conversation history
- **quit/exit** - Exit application

### Example Questions

```
You: How do I configure connection pooling in YugabyteDB?

You: What are the best practices for schema design?

You: Explain how distributed transactions work

You: Show me an example of a YSQL query with joins
```

## Configuration

Use environment variables:

```bash
# Database Configuration
export DB_HOST="127.0.0.1"
export DB_PORT="5433"
export DB_NAME="yugabyte"
export DB_USER="yugabyte"
export DB_PASSWORD="yugabyte"

# Index Configuration
export INDEX_NAME="docs_index"

# OpenAI Configuration
export OPENAI_API_KEY="sk-..."
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMENSIONS="1536"
export LLM_MODEL="gpt-4o-mini"
export LLM_TEMPERATURE="0.7"

# Retrieval Configuration
export TOP_K_DOCUMENTS="5"
```

## Architecture

```
┌─────────────┐
│   User CLI  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Chatbot Engine     │
│  - ConversationChain│
│  - Memory           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  PgDistRagRetriever │
│  - Vector Search    │
│  - Metadata Filter  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  YugabyteDB         │
│  + pg_dist_rag      │
│  - Documents        │
│  - Embeddings       │
└─────────────────────┘
```

## Troubleshooting

### Index Not Ready

If you see "Index is not ready":

```sql
-- Build the index
SELECT dist_rag.build_index('docs_index');

-- Monitor progress (wait 1-5 minutes)
SELECT * FROM dist_rag.work_queue 
WHERE index_id = (SELECT id FROM dist_rag.vector_indexes WHERE index_name = 'docs_index');
```

### No Results

1. Check if documents are indexed:
```sql
SELECT COUNT(*) FROM dist_rag.documents;
```

2. Verify embedding model matches:
```sql
SELECT embedding_model_params FROM dist_rag.vector_indexes WHERE index_name = 'docs_index';
```

### Connection Errors

```bash
# Check YugabyteDB is running
~/code/yugabyte-db/bin/yugabyted status --base_dir /tmp/ybd_pg_dist_rag_test
```

## Advanced Features

### Custom Prompts

Edit `SYSTEM_TEMPLATE` in `chatbot.py` to customize behavior.

### Multiple Indexes

```python
# Switch between indexes
export INDEX_NAME="docs_v2"
```

### Metadata Filtering

```python
# In chatbot.py, modify retriever:
retriever = PgDistRagRetriever(
    # ... other params ...
    filter={"category": "api-docs"}  # Filter by metadata
)
```

## Performance Tips

1. **Chunk Size**: Test 200-1000 tokens for optimal results
2. **Top K**: Adjust `TOP_K_DOCUMENTS` (3-10 works well)
3. **Temperature**: Lower (0.3-0.5) for factual, higher (0.7-1.0) for creative
4. **Model**: Use `gpt-4o` for best quality, `gpt-4o-mini` for speed/cost

## Testing

```bash
# Run unit tests
cd ~/code/langchain-yugabytedb
source .venv/bin/activate
pytest tests/unit_tests/yugabytedb_tests/test_pg_dist_rag_retriever.py -v

# Run integration test
python test_retriever_live.py
```

## License

MIT

## Support

For issues or questions:
- Check YugabyteDB documentation
- Review LangChain docs
- File issues on GitHub
