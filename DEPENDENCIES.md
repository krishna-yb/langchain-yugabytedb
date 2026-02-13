# Dependencies Guide

## Installation

### For Production Use
```bash
pip install -r requirements.txt
```

### For Development (includes tests and demos)
```bash
pip install -r requirements-dev.txt
```

### Using pyproject.toml with pip
```bash
# Install base package
pip install -e .

# Install with demo dependencies
pip install -e ".[demo]"

# Install with test dependencies
pip install -e ".[test]"
```

---

## Package Categories

### Core LangChain Packages
| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | >=0.3.27 | Core LangChain framework |
| `langchain-core` | >=0.3.78,<1.0.0 | LangChain core abstractions |
| `langchain-text-splitters` | >=0.3.9,<1.0.0 | Text splitting utilities |
| `langchain-openai` | >=0.3.35 | OpenAI integration (LLMs, embeddings) |
| `langchain-postgres` | >=0.0.12 | PostgreSQL chat history & vector store |
| `langchain-community` | >=0.4.1 | Community integrations (Entity/KG memory) |

### Database Drivers
| Package | Version | Purpose |
|---------|---------|---------|
| `psycopg` | >=3,<4 | PostgreSQL/YugabyteDB async driver (psycopg3) |
| `psycopg-pool` | >=3.2.1,<4 | Connection pooling for psycopg |
| `psycopg2-binary` | >=2.9.9 | Older driver for PGVector compatibility |
| `asyncpg` | >=0.30.0 | Async PostgreSQL driver |

### Vector Support
| Package | Version | Purpose |
|---------|---------|---------|
| `pgvector` | >=0.2.5,<0.4 | PostgreSQL vector extension support |
| `sqlalchemy` | >=2,<3 | SQL toolkit and ORM |
| `greenlet` | >=3.2.3 | Lightweight concurrency |
| `numpy` | >=1.21,<3 | Numerical computing (vector operations) |

### Memory Types
| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | >=3.0 | Graph library for ConversationKGMemory |

### Testing
| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=9.0.2 | Testing framework |
| `pytest-asyncio` | >=0.25.3 | Async test support |

---

## Memory Types Support

### Fully Compatible with PgDistRagRetriever
✅ **ConversationBufferMemory** - RAM only, all messages
✅ **ConversationBufferWindowMemory** - RAM only, last K messages
✅ **ConversationSummaryMemory** - RAM only, LLM-generated summary
✅ **ConversationSummaryBufferMemory** - RAM only, hybrid buffer+summary
✅ **PostgresChatMessageHistory** - Persistent in YugabyteDB ⭐
✅ **VectorStoreRetrieverMemory** - Persistent vector-based memory

### Available (require custom chain integration)
⚠️ **ConversationEntityMemory** - Requires `langchain-community`, special input format
⚠️ **ConversationKGMemory** - Requires `langchain-community`, NetworkX, special input format

---

## Demo Files

### Memory Types Demo
- **File**: `demo_memory_types.py`
- **Purpose**: Demonstrates all LangChain memory types with PgDistRagRetriever
- **Run**: `python demo_memory_types.py`

### Simple Chatbot Demo
- **File**: `demo_chatbot.py`
- **Purpose**: Simple Q&A chatbot with persistent memory
- **Run**: `python demo_chatbot.py`

### Full Chatbot Application
- **Location**: `docs_chatbot/`
- **Purpose**: Complete chatbot with Streamlit UI
- **Run**: `cd docs_chatbot && streamlit run app.py`

---

## Integration Tests

### Memory Types Tests
- **File**: `tests/integration_tests/test_memory_types.py`
- **Purpose**: Test all memory types with pg_dist_rag
- **Run**: `pytest tests/integration_tests/test_memory_types.py -v`

### Unit Tests
- **Location**: `tests/unit_tests/yugabytedb_tests/`
- **Purpose**: Test PgDistRagRetriever functionality
- **Run**: `pytest tests/unit_tests/yugabytedb_tests/ -v`

---

## Connection String Format

### For PgDistRagRetriever (async)
```python
DB_CONNECTION = "postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
```

### For PostgresChatMessageHistory (sync)
```python
DB_CONN_DIRECT = "postgresql://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
```

### Key Points:
- Use `postgresql+psycopg://` for async operations (SQLAlchemy)
- Use `postgresql://` for sync operations (direct psycopg)
- `psycopg` (version 3) is required for native vector type support
- `psycopg2` doesn't support async, can't be used with async SQLAlchemy

---

## Optional: YugabyteDB Smart Driver (NOT USED)

The YugabyteDB Smart Driver (`psycopg2-yugabytedb-binary`) provides:
- Load balancing across cluster nodes
- Topology-aware routing
- High availability features

**Why we don't use it:**
- It's synchronous (based on psycopg2)
- Our code uses async SQLAlchemy (`create_async_engine`)
- Incompatible with async architecture

If you need the smart driver, refactor from async to sync:
```bash
pip install psycopg2-yugabytedb-binary
```

---

## Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (if not using defaults)
export DB_CONNECTION="postgresql+psycopg://user:pass@host:port/database"
export INDEX_NAME="your_index_name"
```

---

## Package Updates

Last updated: 2026-02-12

To update all packages:
```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:
```bash
pip list --outdated
```
