from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import text

from langchain_yugabytedb.yb_engine import YBEngine

logger = logging.getLogger(__name__)


def _quote_identifier(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _qualified_name(schema: str, table: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"


class PgDistRagRetriever(BaseRetriever):
    """Retriever for pg_dist_rag indices on YugabyteDB.

    Use this when your data is in pg_dist_rag (index created via dist_rag.init_vector_index).
    For a normal pgvector table without pg_dist_rag, use vanilla LangChain (e.g. PGVector.as_retriever()).

    Args:
        connection_string: PostgreSQL connection string.
        index_name: Name of the vector index (must match dist_rag.init_vector_index).
        embeddings: LangChain embeddings (model/dimensions must match the index).
        k: Number of documents to return (default: 4).
        index_schema: Schema of the index table (default: "public").
        documents_schema: Schema for dist_rag tables (default: "dist_rag").
        engine: Optional YBEngine (if set, connection_string is ignored).

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from langchain_yugabytedb import PgDistRagRetriever
        >>> retriever = PgDistRagRetriever(
        ...     connection_string="postgresql+psycopg://yugabyte:@localhost:5433/yugabyte",
        ...     index_name="my_index",
        ...     embeddings=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536),
        ...     k=5,
        ... )
        >>> docs = retriever.get_relevant_documents("What is YugabyteDB?")
    """

    def __init__(
        self,
        *,
        connection_string: Optional[str],
        index_name: str,
        embeddings: Embeddings,
        k: int = 4,
        index_schema: str = "public",
        documents_schema: str = "dist_rag",
        documents_table: str = "documents",
        sources_schema: str = "dist_rag",
        sources_table: str = "sources",
        chunk_text_column: str = "chunk_text",
        embedding_column: str = "embeddings",
        document_id_column: str = "document_id",
        metadata_filters_column: str = "metadata_filters",
        metadata_column: str = "metadata",
        documents_id_column: str = "document_id",
        documents_source_id_column: str = "source_id",
        documents_metadata_column: str = "metadata",
        sources_id_column: str = "id",
        sources_metadata_column: str = "metadata",
        sources_name_column: str = "name",
        engine: Optional[YBEngine] = None,
    ) -> None:
        super().__init__()
        if engine is None and not connection_string:
            raise ValueError(
                "Either 'connection_string' or 'engine' must be provided. "
                "Example: connection_string='postgresql+psycopg://yugabyte:@localhost:5433/yugabyte'"
            )
        
        if not index_name or not index_name.strip():
            raise ValueError(
                "index_name is required and cannot be empty. "
                "This should match the index created via dist_rag.init_vector_index()"
            )
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._engine = engine or YBEngine.from_connection_string(connection_string)
        self._embeddings = embeddings
        self._index_name = index_name
        self._k = k
        self._index_schema = index_schema
        self._documents_schema = documents_schema
        self._documents_table = documents_table
        self._sources_schema = sources_schema
        self._sources_table = sources_table
        self._chunk_text_column = chunk_text_column
        self._embedding_column = embedding_column
        self._document_id_column = document_id_column
        self._metadata_filters_column = metadata_filters_column
        self._metadata_column = metadata_column
        self._documents_id_column = documents_id_column
        self._documents_source_id_column = documents_source_id_column
        self._documents_metadata_column = documents_metadata_column
        self._sources_id_column = sources_id_column
        self._sources_metadata_column = sources_metadata_column
        self._sources_name_column = sources_name_column

    def _validate_connection(self) -> bool:
        """
        Validate database connection and pg_dist_rag extension.
        
        Returns:
            True if connection and extension are valid
            
        Raises:
            RuntimeError: If connection fails or extension is missing
        """
        try:
            # Test connection
            sql = text("SELECT 1")
            result = self._engine._run_as_sync(self._aquery_raw(sql))
            
            if not result:
                raise RuntimeError("Database connection test failed")
            
            # Check if pg_dist_rag extension exists
            sql = text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_dist_rag'
                ) AS extension_exists
            """)
            result = self._engine._run_as_sync(self._aquery_raw(sql))
            row = list(result)[0]
            
            if not row['extension_exists']:
                raise RuntimeError(
                    "pg_dist_rag extension not found. "
                    "Please ensure YugabyteDB is started with pg_dist_rag enabled."
                )
            
            logger.info("Database connection and extension validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            raise RuntimeError(f"Failed to validate database connection: {e}") from e

    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """
        Query index metadata from dist_rag.vector_indexes table.
        
        Returns:
            Dictionary with index configuration including:
            - index_name: Name of the index
            - ai_provider: AI provider (e.g., 'OPENAI')
            - embedding_model_params: Model configuration (model, dimensions)
            - index_creation_status: Status of index creation
            - id: UUID of the index
            
        Raises:
            ValueError: If index is not found
        """
        sql = text(f"""
            SELECT 
                id,
                index_name,
                ai_provider,
                embedding_model_params,
                index_creation_status,
                index_options
            FROM {_qualified_name(self._documents_schema, "vector_indexes")}
            WHERE index_name = :index_name
        """)
        
        try:
            result = self._engine._run_as_sync(
                self._aquery_raw(sql, {"index_name": self._index_name})
            )
            rows = list(result)
            
            if not rows:
                raise ValueError(
                    f"Index '{self._index_name}' not found in dist_rag.vector_indexes. "
                    f"Please create the index using dist_rag.init_vector_index() first."
                )
            
            row = rows[0]
            return {
                "id": str(row["id"]),
                "index_name": row["index_name"],
                "ai_provider": row["ai_provider"],
                "embedding_model_params": row["embedding_model_params"],
                "index_creation_status": row["index_creation_status"],
                "index_options": row["index_options"],
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch index info: {e}")
            raise RuntimeError(f"Failed to query index metadata: {e}") from e

    def validate_embedding_config(self) -> bool:
        """
        Validate that the embedding configuration matches the index configuration.
        
        Compares:
        - Model name (if available from embeddings object)
        - Vector dimensions
        
        Returns:
            True if configuration matches
            
        Raises:
            ValueError: If configuration doesn't match
        """
        index_info = self.get_index_info()
        embedding_params = index_info["embedding_model_params"]
        
        # Check dimensions match
        expected_dimensions = embedding_params.get("dimensions")
        if expected_dimensions:
            # Generate a test embedding to check dimensions
            try:
                test_embedding = self._embeddings.embed_query("test")
                actual_dimensions = len(test_embedding)
                
                if actual_dimensions != expected_dimensions:
                    raise ValueError(
                        f"Embedding dimension mismatch! "
                        f"Index expects {expected_dimensions} dimensions, "
                        f"but embeddings model produces {actual_dimensions} dimensions. "
                        f"Index model config: {embedding_params}"
                    )
            except Exception as e:
                logger.warning(f"Could not validate dimensions: {e}")
        
        # Check model name if available (OpenAI embeddings have model attribute)
        expected_model = embedding_params.get("model")
        if expected_model and hasattr(self._embeddings, "model"):
            actual_model = getattr(self._embeddings, "model")
            if actual_model != expected_model:
                logger.warning(
                    f"Model name mismatch! "
                    f"Index uses '{expected_model}', "
                    f"but embeddings object uses '{actual_model}'. "
                    f"This may lead to poor search results."
                )
        
        logger.info(
            f"Embedding configuration validated successfully. "
            f"Model: {embedding_params.get('model')}, "
            f"Dimensions: {expected_dimensions}"
        )
        return True

    def check_index_ready(self) -> Dict[str, Any]:
        """
        Check if the index is built and ready for queries.
        
        Returns:
            Dictionary with status information:
            - ready: Boolean indicating if index is ready
            - status: Index creation status
            - total_documents: Total number of documents
            - completed_documents: Number of completed documents
            - failed_documents: Number of failed documents
            - processing_documents: Number of documents still processing
        """
        index_info = self.get_index_info()
        index_id = index_info["id"]
        
        # Get document processing status
        sql = text(f"""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed_documents,
                COUNT(*) FILTER (WHERE status = 'FAILED') as failed_documents,
                COUNT(*) FILTER (WHERE status IN ('QUEUED', 'PROCESSING')) as processing_documents
            FROM {_qualified_name(self._documents_schema, "documents")} d
            WHERE d.source_id IN (
                SELECT source_id 
                FROM {_qualified_name(self._documents_schema, "vector_index_source_mappings")}
                WHERE index_id = :index_id
            )
        """)
        
        try:
            result = self._engine._run_as_sync(
                self._aquery_raw(sql, {"index_id": index_id})
            )
            row = list(result)[0]
            
            total = row["total_documents"] or 0
            completed = row["completed_documents"] or 0
            
            is_ready = (
                total > 0 and 
                completed == total and 
                index_info["index_creation_status"] != "NOT_STARTED"
            )
            
            status_info = {
                "ready": is_ready,
                "status": index_info["index_creation_status"],
                "total_documents": total,
                "completed_documents": completed,
                "failed_documents": row["failed_documents"] or 0,
                "processing_documents": row["processing_documents"] or 0,
            }
            
            if not is_ready:
                logger.warning(
                    f"Index '{self._index_name}' is not ready. "
                    f"Status: {status_info}"
                )
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to check index readiness: {e}")
            raise RuntimeError(f"Failed to query index status: {e}") from e

    def list_available_indexes(self) -> list[Dict[str, Any]]:
        """
        List all available vector indexes in pg_dist_rag.
        
        Returns:
            List of dictionaries, each containing:
            - index_name: Name of the index
            - ai_provider: AI provider used
            - embedding_model_params: Model configuration
            - index_creation_status: Build status
            - source_count: Number of sources linked to this index
            
        """
        sql = text(f"""
            SELECT 
                vi.index_name,
                vi.ai_provider,
                vi.embedding_model_params,
                vi.index_creation_status,
                COUNT(DISTINCT vism.source_id) as source_count
            FROM {_qualified_name(self._documents_schema, "vector_indexes")} vi
            LEFT JOIN {_qualified_name(self._documents_schema, "vector_index_source_mappings")} vism
                ON vi.id = vism.index_id
            GROUP BY vi.index_name, vi.ai_provider, vi.embedding_model_params, vi.index_creation_status
            ORDER BY vi.index_name
        """)
        
        try:
            result = self._engine._run_as_sync(self._aquery_raw(sql))
            indexes = []
            
            for row in result:
                indexes.append({
                    "index_name": row["index_name"],
                    "ai_provider": row["ai_provider"],
                    "embedding_model_params": row["embedding_model_params"],
                    "index_creation_status": row["index_creation_status"],
                    "source_count": row["source_count"] or 0,
                })
            
            logger.info(f"Found {len(indexes)} available indexes")
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise RuntimeError(f"Failed to query available indexes: {e}") from e

    @staticmethod
    def get_embedding_config_for_index(
        connection_string: str,
        index_name: str,
        engine: Optional[YBEngine] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve embedding configuration for a specific index.
        
        This is a helper method to get the embedding model parameters that were
        used when creating the index. Use this to ensure your LangChain embeddings
        object matches the index configuration.
        
        Args:
            connection_string: PostgreSQL connection string
            index_name: Name of the index to query
            engine: Optional YBEngine instance
            
        Returns:
            Dictionary with embedding configuration:
            - model: Embedding model name (e.g., "text-embedding-3-large")
            - dimensions: Vector dimensions (e.g., 1536)
            - ai_provider: Provider name (e.g., "OPENAI")
            
        Raises:
            ValueError: If index not found
            RuntimeError: If query fails
            
        Example:
            >>> config = PgDistRagRetriever.get_embedding_config_for_index(
            ...     connection_string="postgresql+psycopg://yugabyte:@localhost:5433/yugabyte",
            ...     index_name="my_index"
            ... )
            >>> print(config)
            {'model': 'text-embedding-3-large', 'dimensions': 1536, 'ai_provider': 'OPENAI'}
        """
        if engine is None and not connection_string:
            raise ValueError("Provide connection_string or engine")
        
        temp_engine = engine or YBEngine.from_connection_string(connection_string)
        
        sql = text("""
            SELECT 
                ai_provider,
                embedding_model_params
            FROM dist_rag.vector_indexes
            WHERE index_name = :index_name
        """)
        
        try:
            async def _fetch():
                async with temp_engine._pool.connect() as conn:
                    result = await conn.execute(sql, {"index_name": index_name})
                    return result.mappings().all()
            
            result = temp_engine._run_as_sync(_fetch())
            rows = list(result)
            
            if not rows:
                raise ValueError(
                    f"Index '{index_name}' not found in dist_rag.vector_indexes. "
                    f"Available indexes can be listed using list_available_indexes()."
                )
            
            row = rows[0]
            params = row["embedding_model_params"]
            
            return {
                "ai_provider": row["ai_provider"],
                "model": params.get("model"),
                "dimensions": params.get("dimensions"),
                "full_params": params,
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch embedding config: {e}")
            raise RuntimeError(f"Failed to query index configuration: {e}") from e

    async def _aquery_raw(
        self, sql: text, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a raw SQL query asynchronously."""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(sql, params or {})
            return result.mappings().all()

    def _build_query(self) -> str:
        index_table = _qualified_name(self._index_schema, self._index_name)
        chunk_text_col = _quote_identifier(self._chunk_text_column)
        document_id_col = _quote_identifier(self._document_id_column)
        embedding_col = _quote_identifier(self._embedding_column)

        # Full query with pg_dist_rag tables (documents and sources).
        # dist_rag.documents has no "metadata" column; use document_name, document_uri, etc.
        documents_table = _qualified_name(self._documents_schema, self._documents_table)
        sources_table = _qualified_name(self._sources_schema, self._sources_table)
        metadata_filters_col = _quote_identifier(self._metadata_filters_column)
        documents_source_id_col = _quote_identifier(self._documents_source_id_column)
        documents_id_col = _quote_identifier(self._documents_id_column)
        sources_metadata_col = _quote_identifier(self._sources_metadata_column)
        sources_id_col = _quote_identifier(self._sources_id_column)
        # dist_rag.sources has source_uri and metadata, not "name"
        sources_uri_col = _quote_identifier("source_uri")

        return f"""
            SELECT
                idx.{chunk_text_col} AS chunk_text,
                idx.{document_id_col} AS document_id,
                idx.{metadata_filters_col} AS metadata_filters,
                jsonb_build_object(
                    'document_name', d.document_name,
                    'document_uri', d.document_uri,
                    'document_checksum', d.document_checksum,
                    'status', d.status
                ) AS document_metadata,
                d.{documents_source_id_col} AS source_id,
                s.{sources_metadata_col} AS source_metadata,
                s.{sources_uri_col} AS source_name
            FROM {index_table} AS idx
            LEFT JOIN {documents_table} AS d
                ON idx.{document_id_col} = d.{documents_id_col}
            LEFT JOIN {sources_table} AS s
                ON d.{documents_source_id_col} = s.{sources_id_col}
            ORDER BY idx.{embedding_col} <=> CAST(:query_vector AS vector)
            LIMIT :k
        """

    async def _aquery(self, query_vector: list[float]) -> Iterable[dict[str, Any]]:
        """Execute vector similarity query asynchronously."""
        sql = text(self._build_query())
        try:
            async with self._engine._pool.connect() as conn:
                result = await conn.execute(
                    sql,
                    {"query_vector": query_vector, "k": self._k},
                )
                return result.mappings().all()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            
            # Provide helpful error messages
            error_msg = str(e).lower()
            if "relation" in error_msg and "does not exist" in error_msg:
                raise RuntimeError(
                    f"Index table '{self._index_name}' not found. "
                    f"Please ensure:\n"
                    f"1. The index was created via dist_rag.init_vector_index()\n"
                    f"2. The index_name parameter matches the created index\n"
                    f"3. The index_schema parameter is correct (current: '{self._index_schema}')\n"
                    f"Original error: {e}"
                ) from e
            elif "operator does not exist" in error_msg and "<=>" in error_msg:
                raise RuntimeError(
                    f"Vector extension not found. "
                    f"Please ensure the 'vector' extension is installed:\n"
                    f"  CREATE EXTENSION IF NOT EXISTS vector;\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise RuntimeError(f"Query failed: {e}") from e

    def _query(self, query_vector: list[float]) -> Iterable[dict[str, Any]]:
        """Execute vector similarity query synchronously."""
        return self._engine._run_as_sync(self._aquery(query_vector))

    def _rows_to_documents(self, rows: Iterable[dict[str, Any]]) -> list[Document]:
        documents: list[Document] = []
        for row in rows:
            # Include all pg_dist_rag metadata fields (source_name is source_uri in schema)
            source_name = row.get("source_name")
            metadata = {
                "document_id": row.get("document_id"),
                "metadata_filters": row.get("metadata_filters"),
                "document_metadata": row.get("document_metadata"),
                "source_id": row.get("source_id"),
                "source_metadata": row.get("source_metadata"),
                "source_name": source_name,
                "source_uri": source_name,  # same value; demo expects source_uri
            }
            
            documents.append(
                Document(page_content=row.get("chunk_text", ""), metadata=metadata)
            )
        return documents

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string to search for
            
        Returns:
            List of Document objects with page_content and metadata
            
        Raises:
            RuntimeError: If query execution fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        
        try:
            logger.debug(f"Generating embeddings for query: {query[:100]}...")
            query_vector = self._embeddings.embed_query(query)
            
            logger.debug(
                f"Query vector generated with {len(query_vector)} dimensions, "
                f"executing similarity search..."
            )
            
            rows = self._query(query_vector)
            documents = self._rows_to_documents(rows)
            
            logger.info(
                f"Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents for query '{query[:50]}...': {e}")
            raise RuntimeError(
                f"Document retrieval failed. Please check:\n"
                f"1. Index '{self._index_name}' exists and is built\n"
                f"2. Embedding model matches index configuration\n"
                f"3. Database connection is active\n"
                f"Error: {e}"
            ) from e

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for a query asynchronously.
        
        Args:
            query: The query string to search for
            
        Returns:
            List of Document objects with page_content and metadata
            
        Raises:
            RuntimeError: If query execution fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        
        try:
            logger.debug(f"Generating embeddings for query: {query[:100]}...")
            query_vector = self._embeddings.embed_query(query)
            
            logger.debug(
                f"Query vector generated with {len(query_vector)} dimensions, "
                f"executing similarity search..."
            )
            
            rows = await self._aquery(query_vector)
            documents = self._rows_to_documents(rows)
            
            logger.info(
                f"Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents for query '{query[:50]}...': {e}")
            raise RuntimeError(
                f"Document retrieval failed. Please check:\n"
                f"1. Index '{self._index_name}' exists and is built\n"
                f"2. Embedding model matches index configuration\n"
                f"3. Database connection is active\n"
                f"Error: {e}"
            ) from e
