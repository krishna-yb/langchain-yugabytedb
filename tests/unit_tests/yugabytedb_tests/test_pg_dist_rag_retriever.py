from __future__ import annotations

from typing import Any, Iterable
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document

from langchain_yugabytedb.retrievers import PgDistRagRetriever
from tests.unit_tests.fake_embeddings import FakeEmbeddings


class DummyEngine:
    def _run_as_sync(self, result: Iterable[dict]) -> Iterable[dict]:
        return result
    
    class _pool:
        @staticmethod
        async def connect():
            return MockConnection()


class MockConnection:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def execute(self, sql: Any, params: dict = None):
        return MockResult()


class MockResult:
    def mappings(self):
        return self
    
    def all(self):
        return []


def test_build_query_contains_expected_tables() -> None:
    retriever = PgDistRagRetriever(
        connection_string=None,
        index_name="company_docs",
        embeddings=FakeEmbeddings(),
        engine=DummyEngine(),
    )

    sql = retriever._build_query()
    assert '"company_docs"' in sql
    assert '"dist_rag"."documents"' in sql
    assert '"dist_rag"."sources"' in sql
    assert "<=>" in sql


def test_rows_to_documents_maps_metadata() -> None:
    retriever = PgDistRagRetriever(
        connection_string=None,
        index_name="company_docs",
        embeddings=FakeEmbeddings(),
        engine=DummyEngine(),
    )

    rows = [
        {
            "chunk_text": "chunk A",
            "document_id": "doc-1",
            "metadata_filters": {"team": "db"},
            "document_metadata": {"title": "Doc 1"},
            "source_id": "source-1",
            "source_metadata": {"origin": "docs"},
            "source_name": "docs-site",
        }
    ]

    docs = retriever._rows_to_documents(rows)
    assert docs == [
        Document(
            page_content="chunk A",
            metadata={
                "document_id": "doc-1",
                "metadata_filters": {"team": "db"},
                "document_metadata": {"title": "Doc 1"},
                "source_id": "source-1",
                "source_metadata": {"origin": "docs"},
                "source_name": "docs-site",
            },
        )
    ]


def test_get_relevant_documents_uses_query_result() -> None:
    retriever = PgDistRagRetriever(
        connection_string=None,
        index_name="company_docs",
        embeddings=FakeEmbeddings(),
        engine=DummyEngine(),
    )

    def fake_query(_: list[float]) -> Iterable[dict]:
        return [
            {
                "chunk_text": "chunk B",
                "document_id": "doc-2",
                "metadata_filters": None,
                "document_metadata": None,
                "source_id": None,
                "source_metadata": None,
                "source_name": None,
            }
        ]

    retriever._query = fake_query  # type: ignore[method-assign]
    docs = retriever._get_relevant_documents("what is yb-voyager?")
    assert len(docs) == 1
    assert docs[0].page_content == "chunk B"


# ========== Configuration & Validation Tests ==========


def test_init_requires_connection_string_or_engine() -> None:
    """Test that initialization fails without connection_string or engine."""
    with pytest.raises(ValueError, match="connection_string.*or.*engine"):
        PgDistRagRetriever(
            connection_string=None,
            index_name="test_index",
            embeddings=FakeEmbeddings(),
            engine=None,
        )


def test_init_requires_non_empty_index_name() -> None:
    """Test that initialization fails with empty index_name."""
    with pytest.raises(ValueError, match="index_name.*required"):
        PgDistRagRetriever(
            connection_string=None,
            index_name="",
            embeddings=FakeEmbeddings(),
            engine=DummyEngine(),
        )


def test_init_requires_positive_k() -> None:
    """Test that initialization fails with invalid k value."""
    with pytest.raises(ValueError, match="k must be positive"):
        PgDistRagRetriever(
            connection_string=None,
            index_name="test_index",
            embeddings=FakeEmbeddings(),
            engine=DummyEngine(),
            k=0,
        )


def test_skip_validation_on_init() -> None:
    """Test that validation can be skipped during initialization."""
    retriever = PgDistRagRetriever(
        connection_string=None,
        index_name="test_index",
        embeddings=FakeEmbeddings(),
        engine=DummyEngine(),
  # Skip validation
    )
    assert retriever._index_name == "test_index"


def test_get_embedding_config_requires_connection() -> None:
    """Test that get_embedding_config_for_index requires connection info."""
    with pytest.raises(ValueError, match="connection_string or engine"):
        PgDistRagRetriever.get_embedding_config_for_index(
            connection_string=None,
            index_name="test_index",
            engine=None,
        )


def test_empty_query_returns_empty_results() -> None:
    """Test that empty queries return empty results."""
    retriever = PgDistRagRetriever(
        connection_string=None,
        index_name="test_index",
        embeddings=FakeEmbeddings(),
        engine=DummyEngine(),
    )
    
    docs = retriever._get_relevant_documents("")
    assert docs == []
    
    docs = retriever._get_relevant_documents("   ")
    assert docs == []
