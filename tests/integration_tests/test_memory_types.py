#!/usr/bin/env python3
"""
Integration tests for different LangChain memory types with PgDistRagRetriever.
Tests all major memory implementations to ensure compatibility.
"""

import os
import uuid
import pytest
import psycopg
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_yugabytedb import PgDistRagRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory,
)

# Optional memory types that require langchain-community
try:
    from langchain.memory import ConversationEntityMemory, ConversationKGMemory
    HAS_ENTITY_MEMORY = True
except ImportError:
    HAS_ENTITY_MEMORY = False
from langchain_postgres import PGVector
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

# Test configuration
DB_CONNECTION = "postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
DB_CONN_DIRECT = "postgresql://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
INDEX_NAME = "test_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.fixture
def embeddings():
    """Create embeddings instance."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )


@pytest.fixture
def retriever(embeddings):
    """Create PgDistRagRetriever instance."""
    return PgDistRagRetriever(
        embeddings=embeddings,
        connection_string=DB_CONNECTION,
        index_name=INDEX_NAME
    )


@pytest.fixture
def llm():
    """Create LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )


class TestConversationBufferMemory:
    """Test ConversationBufferMemory - stores all messages in RAM."""
    
    def test_basic_memory(self, retriever, llm):
        """Test basic in-memory conversation buffer."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # First question
        result1 = qa_chain({"question": "What is YugabyteDB?"})
        assert "answer" in result1
        assert len(result1["source_documents"]) > 0
        
        # Follow-up question (should use memory)
        result2 = qa_chain({"question": "What are its key features?"})
        assert "answer" in result2
        
        # Verify memory has both turns
        chat_history = memory.load_memory_variables({})["chat_history"]
        assert len(chat_history) == 4  # 2 questions + 2 answers


class TestConversationBufferWindowMemory:
    """Test ConversationBufferWindowMemory - stores last K messages."""
    
    def test_windowed_memory(self, retriever, llm):
        """Test memory with sliding window of K messages."""
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=2  # Keep only last 2 exchanges (4 messages)
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Ask 3 questions
        qa_chain({"question": "What is YugabyteDB?"})
        qa_chain({"question": "What is distributed SQL?"})
        qa_chain({"question": "What about scalability?"})
        
        # Memory should only have last 2 exchanges (4 messages)
        chat_history = memory.load_memory_variables({})["chat_history"]
        assert len(chat_history) == 4  # Last 2 Q&A pairs


class TestConversationSummaryMemory:
    """Test ConversationSummaryMemory - stores LLM-generated summary."""
    
    def test_summary_memory(self, retriever, llm):
        """Test memory that summarizes conversation."""
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Multiple questions
        qa_chain({"question": "What is YugabyteDB?"})
        qa_chain({"question": "What databases does it support?"})
        
        # Should have a summary
        summary = memory.load_memory_variables({})
        assert "chat_history" in summary


class TestConversationSummaryBufferMemory:
    """Test ConversationSummaryBufferMemory - hybrid buffer+summary."""
    
    def test_summary_buffer_memory(self, retriever, llm):
        """Test memory combining buffer and summary."""
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=100  # Trigger summary after 100 tokens
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Ask questions
        qa_chain({"question": "What is YugabyteDB?"})
        qa_chain({"question": "Explain its architecture in detail."})
        
        # Memory should manage buffer/summary automatically
        memory_vars = memory.load_memory_variables({})
        assert "chat_history" in memory_vars


@pytest.mark.skipif(not HAS_ENTITY_MEMORY, reason="Requires langchain-community")
class TestConversationEntityMemory:
    """Test ConversationEntityMemory - extracts and tracks entities."""
    
    def test_entity_memory(self, retriever, llm):
        """Test memory that extracts entities from conversation."""
        memory = ConversationEntityMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Ask about specific entities
        qa_chain({"question": "What is YugabyteDB?"})
        
        # Memory should track entities
        memory_vars = memory.load_memory_variables({})
        assert "chat_history" in memory_vars
        # Entities are stored in memory.entity_store


@pytest.mark.skipif(not HAS_ENTITY_MEMORY, reason="Requires langchain-community")
class TestConversationKGMemory:
    """Test ConversationKGMemory - builds knowledge graph."""
    
    def test_knowledge_graph_memory(self, retriever, llm):
        """Test memory that builds knowledge graph."""
        memory = ConversationKGMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Build knowledge through questions
        qa_chain({"question": "What is YugabyteDB?"})
        
        # Memory builds knowledge graph internally
        memory_vars = memory.load_memory_variables({})
        assert "chat_history" in memory_vars


class TestVectorStoreRetrieverMemory:
    """Test VectorStoreRetrieverMemory - stores messages as embeddings."""
    
    def test_vector_store_memory(self, retriever, llm, embeddings):
        """Test memory that stores conversation as vectors."""
        # Create a separate vector store for memory
        memory_vectorstore = PGVector(
            embeddings=embeddings,
            connection=DB_CONNECTION,
            collection_name="chat_memory_test",
            use_jsonb=True,
        )
        
        memory = VectorStoreRetrieverMemory(
            retriever=memory_vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory_key="chat_history",
            return_messages=False,  # VectorStoreRetrieverMemory doesn't work with messages
            input_key="question",
            output_key="answer"
        )
        
        # Note: VectorStoreRetrieverMemory is NOT compatible with
        # ConversationalRetrievalChain (which expects return_messages=True)
        # This is a known limitation - use it with other chain types
        
        # Test memory storage directly
        memory.save_context(
            {"question": "What is YugabyteDB?"},
            {"answer": "YugabyteDB is a distributed SQL database."}
        )
        
        # Retrieve relevant past conversations
        relevant_history = memory.load_memory_variables(
            {"question": "Tell me about distributed databases"}
        )
        assert "chat_history" in relevant_history


class TestPostgresChatMessageHistory:
    """Test PostgresChatMessageHistory - persistent storage in PostgreSQL."""
    
    def test_postgres_persistent_memory(self, retriever, llm):
        """Test persistent chat history in PostgreSQL."""
        session_id = str(uuid.uuid4())
        table_name = "langchain_chat_history"
        
        # Create direct connection for chat history
        conn = psycopg.connect(DB_CONN_DIRECT)
        
        # Initialize chat history
        message_history = PostgresChatMessageHistory(
            table_name,
            session_id,
            sync_connection=conn
        )
        
        # Ensure table exists
        PostgresChatMessageHistory.create_tables(conn, table_name)
        
        # Create memory with persistent backend
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Ask questions
        result1 = qa_chain({"question": "What is YugabyteDB?"})
        assert "answer" in result1
        
        result2 = qa_chain({"question": "What are its benefits?"})
        assert "answer" in result2
        
        # Verify persistence by creating new memory with same session
        new_message_history = PostgresChatMessageHistory(
            table_name,
            session_id,
            sync_connection=conn
        )
        
        messages = new_message_history.messages
        assert len(messages) == 4  # 2 questions + 2 answers
        
        conn.close()




# Test runner helpers
def run_single_test(test_class_name, test_method_name):
    """Run a single test for debugging."""
    pytest.main([
        __file__,
        f"-k {test_class_name} and {test_method_name}",
        "-v",
        "-s"
    ])


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
