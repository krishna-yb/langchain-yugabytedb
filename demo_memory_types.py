#!/usr/bin/env python3
"""
Demo: Different LangChain Memory Types with PgDistRagRetriever

This script demonstrates all major memory types and their characteristics.
Run with: python demo_memory_types.py
"""

import os
import uuid
import psycopg
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_yugabytedb import PgDistRagRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
    ConversationKGMemory,
)
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

# Configuration
DB_CONNECTION = "postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
DB_CONN_DIRECT = "postgresql://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
INDEX_NAME = "test_index"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70)

def create_base_chain(memory, embeddings, llm, retriever):
    """Create a conversational chain with given memory."""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

def test_conversation_buffer_memory(embeddings, llm, retriever):
    """Test 1: ConversationBufferMemory - Stores ALL messages in RAM."""
    print_section("ConversationBufferMemory")
    print("📝 Stores: ALL conversation messages")
    print("💾 Storage: RAM only")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Short sessions, full context needed")
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = create_base_chain(memory, embeddings, llm, retriever)
    
    # Test conversation
    print("\n💬 Q1: What are some soccer dribbling drills?")
    result1 = qa_chain({"question": "What are some soccer dribbling drills?"})
    print(f"📄 Retrieved {len(result1['source_documents'])} documents from pg_dist_rag")
    if result1['source_documents']:
        print(f"   Source: {result1['source_documents'][0].metadata.get('source_name', 'N/A')}")
    print(f"✅ Answer: {result1['answer'][:250]}...")
    
    print("\n💬 Q2: What coaching points should I focus on?")
    result2 = qa_chain({"question": "What coaching points should I focus on?"})
    print(f"📄 Retrieved {len(result2['source_documents'])} documents from pg_dist_rag")
    print(f"✅ Answer: {result2['answer'][:250]}...")
    
    # Show memory state
    chat_history = memory.load_memory_variables({})["chat_history"]
    print(f"\n📊 Memory contains: {len(chat_history)} messages (all history)")
    print(f"   - Question 1: {chat_history[0].content[:50]}...")
    print(f"   - Answer 1: {chat_history[1].content[:50]}...")
    print(f"   - Question 2: {chat_history[2].content[:50]}...")
    print(f"   - Answer 2: {chat_history[3].content[:50]}...")

def test_conversation_buffer_window_memory(embeddings, llm, retriever):
    """Test 2: ConversationBufferWindowMemory - Stores last K messages."""
    print_section("ConversationBufferWindowMemory")
    print("📝 Stores: Last K conversation turns")
    print("💾 Storage: RAM only")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Long conversations, recent context only")
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=1  # Keep only last 1 exchange
    )
    
    qa_chain = create_base_chain(memory, embeddings, llm, retriever)
    
    # Test multiple questions
    print("\n💬 Q1: What are the coaching points for dribbling?")
    r1 = qa_chain({"question": "What are the coaching points for dribbling?"})
    print(f"   Retrieved {len(r1['source_documents'])} docs")
    print(f"   Answer: {r1['answer'][:150]}...")
    
    print("\n💬 Q2: What drill variations can I use?")
    r2 = qa_chain({"question": "What drill variations can I use?"})
    print(f"   Retrieved {len(r2['source_documents'])} docs")
    print(f"   Answer: {r2['answer'][:150]}...")
    
    print("\n💬 Q3: How do I add pressure to the drill?")
    r3 = qa_chain({"question": "How do I add pressure to the drill?"})
    print(f"   Retrieved {len(r3['source_documents'])} docs")
    print(f"   Answer: {r3['answer'][:150]}...")
    
    # Show memory state
    chat_history = memory.load_memory_variables({})["chat_history"]
    print(f"\n📊 Memory contains: {len(chat_history)} messages (last {memory.k} exchange only)")
    print(f"   ⚠️  Q1 and Q2 are forgotten! Only Q3 is kept in memory.")

def test_conversation_summary_memory(embeddings, llm, retriever):
    """Test 3: ConversationSummaryMemory - Stores LLM-generated summary."""
    print_section("ConversationSummaryMemory")
    print("📝 Stores: LLM-generated summary of conversation")
    print("💾 Storage: RAM only")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Long conversations, compressed context")
    
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = create_base_chain(memory, embeddings, llm, retriever)
    
    print("\n💬 Q1: What equipment is needed for soccer drills?")
    qa_chain({"question": "What equipment is needed for soccer drills?"})
    
    print("💬 Q2: How should defenders apply pressure?")
    qa_chain({"question": "How should defenders apply pressure?"})
    
    # Show summary
    memory_vars = memory.load_memory_variables({})
    print(f"\n📊 Memory type: Summary (auto-compressed by LLM)")

def test_conversation_summary_buffer_memory(embeddings, llm, retriever):
    """Test 4: ConversationSummaryBufferMemory - Hybrid buffer + summary."""
    print_section("ConversationSummaryBufferMemory")
    print("📝 Stores: Recent messages + summary of older ones")
    print("💾 Storage: RAM only")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Balance between full context and compression")
    
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=100  # Summary kicks in after 100 tokens
    )
    
    qa_chain = create_base_chain(memory, embeddings, llm, retriever)
    
    print("\n💬 Q1: What are the key skills for dribbling?")
    qa_chain({"question": "What are the key skills for dribbling?"})
    
    print("💬 Q2: What turning techniques should players use?")
    qa_chain({"question": "What turning techniques should players use?"})
    
    print(f"\n📊 Memory: Automatically manages buffer/summary based on token limit")

def test_conversation_entity_memory(embeddings, llm, retriever):
    """Test 5: ConversationEntityMemory - Tracks entities in conversation."""
    print_section("ConversationEntityMemory")
    print("📝 Stores: Extracts and tracks entities (people, places, concepts)")
    print("💾 Storage: RAM only (entity_store dict)")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Conversations with many entities to track")
    
    print("\n⚠️  Note: EntityMemory requires special input format with 'entities'")
    print("   Not compatible with ConversationalRetrievalChain by default")
    print("   Works best with simple LLMChain or custom chains")
    print("\n📊 How it works:")
    print("   - LLM extracts entities from conversation")
    print("   - Stores entity summaries in entity_store (dict)")
    print("   - Retrieves relevant entities for each new query")
    print("   - Example entities: player names, drill types, concepts")
    print("\n✅ Available but requires custom chain integration")


def test_conversation_kg_memory(embeddings, llm, retriever):
    """Test 6: ConversationKGMemory - Builds knowledge graph."""
    print_section("ConversationKGMemory")
    print("📝 Stores: Knowledge graph of conversation (relationships)")
    print("💾 Storage: RAM only (NetworkX graph)")
    print("♻️  Persistence: Lost on restart")
    print("👍 Best for: Building knowledge graphs from conversations")
    
    print("\n⚠️  Note: KGMemory requires 'entities' in input")
    print("   Not compatible with ConversationalRetrievalChain by default")
    print("   Works best with custom chains")
    print("\n📊 How it works:")
    print("   - LLM extracts entities and relationships")
    print("   - Builds NetworkX knowledge graph (memory.kg)")
    print("   - Stores triplets like (entity1, relation, entity2)")
    print("   - Example: (dribbling, requires, ball_control)")
    print("   - Retrieves relevant graph context for queries")
    print("\n✅ Available but requires custom chain integration")


def test_postgres_chat_message_history(embeddings, llm, retriever):
    """Test 5: PostgresChatMessageHistory - Persistent storage in PostgreSQL."""
    print_section("PostgresChatMessageHistory (with ConversationBufferMemory)")
    print("📝 Stores: ALL messages")
    print("💾 Storage: PostgreSQL/YugabyteDB table")
    print("♻️  Persistence: ✅ Survives restarts")
    print("👍 Best for: Production apps, multi-session persistence")
    
    session_id = str(uuid.uuid4())
    table_name = "langchain_chat_history"
    
    # Create connection
    conn = psycopg.connect(DB_CONN_DIRECT)
    
    # Initialize persistent message history
    message_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=conn
    )
    
    # Ensure table exists
    PostgresChatMessageHistory.create_tables(conn, table_name)
    
    # Create memory backed by PostgreSQL
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = create_base_chain(memory, embeddings, llm, retriever)
    
    print("\n💬 Q1: What are some soccer dribbling drill instructions?")
    result1 = qa_chain({"question": "What are some soccer dribbling drill instructions?"})
    print(f"📄 Retrieved {len(result1['source_documents'])} documents")
    print(f"✅ Answer: {result1['answer'][:250]}...")
    
    print("\n💬 Q2: What variations can I add?")
    result2 = qa_chain({"question": "What variations can I add?"})
    print(f"📄 Retrieved {len(result2['source_documents'])} documents")
    print(f"✅ Answer: {result2['answer'][:250]}...")
    
    # Verify persistence
    messages = message_history.messages
    print(f"\n📊 Stored in DB: {len(messages)} messages")
    print(f"🔑 Session ID: {session_id}")
    print(f"📋 Table: {table_name}")
    print(f"\n💬 Message history in DB:")
    for i, msg in enumerate(messages, 1):
        msg_type = "👤 User" if msg.type == "human" else "🤖 AI"
        print(f"   {i}. {msg_type}: {msg.content[:80]}...")
    
    # Test persistence by creating new memory with same session
    print("\n🔄 Testing persistence: Creating new memory instance...")
    new_message_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=conn
    )
    new_messages = new_message_history.messages
    print(f"✅ Retrieved {len(new_messages)} messages from DB")
    print(f"✅ Persistence verified! Chat history survives across instances.")
    
    conn.close()

def print_summary():
    """Print comparison summary."""
    print_section("Memory Type Comparison Summary")
    
    table = """
┌────────────────────────────────┬─────────────┬─────────────┬──────────────────┐
│ Memory Type                    │ Persistent? │ Storage     │ Best For         │
├────────────────────────────────┼─────────────┼─────────────┼──────────────────┤
│ ConversationBufferMemory       │ ❌ No       │ RAM         │ Short sessions   │
│ ConversationBufferWindowMemory │ ❌ No       │ RAM         │ Recent context   │
│ ConversationSummaryMemory      │ ❌ No       │ RAM         │ Compression      │
│ ConversationSummaryBufferMemory│ ❌ No       │ RAM         │ Hybrid approach  │
│ ConversationEntityMemory       │ ❌ No       │ RAM         │ Entity tracking  │
│ ConversationKGMemory           │ ❌ No       │ RAM         │ Knowledge graph  │
│ VectorStoreRetrieverMemory     │ ✅ Yes*     │ Vector DB   │ Similarity search│
│ PostgresChatMessageHistory     │ ✅ Yes      │ PostgreSQL  │ Production apps  │
└────────────────────────────────┴─────────────┴─────────────┴──────────────────┘

* VectorStoreRetrieverMemory persists if using persistent vector store

📌 Key Takeaways:
   • Use PostgresChatMessageHistory for production (persists across restarts)
   • Use Buffer/Window for development/testing
   • Use Summary for long conversations with token limits
   • Combine BufferMemory + PostgresChatMessageHistory for best of both worlds
"""
    print(table)

def main():
    """Run all memory type demonstrations."""
    print("\n" + "=" * 70)
    print("🎯 LangChain Memory Types Demo with PgDistRagRetriever")
    print("=" * 70)
    print("\nThis demo shows how different memory types work with pg_dist_rag.")
    print("Each test will run a small conversation and show memory behavior.")
    print("\n📚 Note: Using documents from your test_index about sports/athletics")
    print("🎯 Focus: Demonstrating memory behavior (not answer quality)\n")
    
    # Initialize components
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    retriever = PgDistRagRetriever(
        embeddings=embeddings,
        connection_string=DB_CONNECTION,
        index_name=INDEX_NAME
    )
    
    # Run tests
    try:
        test_conversation_buffer_memory(embeddings, llm, retriever)
        test_conversation_buffer_window_memory(embeddings, llm, retriever)
        test_conversation_summary_memory(embeddings, llm, retriever)
        test_conversation_summary_buffer_memory(embeddings, llm, retriever)
        test_conversation_entity_memory(embeddings, llm, retriever)
        test_conversation_kg_memory(embeddings, llm, retriever)
        test_postgres_chat_message_history(embeddings, llm, retriever)
        
        # Print summary
        print_summary()
        
        print("\n✅ All memory type demonstrations completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
