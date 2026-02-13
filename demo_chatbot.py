#!/usr/bin/env python3
"""
Simple Document Q&A Chatbot Demo
Uses LangChain + YugabyteDB + pg_dist_rag
"""

import os
import uuid
import psycopg
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_yugabytedb import PgDistRagRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

DB_CONNECTION = "postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"
DB_CONN_DIRECT = "postgresql://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"  # For psycopg
INDEX_NAME = "test_index"

# Session ID for chat history (must be valid UUID)
session_id_str = os.getenv("SESSION_ID", str(uuid.uuid4()))
try:
    SESSION_ID = str(uuid.UUID(session_id_str))  # Validate and normalize UUID
except ValueError:
    SESSION_ID = str(uuid.uuid4())  # Fallback to new UUID if invalid

# PgDistRagRetriever is specifically for pg_dist_rag indices
# Uses dist_rag.documents/sources and index created by dist_rag.init_vector_index().
#
# For normal pgvector (without pg_dist_rag), use vanilla LangChain's PGVector:
#   from langchain_postgres import PGVector
#   vectorstore = PGVector(connection=DB_CONNECTION, embeddings=embeddings, collection_name="my_docs")
#   retriever = vectorstore.as_retriever()

def create_chatbot():
    """Create a conversational chatbot with memory."""
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    # Create retriever for pg_dist_rag index
    retriever = PgDistRagRetriever(
        embeddings=embeddings,
        connection_string=DB_CONNECTION,
        index_name=INDEX_NAME
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # PostgresChatMessageHistory: stores chat messages in YugabyteDB (persistent)
    # Create table and connection
    conn = psycopg.connect(DB_CONN_DIRECT)
    table_name = "langchain_chat_history"
    
    # Initialize table if not exists
    PostgresChatMessageHistory.create_tables(conn, table_name)
    
    message_history = PostgresChatMessageHistory(
        table_name,
        SESSION_ID,  # session_id  
        sync_connection=conn
    )
    
    # ConversationBufferMemory with PostgreSQL backend
    # Messages stored in YugabyteDB, not RAM
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

def chat_loop(qa_chain):
    """Interactive chat loop."""
    print("=" * 80)
    print("Document Q&A Chatbot (powered by LangChain + YugabyteDB + pg_dist_rag)")
    print("=" * 80)
    print(f"\n💾 Chat history stored in YugabyteDB (Session: {SESSION_ID[:8]}...)")
    print("\nAsk questions about your documents. Type 'quit' to exit.\n")
    
    while True:
        # Get user question
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        if not question:
            continue
        
        try:
            # Get answer
            result = qa_chain({"question": question})
            
            # Display answer
            print(f"\nBot: {result['answer']}\n")
            
            # Display sources (optional)
            if result.get('source_documents'):
                print("📚 Sources:")
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    metadata = doc.metadata
                    source = metadata.get('source_uri', 'Unknown')
                    print(f"   {i}. {source}")
                print()
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    """Main function."""
    
    # Check if index is ready
    print("Checking if index is ready...")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    retriever = PgDistRagRetriever(
        embeddings=embeddings,
        connection_string=DB_CONNECTION,
        index_name=INDEX_NAME
    )
    
    status = retriever.check_index_ready()
    is_ready = status.get('ready', False) if isinstance(status, dict) else status[0]
    
    if not is_ready:
        print("⚠️  Warning: Index is not fully built yet.")
        print(f"   Status: {status}")
        print("\n   To build the index, run:")
        print(f"   SELECT dist_rag.build_index('{INDEX_NAME}');\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create and run chatbot
    print("\nInitializing chatbot...")
    qa_chain = create_chatbot()
    print("✅ Ready!\n")
    
    # Start chat loop
    chat_loop(qa_chain)

if __name__ == "__main__":
    main()
