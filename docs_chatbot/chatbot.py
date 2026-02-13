"""Documentation Chatbot Core Logic."""

import logging
import os
from typing import Dict, Any, List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_yugabytedb import PgDistRagRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom system prompt for documentation chatbot
SYSTEM_TEMPLATE = """You are a helpful documentation assistant for YugabyteDB.

Your role is to:
1. Answer questions accurately based on the provided documentation
2. Provide clear, concise explanations with examples when helpful
3. Include relevant code snippets or commands when appropriate
4. Cite sources by mentioning the document name or section
5. If you're unsure, acknowledge it and suggest where to find more information

Guidelines:
- Be precise and technical when needed, but explain complex concepts clearly
- Use formatting (bullet points, code blocks) to improve readability
- If a question is ambiguous, ask clarifying questions
- Always ground your answers in the provided documentation

Context from documentation:
{context}

Chat History:
{chat_history}"""

HUMAN_TEMPLATE = """Question: {question}

Please provide a detailed answer based on the documentation above."""


class DocumentationChatbot:
    """Documentation chatbot using LangChain and pg_dist_rag."""
    
    def __init__(self):
        """Initialize the chatbot."""
        logger.info("Initializing Documentation Chatbot...")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY='your-key'"
            )
        # Initialize components
        self.embeddings = self._create_embeddings()
        self.retriever = self._create_retriever()
        self.llm = self._create_llm()
        self.memory = self._create_memory()
        self.chain = self._create_chain()
        
        logger.info("✅ Chatbot initialized successfully!")
    
    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create embeddings model."""
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        logger.info(f"Creating embeddings: {model}")
        return OpenAIEmbeddings(model=model, dimensions=dimensions)
    
    def _create_retriever(self) -> PgDistRagRetriever:
        """Create document retriever."""
        host = os.getenv("DB_HOST", "127.0.0.1")
        port = os.getenv("DB_PORT", "5433")
        name = os.getenv("DB_NAME", "yugabyte")
        user = os.getenv("DB_USER", "yugabyte")
        password = os.getenv("DB_PASSWORD", "yugabyte")
        connection_string = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        )
        index_name = os.getenv("INDEX_NAME", "docs_index")
        k = int(os.getenv("TOP_K_DOCUMENTS", "5"))
        logger.info(f"Connecting to database: {host}:{port}")
        logger.info(f"Using index: {index_name}")
        retriever = PgDistRagRetriever(
            embeddings=self.embeddings,
            connection_string=connection_string,
            index_name=index_name,
            k=k,
        )
        return retriever
    
    def _create_llm(self) -> ChatOpenAI:
        """Create language model."""
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        logger.info(f"Creating LLM: {model}")
        return ChatOpenAI(model=model, temperature=temperature, streaming=True)
    
    def _create_memory(self) -> ConversationBufferMemory:
        """Create conversation memory."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create conversational retrieval chain."""
        
        # Create custom prompt
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )
        
        return chain
    
    def check_index_status(self) -> Dict[str, Any]:
        """Check the status of the document index."""
        try:
            status = self.retriever.check_index_ready()
            
            if isinstance(status, dict):
                return status
            elif isinstance(status, tuple):
                is_ready, status_dict = status
                return status_dict
            else:
                return {"ready": False, "error": "Unknown status format"}
        
        except Exception as e:
            logger.error(f"Error checking index status: {e}")
            return {"ready": False, "error": str(e)}
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get index configuration information."""
        try:
            return self.retriever.get_index_info()
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"error": str(e)}
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "error": None
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get answer from chain
            result = self.chain({"question": question})
            
            # Extract sources
            sources = []
            if result.get('source_documents'):
                sources = [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get('source_uri', 'Unknown'),
                        "metadata": doc.metadata
                    }
                    for doc in result['source_documents']
                ]
            
            return {
                "answer": result.get('answer', 'No answer generated.'),
                "sources": sources,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return {
                "answer": None,
                "sources": [],
                "error": str(e)
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        messages = self.memory.chat_memory.messages
        history = []
        
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            history.append({
                "role": role,
                "content": msg.content
            })
        
        return history
