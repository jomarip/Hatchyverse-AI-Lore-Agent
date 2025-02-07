"""Test suite for Hatchyverse chatbot queries."""

import os
import sys
import unittest
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader
from src.models.enhanced_chatbot import EnhancedChatbot
from src.models.registry import RelationshipRegistry
from src.models.contextual_retriever import ContextualRetriever

class TestHatchyComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize core components
        cls.knowledge_graph = HatchyKnowledgeGraph()
        cls.data_loader = EnhancedDataLoader(cls.knowledge_graph)
        
        # Set up data directory
        cls.data_dir = Path("data")
        cls.data_loader.set_data_directory(cls.data_dir)
        
        # Load all data
        cls.loaded_entities = cls.data_loader.load_all_data()
        
        # Initialize LLM with OpenAI
        cls.llm = ChatOpenAI(
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-4'),
            temperature=0.7
        )
        
        # Set up vector store
        text_files = [
            "Hatchy World Comic_ Chaos saga.txt",
            "Hatchy World _ world design.txt",
            "HWCS - Simplified main arc and arc suggestions.txt",
            "Hatchyverse Eco Presentation v3.txt"
        ]
        
        # Create vector store from text files
        texts = []
        for file in text_files:
            try:
                with open(cls.data_dir / file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        
        # Initialize embeddings with the latest model
        cls.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # Using the latest embedding model
        )
        cls.vector_store = FAISS.from_texts(texts, cls.embeddings)
        
        # Initialize retriever
        cls.retriever = ContextualRetriever(
            cls.knowledge_graph,
            cls.vector_store
        )
        
        # Initialize chatbot with just the LLM
        cls.chatbot = EnhancedChatbot(cls.llm)
        
    def setUp(self):
        """Set up test case."""
        self.session_id = "test_session"
        
    def test_knowledge_graph_initialization(self):
        """Test knowledge graph initialization."""
        self.assertIsNotNone(self.knowledge_graph)
        self.assertIsInstance(self.knowledge_graph, HatchyKnowledgeGraph)
        
    def test_data_loader(self):
        """Test data loader functionality."""
        self.assertIsNotNone(self.data_loader)
        self.assertIsInstance(self.data_loader, EnhancedDataLoader)
        
    def test_retriever(self):
        """Test retriever initialization."""
        self.assertIsNotNone(self.retriever)
        self.assertIsInstance(self.retriever, ContextualRetriever)
        
    def test_chatbot(self):
        """Test chatbot initialization."""
        self.assertIsNotNone(self.chatbot)
        self.assertIsInstance(self.chatbot, EnhancedChatbot)

if __name__ == '__main__':
    unittest.main() 