"""Test suite for Hatchyverse chatbot queries."""

import os
import sys
import unittest
from pathlib import Path
import logging
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader
from src.models.enhanced_chatbot import EnhancedChatbot
from src.models.relationship_extractor import RelationshipRegistry
from src.models.contextual_retriever import ContextualRetriever

class TestHatchyverseChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize core components
        cls.knowledge_graph = HatchyKnowledgeGraph()
        cls.data_loader = EnhancedDataLoader(cls.knowledge_graph)
        
        # Set up data directory
        cls.data_dir = Path("data")
        cls.data_loader.set_data_directory(cls.data_dir)
        
        # Load all data
        cls.loaded_entities = cls.data_loader.load_all_data()
        
        # Initialize LLM (you'll need to set OPENAI_API_KEY in environment)
        cls.llm = ChatOpenAI(
            model_name="gpt-4",
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
        
        cls.embeddings = OpenAIEmbeddings()
        cls.vector_store = FAISS.from_texts(texts, cls.embeddings)
        
        # Initialize retriever
        cls.retriever = ContextualRetriever(
            cls.knowledge_graph,
            cls.vector_store
        )
        
        # Initialize chatbot
        cls.relationship_registry = RelationshipRegistry()
        cls.chatbot = EnhancedChatbot(cls.llm)
        
    def setUp(self):
        """Set up test case."""
        self.session_id = "test_session"
        
    def test_omniterra_query(self):
        """Test query about Omniterra."""
        response = self.chatbot.process_message(
            self.session_id,
            "what is Omniterra"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_gen1_count_query(self):
        """Test query about Gen1 Hatchy count."""
        response = self.chatbot.process_message(
            self.session_id,
            "How many Gen1 Hatchy are there?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_gen1_fire_hatchy_query(self):
        """Test query about Gen1 Fire Hatchy."""
        response = self.chatbot.process_message(
            self.session_id,
            "Can you list the Gen1 Fire hatchy?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_hatchy_comparison_query(self):
        """Test comparison between Firret and Firadactus."""
        response = self.chatbot.process_message(
            self.session_id,
            "How are Firret and FIradactus similar?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_ixor_query(self):
        """Test query about Ixor."""
        response = self.chatbot.process_message(
            self.session_id,
            "What can you tell me about Ixor?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_rideable_hatchy_query(self):
        """Test query about rideable Hatchy."""
        response = self.chatbot.process_message(
            self.session_id,
            "What Gen1 and Gen2 hatchy are potentially rideable?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)
        
    def test_buzzkill_armor_query(self):
        """Test query about Buzzkill armor."""
        response = self.chatbot.process_message(
            self.session_id,
            "What armor piece(s) are related to buzzkill?"
        )
        self.assertIsNotNone(response)
        self.assertIn('response', response)

if __name__ == '__main__':
    unittest.main() 