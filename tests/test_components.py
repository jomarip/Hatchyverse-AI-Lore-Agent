import unittest
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.data_loader import EnhancedDataLoader
from src.models.enhanced_chatbot import EnhancedChatbot
from src.models.contextual_retriever import ContextualRetriever

class TestHatchyComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        load_dotenv()
        
        # Initialize components
        cls.embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        
        cls.llm = ChatOpenAI(
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-4-0125-preview'),
            temperature=0.7
        )
        
        cls.knowledge_graph = HatchyKnowledgeGraph()
        
        cls.vector_store = Chroma(
            persist_directory=os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
            embedding_function=cls.embeddings
        )
        
        cls.chatbot = EnhancedChatbot(
            llm=cls.llm,
            knowledge_graph=cls.knowledge_graph,
            vector_store=cls.vector_store
        )
        
        # Load test data
        cls.data_dir = Path("data")
        cls.loader = EnhancedDataLoader(cls.knowledge_graph)
        
        # Load a small subset of test data
        test_monster_data = {
            "name": "TestHatchy",
            "generation": "1",
            "element": "Fire",
            "evolves_from": "None",
            "habitat": "Volcano",
            "description": "A test Hatchy for unit testing"
        }
        
        # Create the Volcano location first
        volcano_data = {
            "name": "Volcano",
            "type": "location",
            "description": "A volcanic habitat",
            "attributes": {
                "temperature": "hot",
                "terrain": "volcanic"
            }
        }
        cls.loader.load_entity(volcano_data, "location")
        
        # Then create the monster with relationships
        cls.loader.load_entity(
            test_monster_data,
            entity_type="monster",
            relationship_mapping={
                "evolves_from": "evolution_source",
                "habitat": "lives_in"
            }
        )
    
    def test_knowledge_graph_basic(self):
        """Test basic knowledge graph operations."""
        # Test entity creation and retrieval
        entities = self.knowledge_graph.search_entities("TestHatchy")
        self.assertTrue(len(entities) > 0)
        self.assertEqual(entities[0]["name"], "TestHatchy")
        
        # Test relationship creation and retrieval
        relationships = self.knowledge_graph.get_relationships("TestHatchy")
        self.assertTrue(len(relationships) > 0)
        
        # Test entity types
        entity_types = self.knowledge_graph.get_entity_types()
        self.assertIn("monster", entity_types)
    
    def test_enhanced_loader(self):
        """Test the enhanced data loader."""
        # Test loading a single entity
        test_data = {
            "name": "TestHatchy2",
            "generation": "2",
            "element": "Water",
            "evolves_from": "None",
            "habitat": "Ocean",
            "description": "Another test Hatchy"
        }
        
        self.loader.load_entity(
            test_data,
            entity_type="monster",
            relationship_mapping={
                "evolves_from": "evolution_source",
                "habitat": "lives_in"
            }
        )
        
        # Verify the entity was loaded
        entity = self.knowledge_graph.search_entities("TestHatchy2")[0]
        self.assertEqual(entity["name"], "TestHatchy2")
        self.assertEqual(entity["element"], "Water")
    
    def test_contextual_retriever(self):
        """Test the contextual retriever."""
        retriever = ContextualRetriever(
            knowledge_graph=self.knowledge_graph,
            vector_store=self.vector_store
        )
        
        # Test retrieval with entity context
        context = retriever.get_context("Tell me about TestHatchy")
        self.assertTrue(len(context) > 0)
        self.assertIn("TestHatchy", str(context))
    
    def test_enhanced_chatbot(self):
        """Test the enhanced chatbot."""
        # Test basic response generation
        response = self.chatbot.generate_response("What is TestHatchy?")
        self.assertTrue(isinstance(response, dict))
        self.assertIn("response", response)
        self.assertTrue(len(response["response"]) > 0)
        
        # Test response validation
        self.assertIn("validation", response)
        self.assertTrue(isinstance(response["validation"], dict))
    
    def test_generation_query(self):
        """Test generation-based queries."""
        gen1_entities = self.knowledge_graph.get_entities_by_generation("1")
        self.assertTrue(len(gen1_entities) > 0)
        self.assertTrue(any(e["name"] == "TestHatchy" for e in gen1_entities))
    
    def test_element_query(self):
        """Test element-based queries."""
        fire_entities = self.knowledge_graph.search_entities(
            "",
            entity_type="monster",
            filters={"element": "Fire"}
        )
        self.assertTrue(len(fire_entities) > 0)
        self.assertTrue(any(e["name"] == "TestHatchy" for e in fire_entities))
    
    def test_relationship_query(self):
        """Test relationship queries."""
        # Test habitat relationship
        relationships = self.knowledge_graph.get_relationships("TestHatchy", "lives_in")
        self.assertTrue(len(relationships) > 0)
        self.assertEqual(relationships[0]["target"], "Volcano")
    
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test invalid entity search
        empty_results = self.knowledge_graph.search_entities("NonexistentHatchy")
        self.assertEqual(len(empty_results), 0)
        
        # Test invalid relationship query
        empty_rels = self.knowledge_graph.get_relationships("NonexistentHatchy")
        self.assertEqual(len(empty_rels), 0)
        
        # Test chatbot with invalid query
        response = self.chatbot.generate_response("")
        self.assertIn("validation", response)
        self.assertFalse(response["validation"]["is_valid"])

if __name__ == '__main__':
    unittest.main() 