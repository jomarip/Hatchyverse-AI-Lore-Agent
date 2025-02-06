import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseLLM
from langchain_community.chat_models import ChatOpenAI
from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader
from src.models.enhanced_chatbot import EnhancedChatbot
from src.models.contextual_retriever import ContextualRetriever
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format
    handlers=[
        logging.StreamHandler()
    ]
)

# Set specific loggers to higher levels to reduce noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class InteractiveTest:
    """Interactive testing environment for the Hatchyverse system."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.data_dir = Path("data")
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir.absolute()}")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-4-0125-preview'),
            temperature=0.7
        )
        
        # Initialize knowledge graph
        self.knowledge_graph = HatchyKnowledgeGraph()
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
            embedding_function=self.embeddings
        )
        
        # Initialize chatbot
        self.chatbot = EnhancedChatbot(
            llm=self.llm,
            knowledge_graph=self.knowledge_graph,
            vector_store=self.vector_store
        )
        
        # Load data
        logger.info("Loading data...")
        self._load_all_data()
        
        logger.info("Test environment initialized")
        
    def _load_all_data(self):
        """Load all data using the enhanced loader."""
        try:
            loader = EnhancedDataLoader(self.knowledge_graph)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Load monster data
            gen1_path = self.data_dir / "Hatchy - Monster Data - gen 1.csv"
            if gen1_path.exists():
                gen1_entities = loader.load_csv_data(
                    str(gen1_path),
                    entity_type="monster",
                    relationship_mapping={
                        "evolves_from": "evolution_source",
                        "habitat": "lives_in"
                    }
                )
                if gen1_entities:
                    # Create documents for vector store
                    docs = []
                    for entity in gen1_entities:
                        text = f"Name: {entity['name']}\n"
                        text += f"Generation: 1\n"
                        text += f"Type: {entity.get('attributes', {}).get('element', 'Unknown')}\n"
                        text += f"Description: {entity.get('attributes', {}).get('description', '')}\n"
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            docs.append(Document(
                                page_content=chunk,
                                metadata={"entity_id": entity["id"], "type": "monster", "generation": "1"}
                            ))
                    
                    if docs:
                        self.vector_store.add_documents(docs)
                    logger.info("Loaded Gen1 Hatchy data")
            
            gen2_path = self.data_dir / "Hatchy - Monster Data - gen 2.csv"
            if gen2_path.exists():
                gen2_entities = loader.load_csv_data(
                    str(gen2_path),
                    entity_type="monster",
                    relationship_mapping={
                        "evolves_from": "evolution_source",
                        "habitat": "lives_in"
                    }
                )
                if gen2_entities:
                    # Create documents for vector store
                    docs = []
                    for entity in gen2_entities:
                        text = f"Name: {entity['name']}\n"
                        text += f"Generation: 2\n"
                        text += f"Type: {entity.get('attributes', {}).get('element', 'Unknown')}\n"
                        text += f"Description: {entity.get('attributes', {}).get('description', '')}\n"
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            docs.append(Document(
                                page_content=chunk,
                                metadata={"entity_id": entity["id"], "type": "monster", "generation": "2"}
                            ))
                    
                    if docs:
                        self.vector_store.add_documents(docs)
                    logger.info("Loaded Gen2 Hatchy data")
            
            # Load world data
            world_path = self.data_dir / "Hatchy World _ world design.txt"
            if world_path.exists():
                with open(world_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk, metadata={"type": "world_design"}) for chunk in chunks]
                if docs:
                    self.vector_store.add_documents(docs)
                
                # Load into knowledge graph
                world_entities = loader.load_text_data(str(world_path))
                if world_entities:
                    logger.info(f"Loaded {len(world_entities)} world design entities")
            
            # Load story data
            story_path = self.data_dir / "Hatchy World Comic_ Chaos saga.txt"
            if story_path.exists():
                with open(story_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk, metadata={"type": "story"}) for chunk in chunks]
                if docs:
                    self.vector_store.add_documents(docs)
                
                # Load into knowledge graph
                story_entities = loader.load_text_data(str(story_path))
                if story_entities:
                    logger.info(f"Loaded {len(story_entities)} story entities")
            
            # Load eco presentation data
            eco_path = self.data_dir / "Hatchyverse Eco Presentation v3.txt"
            if eco_path.exists():
                with open(eco_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk, metadata={"type": "eco_presentation"}) for chunk in chunks]
                if docs:
                    self.vector_store.add_documents(docs)
                
                # Load into knowledge graph
                eco_entities = loader.load_text_data(str(eco_path))
                if eco_entities:
                    logger.info(f"Loaded {len(eco_entities)} eco presentation entities")
            
            # Log knowledge graph statistics
            stats = self.knowledge_graph.get_statistics()
            logger.info("\nKnowledge Graph Statistics:")
            for key, value in stats.items():
                logger.info(f"- {key}: {value}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def test_data_loading(self):
        """Test and display loaded data."""
        print("\n=== Data Loading Test ===")
        
        # Get knowledge graph statistics
        stats = self.knowledge_graph.get_statistics()
        print("\nKnowledge Graph Statistics:")
        for key, value in stats.items():
            print(f"- {key}: {value}")
        
        # Test entity retrieval
        print("\nSample Entities by Type:")
        entity_types = self.knowledge_graph.get_entity_types()
        for entity_type in entity_types:
            entities = self.knowledge_graph.search_entities("", entity_type=entity_type, limit=3)
            print(f"\n{entity_type.title()} Entities:")
            for entity in entities:
                print(f"- {entity['name']}")
                if 'description' in entity:
                    print(f"  Description: {entity['description'][:100]}...")
        
        # Test relationship types
        print("\nAvailable Relationship Types:")
        rel_types = self.knowledge_graph.get_relationship_types()
        for rel_type in rel_types:
            print(f"- {rel_type}")
    
    def test_generation_query(self):
        """Test querying Hatchy by generation."""
        print("\n=== Generation Query Test ===")
        
        for gen in ["1", "2", "3"]:
            entities = self.knowledge_graph.get_entities_by_generation(gen)
            print(f"\nGeneration {gen} Hatchy:")
            for entity in entities[:5]:  # Show first 5
                print(f"- {entity['name']}")
    
    def test_element_query(self):
        """Test querying Hatchy by element."""
        print("\n=== Element Query Test ===")
        
        elements = ["Fire", "Water", "Plant", "Dark", "Light", "Void"]
        for element in elements:
            results = self.chatbot.generate_response(f"Tell me about {element} type Hatchy")
            print(f"\n{element} Type Hatchy:")
            print(results["response"])
    
    def test_evolution_query(self):
        """Test querying evolution information."""
        print("\n=== Evolution Query Test ===")
        
        queries = [
            "Which Hatchy can be ridden?",
            "Tell me about final evolution stages",
            "What are the largest Hatchy?"
        ]
        
        for query in queries:
            results = self.chatbot.generate_response(query)
            print(f"\nQuery: {query}")
            print(f"Response: {results['response']}")
            if results.get("validation"):
                print("Validation:", results["validation"])
    
    def test_world_query(self):
        """Test querying world information."""
        print("\n=== World Information Test ===")
        
        queries = [
            "Tell me about Omniterra",
            "What are the different regions in the Hatchy world?",
            "What is the Crystal Lake region?"
        ]
        
        for query in queries:
            results = self.chatbot.generate_response(query)
            print(f"\nQuery: {query}")
            print(f"Response: {results['response']}")
    
    def chat(self, message: str):
        """Chat with the Hatchyverse chatbot."""
        try:
            logger.debug(f"Processing chat message: {message}")
            
            # Generate response
            response = self.chatbot.generate_response(message)
            
            print("\nChatbot Response:")
            print(response["response"])
            
            if response.get("validation"):
                print("\nValidation Results:")
                if not response["validation"]["is_valid"]:
                    print("⚠️ Issues detected:")
                    for issue in response["validation"]["issues"]:
                        print(f"- {issue}")
                else:
                    print("✅ Response validated")
                
                if response["validation"]["enhancements"]:
                    print("\nSuggested Enhancements:")
                    for enhancement in response["validation"]["enhancements"]:
                        print(f"- {enhancement['suggestion']}")
            
        except Exception as e:
            logger.error(f"Error during chat: {e}", exc_info=True)
            print(f"Error occurred: {str(e)}")
    
    def interactive_session(self):
        """Start an interactive testing session."""
        print("\nWelcome to Hatchyverse Interactive Testing!")
        print("Available commands:")
        print("1. 'data' - Test data loading")
        print("2. 'gen' - Test generation queries")
        print("3. 'element' - Test element queries")
        print("4. 'evolution' - Test evolution queries")
        print("5. 'world' - Test world information")
        print("6. 'chat <message>' - Chat with the Hatchyverse chatbot")
        print("7. 'exit' - Exit testing session")
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                logger.debug(f"Received command: {command}")
                
                if command.lower() == 'exit':
                    break
                elif command.lower() == 'data':
                    self.test_data_loading()
                elif command.lower() == 'gen':
                    self.test_generation_query()
                elif command.lower() == 'element':
                    self.test_element_query()
                elif command.lower() == 'evolution':
                    self.test_evolution_query()
                elif command.lower() == 'world':
                    self.test_world_query()
                elif command.lower().startswith('chat '):
                    message = command[5:].strip()
                    self.chat(message)
                else:
                    print("Unknown command. Try 'data', 'gen', 'element', 'evolution', 'world', 'chat <message>', or 'exit'")
            
            except Exception as e:
                logger.error(f"Error during testing: {e}", exc_info=True)
                print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        tester = InteractiveTest()
        tester.interactive_session()
    except Exception as e:
        logger.error("Fatal error in main:", exc_info=True)
        print(f"Fatal error: {str(e)}") 