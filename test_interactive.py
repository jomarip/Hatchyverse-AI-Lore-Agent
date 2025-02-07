import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader
from src.models.enhanced_chatbot import EnhancedChatbot
from src.models.contextual_retriever import ContextualRetriever
from src.models.registry import RelationshipRegistry

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set specific loggers to higher levels to reduce noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Ensure our loggers are at DEBUG level
logging.getLogger('src.models.enhanced_loader').setLevel(logging.DEBUG)
logging.getLogger('src.models.knowledge_graph').setLevel(logging.DEBUG)

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
        self.vector_store = FAISS.from_texts(
            texts=["Initial text for vector store"],
            embedding=self.embeddings
        )
        
        # Initialize chatbot with all components
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
            loader.set_data_directory(self.data_dir)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Load all CSV data with relationships
            loaded_entities = loader.load_all_data()
            logger.info("Loaded entities:")
            for entity_type, count in loaded_entities.items():
                logger.info(f"- {count} {entity_type} entities")
            
            # Load text files for additional context
            text_files = list(self.data_dir.glob("*.txt"))
            text_files.extend(self.data_dir.glob("*.md"))
            
            for text_file in text_files:
                chunks = loader.load_text_data(str(text_file))
                if chunks:
                    logger.info(f"Loaded {len(chunks)} text chunks from {text_file.name}")
            
            # Log knowledge graph statistics
            stats = self.knowledge_graph.get_statistics()
            logger.info("\nKnowledge Graph Statistics:")
            logger.info(f"- Total Entities: {stats['total_entities']}")
            logger.info(f"- Entity Types: {', '.join(stats['entity_types'])}")
            logger.info(f"- Total Relationships: {stats['total_relationships']}")
            logger.info(f"- Relationship Types: {', '.join(stats['relationship_types'])}")
            logger.info(f"- Elements Distribution: {dict(stats['element_counts'])}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
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

def main():
    """Main entry point for the interactive test."""
    try:
        tester = InteractiveTest()
        tester.interactive_session()
    except Exception as e:
        logger.error("Fatal error in main:", exc_info=True)
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    # Set up debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()

def load_knowledge_graph():
    """Load the latest knowledge graph."""
    graph_path = Path("knowledge_graphs/knowledge_graph_latest.json")
    if not graph_path.exists():
        raise FileNotFoundError("Knowledge graph not found. Run build_knowledge_graph.py first.")
        
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    return HatchyKnowledgeGraph.from_dict(graph_data)

def initialize_components():
    """Initialize all required components."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load knowledge graph
        logger.info("Loading knowledge graph...")
        knowledge_graph = load_knowledge_graph()
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7
        )
        
        # Set up vector store
        logger.info("Setting up vector store...")
        data_dir = Path("data")
        text_files = [
            "Hatchy World Comic_ Chaos saga.txt",
            "Hatchy World _ world design.txt",
            "HWCS - Simplified main arc and arc suggestions.txt",
            "Hatchyverse Eco Presentation v3.txt"
        ]
        
        texts = []
        for file in text_files:
            try:
                with open(data_dir / file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts, embeddings)
        
        # Initialize retriever
        retriever = ContextualRetriever(
            knowledge_graph,
            vector_store
        )
        
        # Initialize chatbot
        relationship_registry = RelationshipRegistry()
        chatbot = EnhancedChatbot(llm)
        
        return chatbot
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def run_test_queries(chatbot):
    """Run a set of test queries."""
    test_queries = [
        "what is Omniterra",
        "How many Gen1 Hatchy are there?",
        "Can you list the Gen1 Fire hatchy?",
        "How are Firret and FIradactus similar?",
        "What can you tell me about Ixor?",
        "What Gen1 and Gen2 hatchy are potentially rideable?",
        "What armor piece(s) are related to buzzkill?"
    ]
    
    session_id = "interactive_test"
    
    for query in test_queries:
        print(f"\n=== Testing: {query} ===")
        try:
            response = chatbot.process_message(session_id, query)
            print(f"Response: {response['response']}")
            print(f"Confidence: {response.get('confidence', 'N/A')}")
            if 'sources' in response:
                print("Sources used:", response['sources'])
        except Exception as e:
            print(f"Error processing query: {str(e)}")
        input("\nPress Enter to continue...")

def interactive_mode(chatbot):
    """Run in interactive mode."""
    session_id = "interactive_session"
    print("\nEntering interactive mode. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == 'exit':
            break
            
        try:
            response = chatbot.process_message(session_id, query)
            print(f"\nResponse: {response['response']}")
            print(f"Confidence: {response.get('confidence', 'N/A')}")
            if 'sources' in response:
                print("Sources used:", response['sources'])
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Initialize components
    chatbot = initialize_components()
    
    # Ask user for mode
    mode = input("Select mode (1: Test queries, 2: Interactive): ").strip()
    
    if mode == "1":
        run_test_queries(chatbot)
    elif mode == "2":
        interactive_mode(chatbot)
    else:
        print("Invalid mode selected.") 