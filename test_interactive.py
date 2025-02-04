import logging
from pathlib import Path
from dotenv import load_dotenv
from src.data.data_loader import DataLoader
from src.models.lore_validator import LoreValidator
from src.models.chatbot import LoreChatbot
from langchain_openai import OpenAIEmbeddings
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG level
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
        logger.debug(f"Data directory found: {self.data_dir.absolute()}")
        logger.debug(f"Data directory contents: {[f.name for f in self.data_dir.iterdir()]}")
        
        self.loader = DataLoader(str(self.data_dir))  # Convert Path to string
        
        # Initialize embeddings and validator
        embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        logger.debug(f"Initialized embeddings with model: {os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
        self.validator = LoreValidator(embeddings)
        
        # Load data and build knowledge base
        logger.info("Loading data...")
        entities = self.loader.load_all_data()
        
        # Strategic debug point
        if entities is None:
            logger.error("load_all_data() returned None!")
            raise ValueError("DataLoader.load_all_data() returned None instead of a list of entities")
        
        logger.info(f"Data loaded: {len(entities)} entities")
        if len(entities) == 0:
            logger.warning("No entities were loaded from the data sources")
        else:
            logger.debug(f"Loaded {len(entities)} total entities")
            entity_types = {}
            for entity in entities:
                entity_type = entity.entity_type
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            logger.debug("Entity type breakdown:")
            for entity_type, count in entity_types.items():
                logger.debug(f"  {entity_type}: {count}")
        
        # Build knowledge base
        logger.info("Building knowledge base...")
        try:
            self.validator.build_knowledge_base(entities)
            logger.debug("Knowledge base built successfully")
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}", exc_info=True)
            raise
        
        # Initialize chatbot
        logger.info("Initializing chatbot...")
        self.chatbot = LoreChatbot(
            self.validator,
            model_provider=os.getenv('MODEL_PROVIDER', 'openai'),
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'),
            temperature=0.7
        )
        
        logger.info("Test environment initialized")
    
    def test_data_loading(self):
        """Test and display loaded data."""
        print("\n=== Data Loading Test ===")
        
        if self.loader.monster_data:
            print("\nMonster Data Sources:")
            for source, df in self.loader.monster_data.items():
                print(f"- {source}: {len(df)} entries")
                if len(df) > 0:
                    print(f"  Sample monster: {df.iloc[0]['name']}")
                    logger.debug(f"Sample monster data: {df.iloc[0].to_dict()}")
        
        if self.loader.item_data:
            print("\nItem Data Sources:")
            for source, df in self.loader.item_data.items():
                print(f"- {source}: {len(df)} entries")
                if len(df) > 0:
                    print(f"  Sample item: {df.iloc[0]['name']}")
                    logger.debug(f"Sample item data: {df.iloc[0].to_dict()}")
        
        if self.loader.world_data:
            print("\nWorld Data:")
            world_design = self.loader.world_data.get('world_design', {})
            print(f"- Elements: {len(world_design.get('elements', {}))}")
            print(f"- Biomes: {len(world_design.get('biomes', []))}")
            print(f"- Landmarks: {len(world_design.get('landmarks', []))}")
            logger.debug(f"World data structure: {world_design.keys()}")
    
    def chat(self, message: str):
        """Chat with the Hatchyverse chatbot."""
        try:
            logger.debug(f"Processing chat message: {message}")
            response = self.chatbot.generate_response(message)
            print("\nChatbot Response:")
            print(response["response"])
            
            if response.get("validation"):
                print("\nValidation Results:")
                if response["validation"]["conflicts"]:
                    print("⚠️ Conflicts detected:")
                    for conflict in response["validation"]["conflicts"]:
                        print(f"- {conflict['reason']}")
                else:
                    print("✅ No conflicts detected")
                
        except Exception as e:
            logger.error(f"Error during chat: {e}", exc_info=True)
            print(f"Error occurred: {str(e)}")
    
    def interactive_session(self):
        """Start an interactive testing session."""
        print("\nWelcome to Hatchyverse Interactive Testing!")
        print("Available commands:")
        print("1. 'data' - Test data loading")
        print("2. 'chat <message>' - Chat with the Hatchyverse chatbot")
        print("3. 'exit' - Exit testing session")
        
        chat_history = []
        while True:
            try:
                command = input("\nEnter command: ").strip()
                logger.debug(f"Received command: {command}")
                
                if command.lower() == 'exit':
                    break
                elif command.lower() == 'data':
                    self.test_data_loading()
                elif command.lower().startswith('chat '):
                    message = command[5:].strip()
                    self.chat(message)
                else:
                    print("Unknown command. Try 'data', 'chat <message>', or 'exit'")
            
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