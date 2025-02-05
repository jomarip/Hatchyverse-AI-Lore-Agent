import logging
from pathlib import Path
from dotenv import load_dotenv
from src.data.data_loader import DataLoader
from src.models.lore_validator import LoreValidator
from src.models.chatbot import LoreChatbot
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        
        # Initialize embeddings and validator
        embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        logger.debug(f"Initialized embeddings with model: {os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
        self.validator = LoreValidator(embeddings)
        
        # Initialize chatbot first
        logger.info("Initializing chatbot...")
        self.chatbot = LoreChatbot(
            data_store={
                'hatchy': [],
                'items': [],
                'stories': [],
                'world_info': []
            },
            validator=self.validator
        )
        
        # Load data using new approach
        logger.info("Loading data...")
        self._load_all_data()
        
        logger.info("Test environment initialized")
        
    def _load_all_data(self):
        """Load all data using the new data loading approach."""
        try:
            # Load Gen1 Hatchy
            gen1_path = self.data_dir / "Hatchy - Monster Data - gen 1.csv"
            if gen1_path.exists():
                self.chatbot.load_data(str(gen1_path), "hatchy")
                logger.info(f"Loaded Gen1 hatchy data: {len(self.chatbot.data_store['hatchy'])} entries")
            
            # Load Gen2 Hatchy
            gen2_path = self.data_dir / "Hatchy - Monster Data - gen 2.csv"
            if gen2_path.exists():
                self.chatbot.load_data(str(gen2_path), "hatchy")
                logger.info(f"Updated hatchy data: {len(self.chatbot.data_store['hatchy'])} total entries")
            
            # Load Items
            items_paths = [
                "PFP-hatchyverse - Masters data - 2.EQUIP Info.csv",
                "PFP-hatchyverse - Masters data - masters-items-db.csv"
            ]
            for item_file in items_paths:
                item_path = self.data_dir / item_file
                if item_path.exists():
                    self.chatbot.load_data(str(item_path), "items")
                    logger.info(f"Loaded items from {item_file}: {len(self.chatbot.data_store['items'])} total items")
            
            # Load Story Data with enhanced processing
            story_path = self.data_dir / "Hatchy World Comic_ Chaos saga.txt"
            if story_path.exists():
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", "• ", "Ep ", "Arc "]
                )
                
                with open(story_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    chunks = text_splitter.create_documents([raw_text], metadatas=[{
                        "source": "chaos_saga",
                        "section_type": "story",
                        "title": "Chaos Saga"
                    }])
                    
                    story_data = {
                        "title": "Chaos Saga",
                        "content": raw_text,
                        "chunks": chunks,
                        "metadata": {
                            "source": "chaos_saga",
                            "type": "story"
                        }
                    }
                    self.chatbot.data_store['stories'].append(story_data)
                    self.validator.vector_store.add_documents(chunks)
                    logger.info(f"Loaded Chaos saga story data with {len(chunks)} chunks")
            
            # Load World Design Data
            world_files = [
                "Hatchy World _ world design.txt",
                "Hatchyverse Eco Presentation v3.txt"
            ]
            for world_file in world_files:
                world_path = self.data_dir / world_file
                if world_path.exists():
                    with open(world_path, 'r', encoding='utf-8') as f:
                        world_data = {"title": world_file, "content": f.read()}
                        self.chatbot.data_store['world_info'].append(world_data)
                    logger.info(f"Loaded world design data from {world_file}")
            
            # Log data store statistics
            logger.info("\nData Store Statistics:")
            for data_type, data in self.chatbot.data_store.items():
                logger.info(f"- {data_type}: {len(data)} entries")
                if data and hasattr(self.chatbot.data_schema, 'get'):
                    schema = self.chatbot.data_schema.get(data_type, {})
                    logger.info(f"  Schema: {schema}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def test_data_loading(self):
        """Test and display loaded data."""
        print("\n=== Data Loading Test ===")
        
        for data_type, data in self.chatbot.data_store.items():
            print(f"\n{data_type.title()} Data:")
            print(f"- Total entries: {len(data)}")
            
            if data:
                print("- Sample entry:")
                sample = data[0]
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        if value:  # Only show non-empty values
                            print(f"  {key}: {value}")
                else:
                    print(f"  {sample}")
                
                if data_type in self.chatbot.data_schema:
                    print(f"- Schema:")
                    for field, field_type in self.chatbot.data_schema[data_type].items():
                        print(f"  {field}: {field_type}")
    
    def analyze_story(self, story_text: str):
        """Analyze a story submission for narrative structure and lore consistency."""
        try:
            logger.debug(f"Analyzing story text: {story_text[:100]}...")
            
            # Get narrative analysis
            analysis = self.chatbot.analyze_story_submission(story_text)
            
            print("\nNarrative Analysis Results:")
            print("===========================")
            
            # Print story elements
            if "story_elements" in analysis["narrative_structure"]:
                elements = analysis["narrative_structure"]["story_elements"]
                
                # Characters
                if elements["characters"]:
                    print("\nCharacters:")
                    for char in elements["characters"]:
                        print(f"- {char['name']} ({char['role']})")
                        print(f"  Mentions: {char['mentions']}")
                
                # Settings
                if elements["settings"]:
                    print("\nSettings:")
                    for setting in elements["settings"]:
                        print(f"- {setting['name']} ({setting['type']})")
                        if setting["attributes"]["atmosphere"]:
                            print(f"  Atmosphere: {', '.join(setting['attributes']['atmosphere'])}")
                        if setting["attributes"]["elements"]:
                            print(f"  Elements: {', '.join(setting['attributes']['elements'])}")
                
                # Plot Points
                if elements["plot_points"]:
                    print("\nMajor Plot Points:")
                    for point in elements["plot_points"]:
                        print(f"- Type: {point['type']}")
                        print(f"  Significance: {', '.join(point['significance'])}")
                
                # Themes
                if elements["themes"]:
                    print("\nThemes:")
                    for theme in elements["themes"]:
                        print(f"- {theme['name']} (mentioned {theme['frequency']} times)")
            
            # Print arc analysis
            if "arc_analysis" in analysis["narrative_structure"]:
                arc = analysis["narrative_structure"]["arc_analysis"]
                print("\nStory Arc Analysis:")
                print(f"Complete: {'Yes' if arc['completeness'] else 'No'}")
                if not arc['completeness']:
                    print(f"Missing Elements: {', '.join(arc['missing_elements'])}")
                
                print("\nStructure:")
                for section in arc["structure"]:
                    print(f"- {section['type'].title()}")
            
            # Print temporal analysis
            if "temporal_markers" in analysis["narrative_structure"]:
                markers = analysis["narrative_structure"]["temporal_markers"]
                print("\nTemporal Analysis:")
                for marker in markers:
                    print(f"- {marker['marker']} ({marker['type']}, {marker['relative_position']})")
            
            # Print character relationships
            if "character_relationships" in analysis["narrative_structure"]:
                relationships = analysis["narrative_structure"]["character_relationships"]
                if relationships:
                    print("\nCharacter Relationships:")
                    for rel in relationships:
                        print(f"- {' & '.join(rel['characters'])}: {rel['interaction_type']}")
            
            # Print lore alignment
            if "lore_alignment" in analysis:
                alignment = analysis["lore_alignment"]
                print("\nLore Alignment:")
                print(f"Aligned: {'Yes' if alignment['is_aligned'] else 'No'}")
                
                if alignment["conflicts"]:
                    print("\nConflicts:")
                    for conflict in alignment["conflicts"]:
                        print(f"- {conflict['type']}: {conflict['details']}")
                
                if alignment["enhancements"]:
                    print("\nSuggested Enhancements:")
                    for enhancement in alignment["enhancements"]:
                        print(f"- {enhancement['suggestion']}")
                        if "elements" in enhancement:
                            print(f"  Elements to expand: {', '.join(enhancement['elements'])}")
            
            # Print validation summary
            if "validation_summary" in analysis["narrative_structure"]:
                validation = analysis["narrative_structure"]["validation_summary"]
                print("\nValidation Summary:")
                print(f"Coherent: {'Yes' if validation['is_coherent'] else 'No'}")
                
                if validation["issues"]:
                    print("\nIssues:")
                    for issue in validation["issues"]:
                        print(f"- {issue}")
                
                if validation["strengths"]:
                    print("\nStrengths:")
                    for strength in validation["strengths"]:
                        print(f"- {strength}")
            
        except Exception as e:
            logger.error(f"Error analyzing story: {str(e)}", exc_info=True)
            print(f"Error occurred during analysis: {str(e)}")

    def chat(self, message: str):
        """Chat with the Hatchyverse chatbot."""
        try:
            logger.debug(f"Processing chat message: {message}")
            
            # Log data store status before query
            logger.debug("Current data store status:")
            for data_type, data in self.chatbot.data_store.items():
                logger.debug(f"- {data_type}: {len(data)} entries")
            
            # Check if this looks like a story analysis request
            # Look for story-like indicators in the message
            story_indicators = [
                "analyze this story",
                "check this story",
                "review this story",
                "here's a story",
                "validate this story",
                "story analysis"
            ]
            
            is_story_request = any(indicator in message.lower() for indicator in story_indicators)
            
            if is_story_request:
                # Extract the actual story text - everything after the indicator
                for indicator in story_indicators:
                    if indicator in message.lower():
                        story_text = message[message.lower().find(indicator) + len(indicator):].strip()
                        if story_text:
                            self.analyze_story(story_text)
                            return
            
            # Regular chat response if not a story analysis
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
        print("   - For story analysis, include phrases like 'analyze this story' or 'check this story'")
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