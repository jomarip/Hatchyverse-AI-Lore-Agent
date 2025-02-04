import os
import sys
import pytest
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lore_validator import LoreValidator
from src.models.chatbot import LoreChatbot
from src.data.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Setup test environment."""
    # Load test environment variables
    load_dotenv('.env.test')
    
    # Ensure required environment variables are set
    required_vars = [
        'OPENAI_API_KEY',
        'DATA_DIR',
        'VECTOR_STORE_PATH',
        'MODEL_PROVIDER'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} not set")
            
    # Create vector store directory if it doesn't exist
    vector_store_path = os.getenv('VECTOR_STORE_PATH')
    os.makedirs(vector_store_path, exist_ok=True)

@pytest.fixture(scope="session")
def test_data():
    """Load test data."""
    logger.info("Loading test data...")
    
    data_dir = os.getenv('DATA_DIR')
    loader = DataLoader(data_dir)
    
    # Test loading monsters
    monsters = loader.load_monsters(f"{data_dir}/monsters.csv")
    logger.info(f"Loaded {len(monsters)} monsters")
    assert len(monsters) > 0, "No monsters loaded"
    assert monsters[0].name == "Aquafrost", "First monster should be Aquafrost"
    
    # Test loading items
    items = loader.load_items(f"{data_dir}/items.csv")
    logger.info(f"Loaded {len(items)} items")
    assert len(items) > 0, "No items loaded"
    assert items[0].name == "Frost Crystal", "First item should be Frost Crystal"
    
    # Test loading world data
    locations = loader.load_world_data(f"{data_dir}/world_design.txt")
    logger.info(f"Loaded {len(locations)} locations")
    assert len(locations) > 0, "No locations loaded"
    
    all_entities = monsters + items + locations
    logger.info(f"Loaded total of {len(all_entities)} entities")
    return all_entities

@pytest.fixture(scope="session")
def validator(test_data):
    """Initialize and return the LoreValidator."""
    logger.info("Initializing LoreValidator...")
    
    # Initialize embeddings and validator
    embeddings = OpenAIEmbeddings(model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'))
    validator = LoreValidator(embeddings)
    
    # Build knowledge base
    validator.build_knowledge_base(test_data)
    return validator

@pytest.fixture(scope="session")
def chatbot(validator):
    """Initialize and return the Chatbot."""
    logger.info("Initializing Chatbot...")
    
    return LoreChatbot(
        validator,
        model_provider=os.getenv('MODEL_PROVIDER'),
        model_name=os.getenv('OPENAI_MODEL_NAME')
    )

def test_conflict_detection(validator):
    """Test the conflict detection functionality."""
    test_cases = [
        {
            "input": "A new water-type Hatchy that lives in volcanoes",
            "should_conflict": True,
            "description": "Water type in volcano should conflict"
        },
        {
            "input": "A water-type Hatchy that creates beautiful ice sculptures",
            "should_conflict": False,
            "description": "Water type with ice abilities should not conflict",
            "required_keywords": ["ice", "water"],
            "forbidden_keywords": ["volcano", "lightning"]
        },
        {
            "input": "A dark-type Hatchy that generates lightning",
            "should_conflict": True,
            "description": "Dark type with lightning should conflict"
        }
    ]
    
    for case in test_cases:
        result = validator.check_conflict(case["input"])
        has_conflicts = len(result["conflicts"]) > 0
        logger.info(f"Testing case: {case['description']}")
        logger.info(f"Expected conflict: {case['should_conflict']}, Got: {has_conflicts}")
        assert has_conflicts == case["should_conflict"], \
            f"Conflict detection failed for: {case['description']}"
        
        if "required_keywords" in case:
            for kw in case["required_keywords"]:
                assert kw in case["input"].lower(), f"Missing required keyword: {kw}"
                
        if "forbidden_keywords" in case:
            for kw in case["forbidden_keywords"]:
                assert kw not in case["input"].lower(), f"Contains forbidden keyword: {kw}"

def test_chatbot_queries(chatbot):
    """Test the chatbot query functionality."""
    test_queries = [
        {
            "query": "Tell me about water-type Hatchies",
            "expected_keywords": ["Aquafrost", "ice", "crystal"],
            "alternative_keywords": []
        },
        {
            "query": "What items are available for fire-type Hatchies?",
            "expected_keywords": ["fire", "attack"],
            "alternative_keywords": ["Flame Essence", "flame gems", "fire essence crystals"]
        },
        {
            "query": "Where do dark-type Hatchies live?",
            "expected_keywords": ["Shadow Caverns", "mysterious"],
            "alternative_keywords": ["shadows", "dark"]
        }
    ]
    
    for test in test_queries:
        logger.info(f"\nTesting query: {test['query']}")
        response = chatbot.generate_response(test["query"])
        response_text = response["response"].lower()
        
        # Check required keywords
        for keyword in test["expected_keywords"]:
            assert keyword.lower() in response_text, \
                f"Expected keyword '{keyword}' not found in response"
        
        # Check that at least one alternative keyword is present if alternatives exist
        if test["alternative_keywords"]:
            alt_found = any(kw.lower() in response_text for kw in test["alternative_keywords"])
            assert alt_found, \
                f"None of the alternative keywords {test['alternative_keywords']} found in response"

def test_lore_submission(chatbot):
    """Test the lore submission functionality."""
    test_submission = """
    Name: Frostflame
    Type: Monster
    Element: Ice
    Description: A rare Hatchy that can control both ice and fire, creating beautiful crystalline patterns.
    """
    
    logger.info("\nTesting lore submission...")
    response = chatbot.generate_response(f"[SUBMIT]{test_submission}")
    assert "conflict" in response["response"].lower(), \
        "Submission with conflicting elements should be flagged"

def test_main(setup_environment, test_data, validator, chatbot):
    """Run all tests in sequence."""
    try:
        test_conflict_detection(validator)
        test_chatbot_queries(chatbot)
        test_lore_submission(chatbot)
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__]) 