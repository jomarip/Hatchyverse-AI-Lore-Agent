# Hatchyverse Community Lore Management System
# Hatchyverse Lore AI Agent for Decentralized IP

## Overview
A decentralized intellectual property management system powered by AI that enables real-time lore evolution and validation.

## Core Concept
Our platform transforms traditional fan wikis into dynamic, AI-powered narrative hubs by:

- Leveraging advanced language models (LLMs) and semantic search (via LangChain and FAISS/Pinecone)
- Enabling real-time lore co-creation and validation between fans, developers and artists
- Facilitating decentralized, community-governed IP management on Web3 platforms like Avalanche
- Reducing production costs while empowering community creators

## Key Features

### Dynamic Narrative Engine
- AI "Lorekeepers" that provide context and guide lore contributions
- Real-time conflict detection to maintain canonical consistency
- Interactive fan query system for lore exploration


## Features

### Core Functionality
- **Interactive Lore Exploration**: Users can ask questions about any aspect of Hatchyverse lore
- **Semantic Search**: Advanced search capabilities using embeddings to find relevant lore
- **Lore Validation**: Automatic checking of new submissions against existing canon
- **Multi-Model Support**: Compatible with multiple AI providers (OpenAI, Anthropic, Deepseek)
- **Conflict Detection**: Identifies potential contradictions with existing lore
- **Constructive Feedback**: Provides suggestions for improving submissions

- **Enhanced Knowledge Graph**: Flexible graph structure for storing entities and their relationships
- **Smart Context Retrieval**: Combines vector search and graph traversal for relevant information
- **Multi-Format Data Loading**: Support for CSV, JSON, and text files with relationship extraction
- **Relationship Analysis**: Tools for analyzing connections between entities
- **Timeline Generation**: Create event timelines for entities
- **Network Visualization**: Generate entity relationship networks
- **Validation System**: Ensures response consistency with knowledge graph

### Data Management
- Load and manage monster data across multiple generations
- Support for items, equipment, and world design data
- Type-safe attributes with proper validation
- Asset path management for images and sounds
- Comprehensive test coverage

## Project Structure

```
hatchyverse/
├── src/
│   ├── models/
│   │   ├── enhanced_chatbot.py      # Main chatbot implementation
│   │   ├── knowledge_graph.py       # Knowledge graph management
│   │   ├── contextual_retriever.py  # Context retrieval system
│   │   └── enhanced_loader.py       # Data loading and processing
│   ├── api/                         # API endpoints
│   ├── utils/                       # Utility functions
│   └── config/                      # Configuration files
├── data/                           # Data files
├── tests/                          # Test cases
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Setup and Configuration

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
Create a `.env` file with:
```env
# Common settings
DATA_DIR=./data
VECTOR_STORE_PATH=./data/vector_store

# Choose your model provider (openai/anthropic/deepseek)
MODEL_PROVIDER=openai

# Provider-specific settings
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_NAME=gpt-4-1106-preview

# Optional: Alternative providers
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL_NAME=claude-3-sonnet-20240229

DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL_NAME=deepseek-chat
```

3. **Data Preparation**
Place your data files in the `data` directory:
- `monsters.csv`: Monster/creature data
- `items.csv`: Equipment and items
- `world_design.txt`: World building and locations

4. **Running the Application**
```bash
uvicorn src.api.main:app --reload
```

## Testing

The project includes a comprehensive test suite with multiple testing approaches:

### 1. Unit Tests
Run the unit test suite:
```bash
python3 -m pytest tests/test_components.py
```

The `test_components.py` file contains core unit tests that verify the functionality of key system components:

#### 1. Knowledge Graph Tests
- **test_knowledge_graph_basic**: Tests fundamental knowledge graph operations
  - Entity creation and retrieval
  - Relationship creation and validation
  - Entity type management
  - Basic search functionality

#### 2. Data Loading Tests
- **test_enhanced_loader**: Verifies the data loading system
  - Entity creation with attributes
  - Relationship mapping
  - Automatic target entity creation
  - Data validation and error handling

#### 3. Contextual Retrieval Tests
- **test_contextual_retriever**: Tests the context retrieval system
  - Entity-based context extraction
  - Query-based information retrieval
  - Context relevance verification
  - Integration with vector store

#### 4. Chatbot Tests
- **test_enhanced_chatbot**: Validates the chatbot functionality
  - Response generation
  - Context integration
  - Response validation
  - Error handling

#### 5. Query Tests

Run the retrieval system tests:
```bash
python3 run_tests.py
```
- **test_generation_query**: Tests generation-based entity queries
  - Filtering by generation
  - Generation metadata handling
  - Cache utilization

- **test_element_query**: Validates element-based filtering
  - Element type filtering
  - Attribute-based search
  - Filter combination handling

#### 6. Relationship Tests
- **test_relationship_query**: Tests relationship management
  - Relationship creation
  - Relationship type validation
  - Bidirectional relationship handling
  - Relationship attribute management

#### 7. Error Handling Tests
- **test_error_handling**: Verifies system robustness
  - Invalid entity handling
  - Missing relationship handling
  - Invalid query responses
  - Error message validation

### Test Setup and Configuration

The test suite uses a controlled test environment with:

1. **Test Data**:
   ```python
   test_monster_data = {
       "name": "TestHatchy",
       "generation": "1",
       "element": "Fire",
       "evolves_from": "None",
       "habitat": "Volcano",
       "description": "A test Hatchy for unit testing"
   }
   ```

2. **Location Data**:
   ```python
   volcano_data = {
       "name": "Volcano",
       "type": "location",
       "description": "A volcanic habitat",
       "attributes": {
           "temperature": "hot",
           "terrain": "volcanic"
       }
   }
   ```

3. **Component Initialization**:
   - OpenAI embeddings for vector search
   - ChatGPT for response generation
   - Chroma vector store for similarity search
   - Knowledge graph for entity relationships
   - Enhanced chatbot for query handling
Interactive test features:
- Data loading verification
- Generation-based queries
- Element-based queries
- Evolution relationship testing
- World information queries
- Live chat testing

Available commands:
1. `data` - Test data loading
2. `gen` - Test generation queries
3. `element` - Test element queries
4. `evolution` - Test evolution queries
5. `world` - Test world information
6. `chat <message>` - Chat with the Hatchyverse chatbot
7. `exit` - Exit testing session

### Running the Tests

1. **Run All Component Tests**:
   ```bash
   python3 -m pytest tests/test_components.py -v
   ```

2. **Run Specific Test Categories**:
   ```bash
   # Run only knowledge graph tests
   python3 -m pytest tests/test_components.py -k "knowledge_graph" -v
   
   # Run only chatbot tests
   python3 -m pytest tests/test_components.py -k "chatbot" -v
   ```

3. **Generate Coverage Report**:
   ```bash
   pytest --cov=src --cov-report=term-missing tests/test_components.py
   ```

### Test Validation Criteria

Each test category ensures:

1. **Data Integrity**:
   - Proper entity creation and storage
   - Accurate relationship mapping
   - Consistent attribute handling

2. **Query Functionality**:
   - Accurate search results
   - Proper filter application
   - Relevant context retrieval

3. **Response Quality**:
   - Valid response generation
   - Proper context integration
   - Accurate validation results

4. **Error Handling**:
   - Graceful error management
   - Informative error messages
   - System stability under invalid inputs

## Usage

1. Initialize the knowledge graph:
   ```python
   from src.models.knowledge_graph import HatchyKnowledgeGraph
   from src.models.enhanced_loader import EnhancedDataLoader
   
   # Create knowledge graph
   graph = HatchyKnowledgeGraph()
   
   # Initialize loader
   loader = EnhancedDataLoader(graph)
   
   # Load data
   loader.load_directory('data/')
   ```

2. Use the chatbot:
   ```python
   from src.models.enhanced_chatbot import EnhancedChatbot
   from langchain_core.language_models import BaseLLM
   
   # Initialize chatbot
   chatbot = EnhancedChatbot(
       llm=your_llm,
       knowledge_graph=graph,
       vector_store=your_vector_store
   )
   
   # Generate response
   response = chatbot.generate_response("Tell me about the first generation Hatchies")
   ```

## Data Loading

The system supports multiple data formats:

### CSV Files
```python
loader.load_csv_data(
    'data/entities.csv',
    entity_type='monster',
    relationship_mapping={
        'evolves_from': 'evolution_source',
        'habitat': 'lives_in'
    }
)
```

### JSON Files
```python
loader.load_json_data(
    'data/lore.json',
    entity_mapping={
        'name': 'title',
        'description': 'content',
        'type': 'category'
    }
)
```

### Text Files
```python
loader.load_text_data(
    'data/story.txt',
    chunk_size=1000,
    overlap=200
)
```

## Validation Process

1. **Semantic Analysis**
   - Converts submissions into embeddings
   - Compares with existing lore vectors
   - Identifies potential conflicts

2. **Conflict Detection**
   - Checks element consistency
   - Validates relationship logic
   - Verifies power level balance
   - Ensures consistency with existing canon

3. **Feedback Generation**
   - Provides specific improvement suggestions
   - Highlights related existing lore
   - Explains any detected conflicts
   - Suggests ways to maintain consistency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- Hatchyverse Community
- LangChain Framework
- FastAPI 