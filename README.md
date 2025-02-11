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

### Query Analysis
The QueryAnalyzer class handles different query types including:
- Generation-specific queries (e.g., "gen1", "generation 2")
- Element-based queries (fire, water, plant, etc.)
- Relationship queries (mountable, evolution, habitat, ability)
- Attribute-based queries (size, stage)

### Context Retrieval
The ContextualRetriever combines multiple data sources:
- Vector store for semantic search
- Knowledge graph for entity relationships
- Filtered retrieval based on query type
- Timeline and story context

### Filtered Retrieval
The FilteredRetriever specifically handles:
- Character/Monster information
- Item/Equipment queries
- Story/Event queries
- Location-based queries

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
- `monsters.csv`: Contains monster/creature stats, abilities, elements, and evolution chains
- `items.csv`: Equipment stats, effects, rarity levels, and crafting requirements
- `world_design.txt`: World lore, location descriptions, and regional histories
- `story.txt`: Main storyline events, character interactions, and plot developments
- `factions.json`: Faction relationships, alliances, and hierarchies
- `characters.json`: Character profiles, backstories, and relationships
- `elements.yaml`: Elemental types, strengths/weaknesses, and special properties
- `timeline.json`: Chronological events and historical milestones

4. **Running the Application**
```bash
uvicorn src.api.main:app --reload
```

## Testing

The Hatchyverse system uses a multi-stage testing approach to ensure proper functionality. Follow these steps in order:

### 1. Build Knowledge Graph
First, build and initialize the knowledge graph:
```bash
python scripts/build_knowledge_graph.py
```
This will:
- Load all data from the data directory
- Build the initial knowledge graph
- Create relationship mappings
- Save the graph to `knowledge_graphs/knowledge_graph_latest.json`

### 2. Run Component Tests
Next, run the core component tests:
```bash
python -m unittest tests/test_components.py -v
```
This validates:
- Knowledge graph operations
- Data loading functionality
- Entity relationships
- Basic query handling

### 3. Run Relationship Tests
Test the enhanced relationship extraction:
```bash
python -m unittest tests/test_relationship_extraction.py -v
```
This verifies:
- Data cleaning
- Relationship extraction
- Pattern learning
- Entity resolution
- Relationship confidence scoring

### 4. Run Interactive Tests
Finally, test the chatbot with predefined queries:
```bash
python test_interactive.py
```
This provides:
- Test suite for common queries
- Interactive chat testing
- Relationship validation
- Response verification

Available commands in interactive mode:
1. `data` - Test data loading
2. `gen` - Test generation queries
3. `element` - Test element queries
4. `evolution` - Test evolution queries
5. `world` - Test world information
6. `chat <message>` - Chat with the Hatchyverse chatbot
7. `exit` - Exit testing session

### Test Coverage

The test suite aims to maintain comprehensive coverage:

#### 1. Knowledge Graph (90-100% coverage)
- Entity creation, retrieval, and deletion
- Relationship management
- Attribute validation
- Search functionality
- Data consistency checks
- Index management

#### 2. Data Loading (85-95% coverage)
- CSV file processing
- JSON data handling
- Text chunk processing
- Relationship extraction
- Error handling
- Type conversion
- Data validation

#### 3. Relationship Extraction (85-95% coverage)
- Pattern recognition
- Confidence scoring
- Entity resolution
- Relationship validation
- Pattern learning
- Context analysis

#### 4. Response Generation (80-90% coverage)
- Context integration
- Response formatting
- Validation checks
- Enhancement suggestions
- Error handling
- Source coverage tracking

### Continuous Learning

The system features dynamic pattern learning and relationship refinement:

1. **Pattern Learning**
   - Learns from high-confidence extractions
   - Refines relationship patterns
   - Improves extraction accuracy over time

2. **Relationship Registry**
   - Maintains learned patterns
   - Tracks relationship confidence
   - Persists improvements across sessions

3. **Knowledge Graph Evolution**
   - Updates entity relationships
   - Refines confidence scores
   - Maintains relationship consistency

### Troubleshooting Tests

If you encounter issues:

1. **Knowledge Graph Build Failures**
   - Check data file formats
   - Verify file permissions
   - Ensure proper data directory structure

2. **Component Test Failures**
   - Check Python environment
   - Verify dependencies
   - Review error logs

3. **Relationship Test Failures**
   - Check pattern definitions
   - Verify entity resolution
   - Review confidence thresholds

4. **Interactive Test Issues**
   - Check LLM API keys
   - Verify vector store initialization
   - Review chat history handling

For detailed logs, run tests with increased verbosity:
```bash
python -m unittest -v tests/test_components.py
python -m unittest -v tests/test_relationship_extraction.py
```

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

## Extending System Capabilities

### Adding New Query Capabilities

The system uses a flexible pattern-based approach for query understanding. Here's how to add new capabilities:

1. **Add New Attribute Patterns**
   ```python
   # In src/models/contextual_retriever.py
   self.attribute_patterns.update({
       'rarity': {
           'patterns': [
               r'(common|rare|legendary|mythic)',
               r'rarity\s+(level|type|of)\s+(\w+)'
           ],
           'value_map': {
               'uncommon': 'rare',
               'super rare': 'legendary'
           },
           'attribute': 'rarity'
       },
       'power_level': {
           'patterns': [
               r'power\s+(?:level|rating)\s*(\d+)',
               r'level\s*(\d+)\s*power'
           ],
           'attribute': 'power_level'
       }
   })
   ```

2. **Add New Query Types**
   ```python
   # Add new query type patterns
   self.query_type_patterns.update({
       'comparison': [
           r'compare\s+(?:between\s+)?(.+)\s+and\s+(.+)',
           r'difference\s+between\s+(.+)\s+and\s+(.+)'
       ],
       'evolution_chain': [
           r'evolution(?:\s+chain|\s+line|\s+path)\s+(?:of\s+)?(\w+)',
           r'how\s+does\s+(\w+)\s+evolve'
       ]
   })
   ```

3. **Add Content Filters**
   ```python
   # Add new attribute filters
   attribute_filters = ['size', 'mountable', 'habitat', 'rarity', 'power_level']
   
   # Add filter logic
   if attr == 'rarity':
       rarity_value = filters['rarity'].lower()
       if not any(term in desc for term in [rarity_value, f"rarity: {rarity_value}"]):
           matches_all = False
           break
   ```

### Adding New Data Types

1. **Define New Entity Types**
   ```python
   # In src/models/knowledge_graph.py
   VALID_ENTITY_TYPES = [
       'monster',
       'location',
       'item',
       'ability',
       'quest',
       'npc'
   ]
   ```

2. **Create Type-Specific Loaders**
   ```python
   # In src/models/enhanced_loader.py
   def load_quest_data(self, file_path: str):
       """Load quest data with specific handling."""
       try:
           with open(file_path, 'r') as f:
               data = json.load(f)
           
           for quest in data:
               # Process quest-specific fields
               entity_id = self.knowledge_graph.add_entity(
                   name=quest['title'],
                   entity_type='quest',
                   attributes={
                       'difficulty': quest.get('difficulty'),
                       'rewards': quest.get('rewards'),
                       'prerequisites': quest.get('prerequisites')
                   }
               )
               
               # Add quest-specific relationships
               if 'required_items' in quest:
                   for item in quest['required_items']:
                       self.knowledge_graph.add_relationship(
                           entity_id,
                           item['id'],
                           'requires_item'
                       )
       except Exception as e:
           logger.error(f"Error loading quest data: {str(e)}")
   ```

3. **Add Type-Specific Validation Rules**
   ```python
   # In src/models/response_validator.py
   def validate_quest_response(self, response: str, context: List[Dict[str, Any]]):
       """Validate quest-specific response content."""
       issues = []
       
       # Check for required quest information
       required_fields = ['difficulty', 'rewards', 'prerequisites']
       for field in required_fields:
           if field.lower() not in response.lower():
               issues.append({
                   'type': 'missing_field',
                   'field': field,
                   'message': f"Response should include quest {field}"
               })
       
       return issues
   ```

### Adding New Relationship Types

1. **Define New Relationships**
   ```python
   # In src/models/knowledge_graph.py
   VALID_RELATIONSHIP_TYPES = [
       'evolves_from',
       'lives_in',
       'drops_item',
       'teaches_ability',
       'starts_quest',
       'requires_item'
   ]
   ```

2. **Add Relationship Extraction Patterns**
   ```python
   # In src/models/enhanced_loader.py
   self.relationship_patterns.update({
       'teaches_ability': r'can teach|learns|grants ability\s+(\w+)',
       'drops_item': r'drops|can drop|leaves behind\s+(\w+)',
       'starts_quest': r'begins|starts|initiates quest\s+(\w+)'
   })
   ```

3. **Implement Relationship-Specific Logic**
   ```python
   def process_ability_relationships(self, entity_id: str, description: str):
       """Process ability-related relationships from description."""
       ability_matches = re.finditer(
           self.relationship_patterns['teaches_ability'],
           description
       )
       
       for match in ability_matches:
           ability_name = match.group(1)
           ability_id = self._get_or_create_entity(
               ability_name,
               'ability'
           )
           if ability_id:
               self.knowledge_graph.add_relationship(
                   entity_id,
                   ability_id,
                   'teaches_ability'
               )
   ```

### Adding New Response Formats

1. **Create Format Templates**
   ```python
   # In src/models/enhanced_chatbot.py
   RESPONSE_TEMPLATES = {
       'quest': """
           # {quest_name}
           
           ## Overview
           {description}
           
           ## Requirements
           - Level: {level_requirement}
           - Prerequisites: {prerequisites}
           
           ## Rewards
           {rewards}
           
           ## Steps
           {steps}
       """,
       'ability': """
           # {ability_name}
           
           ## Description
           {description}
           
           ## Stats
           - Power: {power}
           - Cost: {cost}
           - Cooldown: {cooldown}
           
           ## Learned By
           {learned_by}
       """
   }
   ```

2. **Add Format-Specific Processing**
   ```python
   def format_quest_response(self, quest_data: Dict[str, Any]) -> str:
       """Format quest information using template."""
       return self.RESPONSE_TEMPLATES['quest'].format(
           quest_name=quest_data['name'],
           description=quest_data.get('description', 'No description available'),
           level_requirement=quest_data.get('level_req', 'None'),
           prerequisites=self._format_prerequisites(quest_data.get('prerequisites', [])),
           rewards=self._format_rewards(quest_data.get('rewards', {})),
           steps=self._format_quest_steps(quest_data.get('steps', []))
       )
   ```

### Testing New Capabilities

1. **Add Test Cases**
   ```python
   # In tests/test_components.py
   def test_new_attribute_patterns():
       analyzer = QueryAnalyzer()
       
       # Test rarity patterns
       result = analyzer.analyze("show me legendary hatchy")
       assert result['filters']['rarity'] == 'legendary'
       
       # Test power level patterns
       result = analyzer.analyze("find power level 5 monsters")
       assert result['filters']['power_level'] == '5'
   ```

2. **Add Integration Tests**
   ```python
   def test_quest_system_integration():
       # Test quest data loading
       loader.load_quest_data('test_data/quests.json')
       
       # Test quest retrieval
       response = chatbot.generate_response("tell me about the dragon quest")
       assert 'Requirements' in response
       assert 'Rewards' in response
       
       # Test relationship handling
       quest = knowledge_graph.get_entity_by_name("Dragon Quest")
       relationships = knowledge_graph.get_entity_relationships(quest['id'])
       assert any(r['type'] == 'requires_item' for r in relationships)
   ```

Remember to update the documentation when adding new capabilities and ensure proper error handling and validation for new features. 