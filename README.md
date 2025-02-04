# Hatchyverse Community Lore Chatbot

An AI-powered chatbot system for exploring, validating, and contributing to the Hatchyverse lore. This system helps maintain consistency in the expanding Hatchyverse universe while enabling community participation in lore creation.

## Features

- **Interactive Lore Exploration**: Users can ask questions about any aspect of Hatchyverse lore
- **Semantic Search**: Advanced search capabilities using embeddings to find relevant lore
- **Lore Validation**: Automatic checking of new submissions against existing canon
- **Multi-Model Support**: Compatible with multiple AI providers (OpenAI, Anthropic, Deepseek)
- **Conflict Detection**: Identifies potential contradictions with existing lore
- **Constructive Feedback**: Provides suggestions for improving submissions

## Architecture

### Core Components

1. **LoreEntity (`src/models/lore_entity.py`)**
   - Base model for all lore elements (monsters, items, locations, etc.)
   - Handles metadata, relationships, and validation timestamps
   - Supports rich entity relationships and hierarchies

2. **LoreValidator (`src/models/lore_validator.py`)**
   - Manages the vector store for semantic search
   - Performs conflict detection using similarity scoring
   - Provides detailed validation feedback
   - Features adaptive threshold optimization

3. **LoreChatbot (`src/models/chatbot.py`)**
   - Handles user interactions and query processing
   - Supports multiple AI model providers
   - Maintains conversation context
   - Processes both queries and submissions

4. **DataLoader (`src/data/data_loader.py`)**
   - Processes various data formats (CSV, TXT)
   - Loads and normalizes existing lore
   - Handles different entity types:
     - Monsters
     - Items
     - World/Location data

### API Endpoints (`src/api/main.py`)

- **/chat**: Handle general lore queries
  - POST: Accept user messages
  - Returns: AI response with relevant lore

- **/submit**: Process new lore submissions
  - POST: Accept structured lore entries
  - Returns: Validation results and feedback

- **/health**: System health monitoring
  - GET: Check system status
  - Returns: Component health information

## Setup

1. **Environment Setup**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd hatchyverse-lore-chatbot

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Unix
   # or
   .\venv\Scripts\activate  # Windows

   # Install dependencies
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

## Usage

### Querying Lore
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about water-type Hatchies",
    "chat_history": []
  }'
```

### Submitting New Lore
```bash
curl -X POST "http://localhost:8000/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Frostflame",
    "entity_type": "Monster",
    "element": "Ice",
    "content": "A rare Hatchy that can control both ice and fire."
  }'
```

## Data Models

### LoreEntity Structure
```python
{
    "id": "monster_001",
    "name": "Aquafrost",
    "entity_type": "Monster",
    "element": "Water",
    "description": "A rare water-type Hatchy with ice abilities",
    "relationships": {
        "evolves_from": ["basic_water_hatchy"],
        "habitat": ["frozen_lakes"]
    },
    "metadata": {
        "rarity": "Rare",
        "evolution_level": "2"
    }
}
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

3. **Feedback Generation**
   - Provides specific improvement suggestions
   - Highlights related existing lore
   - Explains any detected conflicts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Acknowledgments

- Hatchyverse Community
- LangChain Framework
- FastAPI 