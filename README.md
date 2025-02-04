# Hatchyverse Community Lore Management System

An AI-powered system for exploring, managing, validating, and contributing to the Hatchyverse lore. This system helps maintain consistency in the expanding Hatchyverse universe while enabling community participation in lore creation.

## Features

### Core Functionality
- **Interactive Lore Exploration**: Users can ask questions about any aspect of Hatchyverse lore
- **Semantic Search**: Advanced search capabilities using embeddings to find relevant lore
- **Lore Validation**: Automatic checking of new submissions against existing canon
- **Multi-Model Support**: Compatible with multiple AI providers (OpenAI, Anthropic, Deepseek)
- **Conflict Detection**: Identifies potential contradictions with existing lore
- **Constructive Feedback**: Provides suggestions for improving submissions

### Data Management
- Load and manage monster data across multiple generations
- Support for items, equipment, and world design data
- Type-safe attributes with proper validation
- Asset path management for images and sounds
- Comprehensive test coverage

## Project Structure

```
hatchyverse/
├── data/                    # Data files
│   ├── Hatchy - Monster Data - gen 1.csv
│   ├── Hatchy - Monster Data - gen 2.csv
│   ├── PFP-hatchyverse - Masters data/
│   └── Hatchy World _ world design.txt
├── models/                  # Core data models
│   ├── __init__.py
│   ├── monster.py          # Monster class definition
│   ├── data_loader.py      # Data loading functionality
│   ├── lore_entity.py      # Base model for all lore elements
│   └── lore_validator.py   # Lore validation logic
├── api/                    # API endpoints
│   └── main.py            # FastAPI routes
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_monster_loader.py
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Setup and Configuration

1. **Environment Setup**

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

### Basic Data Access
```python
from models import DataLoader, Element

# Create a data loader instance
loader = DataLoader()

# Load monster data
loader.load_gen1_monsters()
loader.load_gen2_monsters()

# Get a specific monster by ID
celestion = loader.get_monster_by_id(0)
print(celestion)  # Celestion (Void) - ID: 0

# Get all monsters of a specific element
plant_monsters = loader.get_monsters_by_element(Element.PLANT)
for monster in plant_monsters:
    print(monster.name)
```

### Lore Exploration API
```python
from api.main import lore_chatbot

# Ask questions about the lore
response = lore_chatbot.ask("Tell me about water-type Hatchies")

# Submit new lore for validation
submission = {
    "name": "Frostflame",
    "entity_type": "Monster",
    "element": "Ice",
    "description": "A rare Hatchy that can control both ice and fire."
}
validation_result = lore_chatbot.validate_submission(submission)
```

### API Endpoints

- **/chat**: Handle general lore queries
  - POST: Accept user messages
  - Returns: AI response with relevant lore

- **/submit**: Process new lore submissions
  - POST: Accept structured lore entries
  - Returns: Validation results and feedback

- **/health**: System health monitoring
  - GET: Check system status
  - Returns: Component health information

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

## Testing

Run the tests using pytest:
```bash
pytest tests/
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