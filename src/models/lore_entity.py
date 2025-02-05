from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class EntityReference(BaseModel):
    """Reference to another entity with context"""
    entity_id: str
    relationship_type: str
    context: str
    source_text: str

class LoreEntity(BaseModel):
    """Enhanced base model for all lore elements with better cross-referencing."""
    
    id: str
    name: str
    entity_type: str
    element: Optional[str] = None
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    
    # Enhanced relationship tracking
    references: List[EntityReference] = Field(default_factory=list)
    mentioned_in: List[str] = Field(default_factory=list)  # IDs of entities that mention this one
    context_tags: List[str] = Field(default_factory=list)  # Semantic tags for better retrieval
    
    # Source text fragments that define this entity
    source_fragments: List[Dict[str, str]] = Field(default_factory=list)
    
    def add_reference(self, target_id: str, relationship: str, context: str, source: str):
        """Add a reference to another entity with context."""
        self.references.append(EntityReference(
            entity_id=target_id,
            relationship_type=relationship,
            context=context,
            source_text=source
        ))
    
    def add_source_fragment(self, text: str, source: str, context_type: str):
        """Add a source text fragment with context."""
        self.source_fragments.append({
            "text": text,
            "source": source,
            "context_type": context_type
        })
    
    def add_context_tags(self, tags: List[str]):
        """Add semantic context tags for improved retrieval."""
        self.context_tags.extend(tags)
        self.context_tags = list(set(self.context_tags))  # Remove duplicates

    class Config:
        json_schema_extra = {
            "example": {
                "id": "monster_001",
                "name": "Aquafrost",
                "entity_type": "Monster",
                "element": "Water",
                "description": "A rare water-type Hatchy with ice manipulation abilities.",
                "relationships": {
                    "evolves_from": ["basic_water_hatchy"],
                    "habitat": ["frozen_lakes", "arctic_regions"]
                },
                "sources": ["monster_data_gen1.csv"],
                "metadata": {
                    "rarity": "Rare",
                    "evolution_level": "2",
                    "power_level": "85"
                }
            }
        } 