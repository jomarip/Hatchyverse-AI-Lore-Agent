from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class LoreEntity(BaseModel):
    """Base model for all lore entities in the Hatchyverse."""
    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of entity (Character/Location/Item/Event/Monster)")
    element: Optional[str] = Field(None, description="Element type if applicable")
    description: str = Field(..., description="Detailed description of the entity")
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Related entities and their relationship types"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source documents where this entity is referenced"
    )
    last_validated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last validation"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata specific to entity type"
    )

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