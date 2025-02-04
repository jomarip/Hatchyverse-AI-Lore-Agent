import pandas as pd
from typing import List, Dict, Any
import os
from ..models.lore_entity import LoreEntity
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of Hatchyverse data files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_monsters(self, file_path: str) -> List[LoreEntity]:
        """
        Loads monster data from CSV file.
        
        Args:
            file_path: Path to the monsters CSV file
            
        Returns:
            List of LoreEntity objects
        """
        try:
            df = pd.read_csv(file_path)
            monsters = []
            
            for _, row in df.iterrows():
                monster = LoreEntity(
                    id=f"monster_{row['Monster ID']}",
                    name=row['Name'],
                    entity_type="Monster",
                    element=row['Element'],
                    description=row['Description'],
                    metadata={
                        "evolution_level": str(row.get('evolution level', '')),
                        "rarity": str(row.get('rarity', '')),
                        "power_level": str(row.get('power_level', ''))
                    },
                    sources=[os.path.basename(file_path)]
                )
                monsters.append(monster)
                
            logger.info(f"Loaded {len(monsters)} monsters from {file_path}")
            return monsters
            
        except Exception as e:
            logger.error(f"Error loading monsters from {file_path}: {str(e)}")
            raise
            
    def load_items(self, file_path: str) -> List[LoreEntity]:
        """
        Loads item data from CSV file.
        
        Args:
            file_path: Path to the items CSV file
            
        Returns:
            List of LoreEntity objects
        """
        try:
            df = pd.read_csv(file_path)
            items = []
            
            for _, row in df.iterrows():
                if pd.notna(row.get('description', '')):
                    item = LoreEntity(
                        id=f"item_{row['id']}",
                        name=row['name'],
                        entity_type="Item",
                        description=row['description'],
                        metadata={
                            "category": str(row.get('category_id', '')),
                            "rarity": str(row.get('rarity', '')),
                            "type": str(row.get('type', ''))
                        },
                        sources=[os.path.basename(file_path)]
                    )
                    items.append(item)
                    
            logger.info(f"Loaded {len(items)} items from {file_path}")
            return items
            
        except Exception as e:
            logger.error(f"Error loading items from {file_path}: {str(e)}")
            raise
            
    def load_world_data(self, file_path: str) -> List[LoreEntity]:
        """
        Loads world building data from text file.
        
        Args:
            file_path: Path to the world design text file
            
        Returns:
            List of LoreEntity objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split content into sections (this is a simple example - adjust based on actual format)
            sections = content.split('\n\n')
            locations = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    location = LoreEntity(
                        id=f"location_{i}",
                        name=f"Location {i}",  # You might want to extract actual names
                        entity_type="Location",
                        description=section.strip(),
                        sources=[os.path.basename(file_path)]
                    )
                    locations.append(location)
                    
            logger.info(f"Loaded {len(locations)} locations from {file_path}")
            return locations
            
        except Exception as e:
            logger.error(f"Error loading world data from {file_path}: {str(e)}")
            raise
            
    def load_all_data(self) -> List[LoreEntity]:
        """
        Loads all available data files.
        
        Returns:
            Combined list of all LoreEntity objects
        """
        all_entities = []
        
        # Load monsters if file exists
        monster_path = os.path.join(self.data_dir, "monsters.csv")
        if os.path.exists(monster_path):
            all_entities.extend(self.load_monsters(monster_path))
            
        # Load items if file exists
        items_path = os.path.join(self.data_dir, "items.csv")
        if os.path.exists(items_path):
            all_entities.extend(self.load_items(items_path))
            
        # Load world data if file exists
        world_path = os.path.join(self.data_dir, "world_design.txt")
        if os.path.exists(world_path):
            all_entities.extend(self.load_world_data(world_path))
            
        logger.info(f"Loaded total of {len(all_entities)} entities")
        return all_entities 