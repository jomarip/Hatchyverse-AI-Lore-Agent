"""Factory module for creating and configuring retrievers."""

from typing import Optional
from langchain_core.retrievers import BaseRetriever
from .data_loader import DataLoader
from .chatbot import FilteredRetriever
import logging

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """Factory class for creating and configuring retrievers with data sources."""
    
    def __init__(self, data_dir: str):
        self.data_loader = DataLoader(data_dir)
    
    def create_filtered_retriever(
        self,
        base_retriever: BaseRetriever,
        vector_store: Optional[any] = None
    ) -> FilteredRetriever:
        """Create and configure a FilteredRetriever with loaded data."""
        try:
            # Load all data
            self.data_loader.load_all_data()
            
            # Initialize the filtered retriever
            retriever = FilteredRetriever(
                base_retriever=base_retriever,
                vector_store=vector_store,
                item_store={},
                character_store={},
                timeline_store={},
                story_store={}
            )
            
            # Populate the stores with processed data
            self._populate_item_store(retriever)
            self._populate_character_store(retriever)
            self._populate_timeline_store(retriever)
            self._populate_story_store(retriever)
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating filtered retriever: {e}")
            raise
    
    def _populate_item_store(self, retriever: FilteredRetriever):
        """Populate the item store with processed item data."""
        # Initialize categories
        retriever.item_store = {
            "element": {},
            "type": {},
            "rarity": {}
        }
        
        # Process equipment data
        if 'equipment' in self.data_loader.item_data:
            equipment_df = self.data_loader.item_data['equipment']
            for _, row in equipment_df.iterrows():
                item_data = {
                    "name": row.get('name', ''),
                    "type": "equipment",
                    "element": row.get('element', 'none'),
                    "rarity": row.get('rarity', 'common'),
                    "effects": [],
                    "description": row.get('description', ''),
                    "metadata": {}
                }
                
                # Add to element category
                if item_data["element"] not in retriever.item_store["element"]:
                    retriever.item_store["element"][item_data["element"]] = []
                retriever.item_store["element"][item_data["element"]].append(item_data)
                
                # Add to type category
                if "equipment" not in retriever.item_store["type"]:
                    retriever.item_store["type"]["equipment"] = []
                retriever.item_store["type"]["equipment"].append(item_data)
                
                # Add to rarity category
                if item_data["rarity"] not in retriever.item_store["rarity"]:
                    retriever.item_store["rarity"][item_data["rarity"]] = []
                retriever.item_store["rarity"][item_data["rarity"]].append(item_data)
        
        # Process general items data
        if 'items' in self.data_loader.item_data:
            items_df = self.data_loader.item_data['items']
            for _, row in items_df.iterrows():
                item_data = {
                    "name": row.get('name', ''),
                    "type": "consumable",  # Default type, adjust based on actual data
                    "element": row.get('element', 'none'),
                    "rarity": row.get('rarity', 'common'),
                    "effects": [],
                    "description": row.get('description', ''),
                    "metadata": {}
                }
                
                # Add to categories (similar to equipment)
                for category in ["element", "type", "rarity"]:
                    key = item_data[category]
                    if key not in retriever.item_store[category]:
                        retriever.item_store[category][key] = []
                    retriever.item_store[category][key].append(item_data)
    
    def _populate_character_store(self, retriever: FilteredRetriever):
        """Populate the character store with processed character data."""
        character_data = self.data_loader.get_character_data()
        retriever.character_store = character_data
    
    def _populate_timeline_store(self, retriever: FilteredRetriever):
        """Populate the timeline store with processed timeline data."""
        timeline_data = self.data_loader.get_timeline_data()
        retriever.timeline_store = {
            str(i): event for i, event in enumerate(timeline_data)
        }
    
    def _populate_story_store(self, retriever: FilteredRetriever):
        """Populate the story store with processed story data."""
        # Add story segments from chaos saga
        if 'chaos_saga' in self.data_loader.story_data:
            saga_data = self.data_loader.story_data['chaos_saga']
            retriever.story_store = {
                'chaos_saga': {
                    'title': saga_data['title'],
                    'segments': saga_data['segments'],
                    'characters': saga_data['characters'],
                    'locations': saga_data['locations']
                }
            } 