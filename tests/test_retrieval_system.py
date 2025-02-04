"""Test suite for the retrieval system components."""

import pytest
import shutil
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.data.data_loader import DataLoader
from src.models.retriever_factory import RetrieverFactory
from src.models.chatbot import FilteredRetriever

# Test data fixtures
@pytest.fixture
def test_data_dir(tmp_path):
    """Create a test directory with copies of actual data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Source data files to copy
    source_files = [
        "Hatchy - Monster Data - gen 1.csv",
        "Hatchy - Monster Data - gen 2.csv",
        "Hatchy Production Economy - Monster Data.csv",
        "PFP-hatchyverse - Masters data - 2.EQUIP Info.csv",
        "PFP-hatchyverse - Masters data - masters-items-db.csv",
        "Hatchy World Comic_ Chaos saga.txt",
        "Hatchy World _ world design.txt"
    ]
    
    # Get the data directory path
    workspace_root = Path(__file__).parent.parent
    source_data_dir = workspace_root / "data"
    
    # Copy each file to the test directory
    for file_name in source_files:
        source_path = source_data_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, data_dir / file_name)
        else:
            print(f"Warning: Source file {file_name} not found in {source_data_dir}")
    
    return data_dir

@pytest.fixture
def mock_base_retriever():
    """Create a mock base retriever."""
    retriever = Mock(spec=BaseRetriever)
    retriever.invoke.return_value = [
        Document(
            page_content="Test content",
            metadata={"source": "test"}
        )
    ]
    return retriever

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.similarity_search.return_value = [
        Document(
            page_content="Test vector content",
            metadata={"source": "vector"}
        )
    ]
    return store

class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_load_monster_data(self, test_data_dir):
        """Test loading monster data from CSV."""
        loader = DataLoader(test_data_dir)
        loader._load_monster_data()
        
        # Test with actual data
        assert 'gen1' in loader.monster_data
        assert 'gen2' in loader.monster_data
        assert 'production' in loader.monster_data
        
        # Verify some data was loaded
        for source, df in loader.monster_data.items():
            assert not df.empty, f"Monster data for {source} is empty"
            assert 'name' in df.columns, f"Monster data for {source} missing 'name' column"
    
    def test_load_item_data(self, test_data_dir):
        """Test loading item data from CSV."""
        loader = DataLoader(test_data_dir)
        loader._load_item_data()
        
        # Test with actual data
        assert 'equipment' in loader.item_data
        assert 'items' in loader.item_data
        
        # Verify some data was loaded
        for source, df in loader.item_data.items():
            assert not df.empty, f"Item data for {source} is empty"
            assert 'name' in df.columns, f"Item data for {source} missing 'name' column"
    
    def test_process_story_content(self, test_data_dir):
        """Test processing story content."""
        loader = DataLoader(test_data_dir)
        loader._load_story_data()
        
        assert 'chaos_saga' in loader.story_data
        saga_data = loader.story_data['chaos_saga']
        
        # Test with actual data
        assert len(saga_data['segments']) > 0, "No story segments loaded"
        assert len(saga_data['characters']) > 0, "No characters extracted"
        assert len(saga_data['locations']) > 0, "No locations extracted"
    
    def test_process_world_content(self, test_data_dir):
        """Test processing world design content."""
        loader = DataLoader(test_data_dir)
        loader._load_world_data()
        
        assert 'world_design' in loader.world_data
        world_data = loader.world_data['world_design']
        
        # Test with actual data
        assert len(world_data['elements']) > 0, "No elements loaded"
        assert len(world_data['biomes']) > 0, "No biomes loaded"
        assert len(world_data['landmarks']) > 0, "No landmarks loaded"

class TestFilteredRetriever:
    """Test cases for FilteredRetriever class."""
    
    def test_character_query(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test character-specific query handling."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        # Test with actual character names from your data
        results = retriever.invoke("Who is FireDragon?")
        assert len(results) > 0
        
        # Test with another character
        results = retriever.invoke("Tell me about WaterSprite")
        assert len(results) > 0
    
    def test_location_query(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test location-specific query handling."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        # Test with actual locations from your data
        results = retriever.invoke("Tell me about Fire Temple")
        assert len(results) > 0
        
        # Test with another location
        results = retriever.invoke("What is in the Water City?")
        assert len(results) > 0
    
    def test_item_query(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test item-specific query handling."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        # Test with actual items from your data
        results = retriever.invoke("What is Flame Sword?")
        assert len(results) > 0
        
        # Test with another item
        results = retriever.invoke("Tell me about Ice Shield")
        assert len(results) > 0
    
    def test_story_query(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test story-specific query handling."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        # Test with actual story content
        results = retriever.invoke("Tell me about the Chaos saga")
        assert len(results) > 0
        
        # Test with specific chapter
        results = retriever.invoke("What happens in Chapter 1?")
        assert len(results) > 0

class TestRetrieverFactory:
    """Test cases for RetrieverFactory class."""
    
    def test_create_filtered_retriever(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test creation and configuration of FilteredRetriever."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        assert isinstance(retriever, FilteredRetriever)
        assert len(retriever.character_store) > 0
        assert len(retriever.item_store["element"]) > 0
        assert len(retriever.story_store) > 0
    
    def test_populate_stores(self, test_data_dir, mock_base_retriever, mock_vector_store):
        """Test population of data stores."""
        factory = RetrieverFactory(test_data_dir)
        retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
        
        # Test with actual data
        assert len(retriever.character_store) > 0, "No characters loaded"
        assert len(retriever.item_store['element']) > 0, "No elemental items loaded"
        assert len(retriever.story_store) > 0, "No story content loaded"
        assert len(retriever.timeline_store) > 0, "No timeline events loaded"

def test_end_to_end(test_data_dir, mock_base_retriever, mock_vector_store):
    """Test the entire retrieval pipeline with actual data."""
    factory = RetrieverFactory(test_data_dir)
    retriever = factory.create_filtered_retriever(mock_base_retriever, mock_vector_store)
    
    # Test different types of queries with actual content
    test_queries = [
        # Character queries
        "Who is FireDragon?",
        "Tell me about WaterSprite",
        # Location queries
        "What can I find in the Fire Temple?",
        "Tell me about Water City",
        # Item queries
        "What is the Flame Sword?",
        "Tell me about Ice Shield",
        # Story queries
        "What happens in the Chaos saga?",
        "Tell me about Chapter 1"
    ]
    
    for query in test_queries:
        results = retriever.invoke(query)
        assert len(results) > 0, f"No results found for query: {query}"
        assert any(len(doc.page_content.strip()) > 0 for doc in results), f"Empty results for query: {query}" 