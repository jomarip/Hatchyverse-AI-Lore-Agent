import unittest
from pathlib import Path
from models import Monster, Element
from src.data.data_loader import DataLoader
from src.models.lore_entity import LoreEntity

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader("data")
        
    def test_load_monsters(self):
        """Test loading monster data"""
        # Load all data
        entities = self.data_loader.load_all_data()
        
        # Filter for monsters
        monsters = [e for e in entities if e.entity_type == "Monster"]
        
        # Verify we have monsters loaded
        self.assertTrue(len(monsters) > 0)
        
        # Test specific monster (Celestion)
        celestion = next((m for m in monsters if m.name == "Celestion"), None)
        self.assertIsNotNone(celestion)
        self.assertEqual(celestion.element, "Void")
        
    def test_get_monsters_by_element(self):
        """Test filtering monsters by element"""
        entities = self.data_loader.load_all_data()
        monsters = [e for e in entities if e.entity_type == "Monster"]
        
        # Get all Plant type monsters
        plant_monsters = [m for m in monsters if m.element == "Plant"]
        self.assertTrue(len(plant_monsters) > 0)
        for monster in plant_monsters:
            self.assertEqual(monster.element, "Plant")
            
    def test_monster_metadata(self):
        """Test monster metadata and properties"""
        entities = self.data_loader.load_all_data()
        monsters = [e for e in entities if e.entity_type == "Monster"]
        
        # Get first monster
        monster = monsters[0]
        
        self.assertIsNotNone(monster)
        self.assertTrue('height' in monster.metadata)
        self.assertTrue('weight' in monster.metadata)
        self.assertTrue(isinstance(monster.description, str))
        self.assertTrue(len(monster.sources) > 0)

if __name__ == '__main__':
    unittest.main() 