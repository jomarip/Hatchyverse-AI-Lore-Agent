"""Test suite for improved relationship extraction."""

import unittest
import logging
from pathlib import Path
import re
from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader
from src.models.relationship_extractor import AdaptiveRelationshipExtractor
from src.data.cleaners import DataCleaner

class TestRelationshipExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        
        # Initialize components
        cls.knowledge_graph = HatchyKnowledgeGraph()
        cls.data_loader = EnhancedDataLoader(cls.knowledge_graph)
        cls.cleaner = DataCleaner()
        cls.extractor = AdaptiveRelationshipExtractor()
        
        # Register core relationship types
        core_types = [
            ('evolves_from', 'evolves_into'),
            ('member_of', 'has_member'),
            ('allied_with', 'allied_with'),
            ('opposes', 'opposed_by'),
            ('controls', 'controlled_by'),
            ('commands', 'commanded_by'),
            ('mentors', 'mentored_by')  # Add mentors relationship type
        ]
        
        for rel_type, inverse in core_types:
            if rel_type not in cls.knowledge_graph.relationship_registry:
                cls.knowledge_graph.relationship_registry[rel_type] = {
                    'inverse': inverse,
                    'confidence_threshold': 0.8
                }
        
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Test faction name cleaning
        test_cases = [
            ("Royal Army who opposes the rebels", "Royal Army"),
            ("Dark Empire, the evil faction", "Dark Empire"),
            ("Rebels (freedom fighters)", "Rebels"),
            ("The Light Kingdom and its allies", "Light Kingdom")
        ]
        
        for input_name, expected in test_cases:
            cleaned = self.cleaner.clean_faction_name(input_name)
            self.assertEqual(cleaned, expected)
            
        # Test element cleaning
        element_cases = [
            ("fire", "Fire"),
            ("WATER", "Water"),
            ("plant type", "Plant"),
            ("lunar", "Dark"),  # Special case
            ("solar", "Light")  # Special case
        ]
        
        for input_elem, expected in element_cases:
            cleaned = self.cleaner.clean_element_name(input_elem)
            self.assertEqual(cleaned, expected)
            
    def test_relationship_extraction(self):
        """Test relationship extraction from text."""
        # Test political relationship extraction
        political_text = """
        The Dark Empire opposes the Light Kingdom.
        They are allied with the Shadow Guild.
        The Emperor controls the outer regions.
        """
        
        relationships = self.extractor.extract_relationships(political_text)
        
        # Verify relationships were extracted
        self.assertTrue(any(r.type == 'opposes' and r.target == 'Light Kingdom' for r in relationships))
        self.assertTrue(any(r.type == 'allied_with' and r.target == 'Shadow Guild' for r in relationships))
        
        # Verify confidence scores
        for rel in relationships:
            self.assertIn('confidence', rel.__dict__)
            self.assertGreaterEqual(rel.confidence, 0.7)
            
    def test_entity_resolution(self):
        """Test entity resolution and relationship creation."""
        # Create test entities
        monster_data = {
            'name': 'TestMonster',
            'element': 'Fire',
            'faction': 'Test Faction who fights for freedom',
            'Political Conflict': 'opposes the Dark Empire and its allies'
        }
        
        # Add entity and process relationships
        entity_id = self.knowledge_graph.add_entity(
            name=monster_data['name'],
            entity_type='monster',
            attributes=monster_data
        )
        
        # Add element relationship directly
        element_id = self.knowledge_graph.resolve_entity('Fire', 'element')
        self.knowledge_graph.add_relationship(
            source_id=entity_id,
            target_id=element_id,
            relationship_type='has_element',
            metadata={'confidence': 1.0}
        )
        
        # Extract relationships from text
        text = f"{monster_data['name']} is a {monster_data['element']} type monster that {monster_data['Political Conflict']}"
        relationships = self.extractor.extract_relationships(text)
        
        # Add extracted relationships to graph
        for rel in relationships:
            if rel.confidence >= 0.8:
                self.knowledge_graph.add_relationship(
                    source_id=entity_id,
                    target_id=self.knowledge_graph.resolve_entity(rel.target, 'faction'),
                    relationship_type=rel.type,
                    metadata={'confidence': rel.confidence}
                )
        
        # Verify relationships in graph
        entity_rels = self.knowledge_graph.get_relationships(entity_id)
        self.assertTrue(len(entity_rels) > 0)
        
        # Check element relationship
        element_rels = [r for r in entity_rels if r['type'] == 'has_element']
        self.assertTrue(any(r['target_id'] for r in element_rels))
        
        # Check faction relationships
        faction_rels = [r for r in entity_rels if r['type'] in ['member_of', 'opposes']]
        self.assertTrue(len(faction_rels) > 0)
        
    def test_relationship_confidence(self):
        """Test relationship confidence scoring."""
        # Create test entity with various relationship types
        entity_data = {
            'name': 'ConfidenceTest',
            'element': 'Water',  # Should have high confidence
            'Political Conflict': 'maybe opposes the Dark Forces',  # Should have lower confidence
            'Character Description': 'Definitely leads the Water Tribe'  # Should have high confidence
        }
        
        # Add entity to graph
        entity_id = self.knowledge_graph.add_entity(
            name=entity_data['name'],
            entity_type='character',
            attributes=entity_data
        )
        
        # Extract and add relationships
        text = f"{entity_data['name']} is a {entity_data['element']} element character. {entity_data['Political Conflict']}. {entity_data['Character Description']}"
        relationships = self.extractor.extract_relationships(text)
        
        for rel in relationships:
            if rel.confidence >= 0.7:
                self.knowledge_graph.add_relationship(
                    source_id=entity_id,
                    target_id=self.knowledge_graph.resolve_entity(rel.target, 'faction'),
                    relationship_type=rel.type,
                    metadata={'confidence': rel.confidence}
                )
        
        # Get all relationships
        all_rels = self.knowledge_graph.get_relationships(entity_id)
        
        # Check confidence scores
        for rel in all_rels:
            confidence = rel['metadata'].get('confidence', 0)
            
            # Element relationships should have highest confidence
            if rel['type'] == 'has_element':
                self.assertGreaterEqual(confidence, 0.9)
            
            # All relationships should have some confidence
            self.assertGreater(confidence, 0)
            self.assertLessEqual(confidence, 1.0)
            
    def test_pattern_learning(self):
        """Test the pattern learning capability."""
        # Test multiple similar relationships
        test_phrases = [
            "under the tutelage of Master Kael",
            "guided by the tutelage of Elder Lira",
            "mentored through the tutelage of Grand Mage"
        ]
        
        # Process multiple similar relationships
        for text in test_phrases:
            _ = self.extractor.extract_relationships(text)
            
        # Test learned pattern with matching structure
        new_text = "trained under the tutelage of Archmage Velnor"  # Match learned pattern verbs
        relationships = self.extractor.extract_relationships(new_text)
        
        # Check for any mentor relationship regardless of target
        mentor_rels = [r for r in relationships if r.type == 'mentors']
        self.assertTrue(
            len(mentor_rels) > 0,
            f"No mentor relationships found. All relationships: {relationships}"
        )
            
    def test_fallback_entity_creation(self):
        """Test fallback entity creation for unknown targets."""
        # Try to create a relationship with unknown target
        source_id = self.knowledge_graph.add_entity(
            name='SourceEntity',
            entity_type='character',
            attributes={'name': 'SourceEntity'}
        )
        
        rel_id = self.knowledge_graph.add_relationship(
            source_id=source_id,
            target_id='UnknownTarget',
            relationship_type='allied_with'
        )
        
        # Verify relationship was created
        self.assertIsNotNone(rel_id)
        
        # Verify target entity was created
        rel = self.knowledge_graph.relationships[rel_id]
        target_entity = self.knowledge_graph.get_entity(rel['target_id'])
        
        self.assertIsNotNone(target_entity)
        self.assertEqual(target_entity['entity_type'], 'auto_target')
        self.assertTrue(target_entity['attributes'].get('auto_created', False))

if __name__ == '__main__':
    unittest.main() 