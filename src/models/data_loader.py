"""Module for loading and processing data from various sources."""

import pandas as pd
import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from thefuzz import fuzz

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class DataLoader:
    """Handles loading and processing data from various sources."""
    
    # Column name mappings for different data types
    COLUMN_MAPS = {
        'monster': {
            'name': ['Name', 'Monster Name', 'MonsterName'],
            'element': ['Element', 'ElementType', 'Type'],
            'description': ['Description', 'Desc', 'Details'],
            'abilities': ['Abilities', 'Skills', 'Powers']
        },
        'item': {
            'name': ['Name', 'Item Name', 'ItemName'],
            'type': ['Type', 'Category', 'ItemType'],
            'element': ['Element', 'ElementType'],
            'rarity': ['Rarity', 'Grade', 'Quality'],
            'description': ['Description', 'Desc', 'Details']
        }
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.monster_data: Dict[str, pd.DataFrame] = {}
        self.item_data: Dict[str, pd.DataFrame] = {}
        self.story_data: Dict[str, Any] = {}
        self.world_data: Dict[str, Any] = {}
    
    def normalize_column_name(self, column: str, data_type: str) -> str:
        """Normalize column names using fuzzy matching."""
        column = column.strip()
        
        # Check exact matches first
        for standard_name, variants in self.COLUMN_MAPS[data_type].items():
            if column in variants or column.lower() == standard_name.lower():
                return standard_name
        
        # Try fuzzy matching if no exact match
        best_match = None
        best_score = 0
        for standard_name, variants in self.COLUMN_MAPS[data_type].items():
            for variant in variants + [standard_name]:
                score = fuzz.ratio(column.lower(), variant.lower())
                if score > best_score and score > 80:  # 80% similarity threshold
                    best_score = score
                    best_match = standard_name
        
        return best_match or column
    
    def validate_required_columns(self, df: pd.DataFrame, required_columns: Set[str], file_name: str):
        """Validate that required columns are present."""
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns in {file_name}: {missing_columns}"
            )
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe values."""
        # Convert to string and strip whitespace for text columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
        
        # Convert empty strings and 'nan' to None
        df = df.replace(r'^\s*$', None, regex=True)
        df = df.replace('nan', None)
        
        return df
    
    def load_all_data(self):
        """Load all data sources."""
        self._load_monster_data()
        self._load_item_data()
        self._load_story_data()
        self._load_world_data()
    
    def _load_monster_data(self):
        """Load monster data from CSV files."""
        monster_files = {
            'gen1': 'Hatchy - Monster Data - gen 1.csv',
            'gen2': 'Hatchy - Monster Data - gen 2.csv',
            'hatchipedia': 'Hatchipedia - monsters.csv',
            'production': 'Hatchy Production Economy - Monster Data.csv'
        }
        
        required_columns = {'name', 'element'}
        
        for key, filename in monster_files.items():
            try:
                file_path = self.data_dir / filename
                if file_path.exists():
                    # Read CSV
                    df = pd.read_csv(file_path)
                    
                    # Normalize column names
                    df.columns = [self.normalize_column_name(col, 'monster') for col in df.columns]
                    
                    # Validate required columns
                    self.validate_required_columns(df, required_columns, filename)
                    
                    # Clean data
                    df = self.clean_dataframe(df)
                    
                    self.monster_data[key] = df
                    logger.info(f"Loaded monster data from {filename}")
            except Exception as e:
                logger.error(f"Error loading monster data from {filename}: {e}")
    
    def _load_item_data(self):
        """Load item data from CSV files."""
        item_files = {
            'equipment': 'PFP-hatchyverse - Masters data - 2.EQUIP Info.csv',
            'items': 'PFP-hatchyverse - Masters data - masters-items-db.csv'
        }
        
        required_columns = {'name', 'type'}
        
        for key, filename in item_files.items():
            try:
                file_path = self.data_dir / filename
                if file_path.exists():
                    # Read CSV
                    df = pd.read_csv(file_path)
                    
                    # Normalize column names
                    df.columns = [self.normalize_column_name(col, 'item') for col in df.columns]
                    
                    # Validate required columns
                    self.validate_required_columns(df, required_columns, filename)
                    
                    # Clean data
                    df = self.clean_dataframe(df)
                    
                    self.item_data[key] = df
                    logger.info(f"Loaded item data from {filename}")
            except Exception as e:
                logger.error(f"Error loading item data from {filename}: {e}")
    
    def _load_story_data(self):
        """Load story data from text files."""
        try:
            story_file = self.data_dir / 'Hatchy World Comic_ Chaos saga.txt'
            if story_file.exists():
                with open(story_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.story_data['chaos_saga'] = self._process_story_content(content)
                logger.info("Loaded story data from Chaos saga")
        except Exception as e:
            logger.error(f"Error loading story data: {e}")
    
    def _load_world_data(self):
        """Load world design data from text files."""
        try:
            world_file = self.data_dir / 'Hatchy World _ world design.txt'
            if world_file.exists():
                with open(world_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.world_data['world_design'] = self._process_world_content(content)
                logger.info("Loaded world design data")
        except Exception as e:
            logger.error(f"Error loading world data: {e}")
    
    def _process_story_content(self, content: str) -> Dict[str, Any]:
        """Process raw story content into structured data."""
        # Split content into chapters/segments
        segments = []
        characters = set()
        locations = set()
        
        current_segment = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for chapter/section headers
            if line.startswith('#') or line.isupper():
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    'title': line.lstrip('#').strip(),
                    'content': '',
                    'characters': set(),
                    'locations': set()
                }
            elif current_segment:
                current_segment['content'] += line + '\n'
                
                # Extract characters and locations
                words = line.split()
                for word in words:
                    if word.istitle() and len(word) > 1:
                        if any(location_type in word for location_type in [
                            'City', 'Village', 'Temple', 'Forest', 'Mountain', 
                            'Castle', 'Tower', 'Cave', 'Palace', 'Shrine'
                        ]):
                            current_segment['locations'].add(word)
                            locations.add(word)
                        else:
                            current_segment['characters'].add(word)
                            characters.add(word)
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        # Convert sets to lists for JSON serialization
        for segment in segments:
            segment['characters'] = list(segment['characters'])
            segment['locations'] = list(segment['locations'])
        
        return {
            'title': 'Chaos Saga',
            'segments': segments,
            'characters': list(characters),
            'locations': list(locations)
        }
    
    def _process_world_content(self, content: str) -> Dict[str, Any]:
        """Process raw world design content into structured data."""
        sections = content.split('\n\n')
        
        processed_data = {
            'elements': {},
            'locations': {},
            'regions': {},
            'biomes': set(),
            'landmarks': set()
        }
        
        current_element = None
        current_region = None
        
        for section in sections:
            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Process element sections
                if line.startswith('* ') and 'Theme:' in line:
                    element_name = line.split('Theme:')[0].strip('* ').strip()
                    theme = line.split('Theme:')[1].strip()
                    current_element = element_name
                    processed_data['elements'][element_name] = {
                        'theme': theme,
                        'motifs': [],
                        'affinity': ''
                    }
                elif current_element and line.startswith('* Motifs:'):
                    motifs = [m.strip() for m in line.split('Motifs:')[1].split(',')]
                    processed_data['elements'][current_element]['motifs'].extend(motifs)
                elif current_element and line.startswith('* Life Affinity:'):
                    affinity = line.split('Life Affinity:')[1].strip()
                    processed_data['elements'][current_element]['affinity'] = affinity
                
                # Process regions
                elif line.startswith(str(len(processed_data['regions']) + 1) + '.'):
                    current_region = line.split('.')[1].split(':')[0].strip()
                    processed_data['regions'][current_region] = {
                        'name': current_region,
                        'locations': [],
                        'landmarks': []
                    }
                elif current_region and line.startswith('* '):
                    location = line.strip('* ').strip()
                    processed_data['regions'][current_region]['locations'].append(location)
                    processed_data['biomes'].add(location)
                elif current_region and line.startswith('- '):
                    landmark = line.strip('- ').strip()
                    processed_data['regions'][current_region]['landmarks'].append(landmark)
                    processed_data['landmarks'].add(landmark)
                
                # Process standalone locations
                elif line.startswith('* ') and not any(keyword in line for keyword in ['Theme:', 'Motifs:', 'Life Affinity:']):
                    location = line.strip('* ').strip()
                    processed_data['biomes'].add(location)
                elif line.startswith('- '):
                    landmark = line.strip('- ').strip()
                    processed_data['landmarks'].add(landmark)
        
        # Convert sets to lists
        processed_data['biomes'] = list(processed_data['biomes'])
        processed_data['landmarks'] = list(processed_data['landmarks'])
        
        return processed_data
    
    def get_character_data(self) -> Dict[str, Any]:
        """Extract and combine character data from all sources."""
        characters = {}
        
        # Extract characters from monster data
        for source, df in self.monster_data.items():
            if 'name' in df.columns:
                for _, row in df.iterrows():
                    name = row['name']
                    if pd.isna(name):
                        continue
                    if name not in characters:
                        characters[name] = {
                            'name': name,
                            'type': 'monster',
                            'element': row.get('element', 'unknown'),
                            'source': source,
                            'appearances': [],
                            'abilities': []
                        }
        
        # Extract characters from story data
        if 'chaos_saga' in self.story_data:
            for char_name in self.story_data['chaos_saga']['characters']:
                if char_name not in characters:
                    characters[char_name] = {
                        'name': char_name,
                        'type': 'story_character',
                        'appearances': ['chaos_saga'],
                        'locations': [],
                        'relationships': []
                    }
        
        return characters
    
    def get_timeline_data(self) -> List[Dict[str, Any]]:
        """Extract and combine timeline data from all sources."""
        timeline = []
        
        # Extract timeline events from story data
        if 'chaos_saga' in self.story_data:
            for segment in self.story_data['chaos_saga']['segments']:
                timeline.append({
                    'name': segment['title'],
                    'type': 'story_event',
                    'source': 'chaos_saga',
                    'characters': segment['characters'],
                    'locations': segment['locations'],
                    'description': segment['content'][:200] + '...' if len(segment['content']) > 200 else segment['content']
                })
        
        return timeline
    
    def get_location_data(self) -> Dict[str, Any]:
        """Extract and combine location data from all sources."""
        locations = {}
        
        # Extract locations from world data
        if 'world_design' in self.world_data:
            # Add biomes
            for biome in self.world_data['world_design']['biomes']:
                locations[biome] = {
                    'name': biome,
                    'type': 'biome',
                    'landmarks': [],
                    'characters': [],
                    'events': []
                }
            
            # Add landmarks
            for landmark in self.world_data['world_design']['landmarks']:
                locations[landmark] = {
                    'name': landmark,
                    'type': 'landmark',
                    'parent_biome': None,
                    'characters': [],
                    'events': []
                }
        
        # Add locations from story data
        if 'chaos_saga' in self.story_data:
            for location in self.story_data['chaos_saga']['locations']:
                if location not in locations:
                    locations[location] = {
                        'name': location,
                        'type': 'story_location',
                        'appearances': ['chaos_saga'],
                        'characters': [],
                        'events': []
                    }
        
        return locations 