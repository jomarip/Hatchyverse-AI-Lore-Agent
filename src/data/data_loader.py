import pandas as pd
from typing import List, Dict, Any, Optional
import os
from ..models.lore_entity import LoreEntity
import logging
import glob
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of Hatchyverse data files."""
    
    # Column name mappings for different file formats
    MONSTER_COLUMNS = {
        'name': ['Name', 'name', 'monster_name'],
        'element': ['Element', 'element', 'type'],
        'description': ['Description', 'description', 'desc'],
        'id': ['Monster ID', 'id', 'monster_id'],
        'height': ['Height', 'height'],
        'weight': ['Weight', 'weight']
    }
    
    ITEM_COLUMNS = {
        'name': ['Name', 'name', 'item_name'],
        'type': ['Type', 'type', 'category'],
        'description': ['Description', 'description', 'desc'],
        'id': ['ID name', 'id', 'item_id'],
        'rarity': ['Rarity', 'rarity', 'grade']
    }
    
    def __init__(self, data_dir: str):
        """Initialize the DataLoader with a data directory."""
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        logger.debug(f"DataLoader initialized with directory: {data_dir}")
        logger.debug(f"Data directory contents: {os.listdir(data_dir)}")
        
        self.monster_data = {}
        self.item_data = {}
        self.story_data = {}
        self.world_data = {}
        
    def _get_column_value(self, row: pd.Series, column_mappings: List[str], default: str = '') -> str:
        """Helper to get column value using multiple possible names."""
        for col in column_mappings:
            if col in row and pd.notna(row[col]):
                return str(row[col])
        return default
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], source: str) -> bool:
        """Validate that DataFrame has required columns."""
        missing_columns = []
        for col_group in required_columns:
            if not any(col in df.columns for col in self.MONSTER_COLUMNS.get(col_group, [])):
                missing_columns.append(col_group)
        
        if missing_columns:
            logger.warning(f"Missing required columns in {source}: {missing_columns}")
            return False
        return True
        
    def _process_monsters(self, df: pd.DataFrame, source: str) -> List[LoreEntity]:
        """Process monster DataFrame into LoreEntity objects."""
        entities = []
        
        for idx, row in df.iterrows():
            try:
                # Get required fields
                name = self._get_column_value(row, self.MONSTER_COLUMNS['name'])
                element = self._get_column_value(row, self.MONSTER_COLUMNS['element'])
                description = self._get_column_value(row, self.MONSTER_COLUMNS['description'])
                monster_id = self._get_column_value(row, self.MONSTER_COLUMNS['id'])
                
                if not name:
                    logger.warning(f"Skipping monster in {source} row {idx}: Missing name")
                    continue
                
                # Create metadata dictionary
                metadata = {
                    'height': self._get_column_value(row, self.MONSTER_COLUMNS['height']),
                    'weight': self._get_column_value(row, self.MONSTER_COLUMNS['weight'])
                }
                
                # Remove None values from metadata
                metadata = {k: v for k, v in metadata.items() if v}
                
                # Create entity
                entity = LoreEntity(
                    id=monster_id if monster_id else f"monster_{idx}",
                    name=name,
                    entity_type="Monster",
                    element=element if element else None,
                    description=description if description else f"A {element} type monster",
                    metadata=metadata,
                    sources=[source]
                )
                
                entities.append(entity)
                logger.debug(f"Added monster: {name} ({monster_id})")
            except Exception as e:
                logger.error(f"Error processing monster in {source} row {idx}: {str(e)}", exc_info=True)
                continue
        
        return entities
        
    def _process_items(self, df: pd.DataFrame, source: str) -> List[LoreEntity]:
        """Process item DataFrame into LoreEntity objects."""
        if df.empty:
            logger.warning(f"Empty item DataFrame from {source}")
            return []
            
        entities = []
        logger.debug(f"Processing {len(df)} items from {source}")
        logger.debug(f"Columns in {source}: {df.columns.tolist()}")
        
        if not self._validate_dataframe(df, ['name'], source):
            return []
            
        for idx, row in df.iterrows():
            try:
                name = self._get_column_value(row, self.ITEM_COLUMNS['name'])
                if not name:
                    logger.debug(f"Skipping item row {idx} without name in {source}")
                    continue
                    
                item_id = self._get_column_value(row, self.ITEM_COLUMNS['id'], f"item_{idx}")
                item_type = self._get_column_value(row, self.ITEM_COLUMNS['type'], 'Unknown')
                description = self._get_column_value(row, self.ITEM_COLUMNS['description'], f"A {item_type} item")
                
                metadata = {
                    'type': item_type,
                    'rarity': self._get_column_value(row, self.ITEM_COLUMNS['rarity'], 'common'),
                    'source': source
                }
                
                # Add any additional metadata columns
                for col in df.columns:
                    if not any(col in mapping for mapping in self.ITEM_COLUMNS.values()):
                        if pd.notna(row[col]):
                            metadata[col.lower()] = str(row[col])
                
                entity = LoreEntity(
                    id=item_id,
                    name=name,
                    entity_type="Item",
                    description=description,
                    metadata=metadata,
                    sources=[source]
                )
                entities.append(entity)
                logger.debug(f"Added item: {name} ({item_id})")
            except Exception as e:
                logger.error(f"Error processing item in {source} row {idx}: {str(e)}", exc_info=True)
                continue
        
        return entities
        
    def _load_monster_data(self):
        """Load monster data from CSV files."""
        monster_file_patterns = [
            "Hatchy - Monster Data - gen *.csv",
            "Hatchy Production Economy - Monster Data.csv"
        ]
        
        logger.debug(f"Searching for monster data files with patterns: {monster_file_patterns}")
        for pattern in monster_file_patterns:
            pattern_path = os.path.join(self.data_dir, pattern)
            logger.debug(f"Searching with pattern: {pattern_path}")
            files = glob.glob(pattern_path)  # Removed recursive=True as files are in root
            logger.debug(f"Found {len(files)} files matching pattern '{pattern}': {files}")
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    filename = os.path.basename(file_path)
                    self.monster_data[filename] = df
                    logger.info(f"Loaded monster data from {filename} with {len(df)} rows")
                    logger.debug(f"Columns in {filename}: {list(df.columns)}")
                except Exception as e:
                    logger.error(f"Error loading monster data from {file_path}: {str(e)}", exc_info=True)
                    
    def _load_item_data(self):
        """Load item data from CSV files."""
        item_file_patterns = [
            "PFP-hatchyverse - Masters data - *.csv",
            "Gen3 List  - Asset list .csv"
        ]
        
        logger.debug(f"Searching for item data files with patterns: {item_file_patterns}")
        for pattern in item_file_patterns:
            pattern_path = os.path.join(self.data_dir, pattern)
            logger.debug(f"Searching with pattern: {pattern_path}")
            files = glob.glob(pattern_path)  # Removed recursive=True as files are in root
            logger.debug(f"Found {len(files)} files matching pattern '{pattern}': {files}")
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    filename = os.path.basename(file_path)
                    self.item_data[filename] = df
                    logger.info(f"Loaded item data from {filename} with {len(df)} rows")
                    logger.debug(f"Columns in {filename}: {list(df.columns)}")
                except Exception as e:
                    logger.error(f"Error loading item data from {file_path}: {str(e)}", exc_info=True)
                
    def _load_story_data(self):
        """Load story data from text files."""
        story_file_patterns = [
            "Hatchy World Comic_ Chaos saga.txt"
        ]
        
        logger.debug(f"Searching for story data files with patterns: {story_file_patterns}")
        for pattern in story_file_patterns:
            pattern_path = os.path.join(self.data_dir, pattern)
            logger.debug(f"Searching with pattern: {pattern_path}")
            files = glob.glob(pattern_path)  # Removed recursive=True as files are in root
            logger.debug(f"Found {len(files)} files matching pattern '{pattern}': {files}")
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    filename = os.path.basename(file_path)
                    self.story_data[filename] = {
                        'segments': self._parse_story_content(content)
                    }
                    logger.info(f"Loaded story data from {filename}")
                except Exception as e:
                    logger.error(f"Error loading story data from {file_path}: {str(e)}", exc_info=True)
                
    def _load_world_data(self):
        """Load world design data from text files."""
        world_file_patterns = [
            "Hatchy World _ world design.txt",
            "Hatchyverse Eco Presentation*.txt"
        ]
        
        logger.debug(f"Searching for world data files with patterns: {world_file_patterns}")
        for pattern in world_file_patterns:
            pattern_path = os.path.join(self.data_dir, pattern)
            logger.debug(f"Searching with pattern: {pattern_path}")
            files = glob.glob(pattern_path)  # Removed recursive=True as files are in root
            logger.debug(f"Found {len(files)} files matching pattern '{pattern}': {files}")
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    filename = os.path.basename(file_path)
                    parsed_content = self._parse_world_content(content)
                    logger.debug(f"Parsed world content: {len(parsed_content.get('elements', {}))} elements, "
                               f"{len(parsed_content.get('regions', {}))} regions, "
                               f"{len(parsed_content.get('landmarks', []))} landmarks, "
                               f"{len(parsed_content.get('lore', []))} lore entries")
                    self.world_data[filename] = parsed_content
                    logger.info(f"Loaded world design data from {filename}")
                except Exception as e:
                    logger.error(f"Error loading world data from {file_path}: {str(e)}", exc_info=True)
                    
    def _parse_story_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse story content into segments with enhanced semantic understanding."""
        segments = []
        current_segment = None
        
        # Enhanced keywords for better concept detection
        concept_keywords = {
            'evolution': ['evolve', 'evolution', 'transform', 'stage', 'form'],
            'ability': ['ability', 'power', 'skill', 'attack', 'technique'],
            'location': ['found in', 'located', 'habitat', 'region', 'area', 'omniterra', 'felkyn'],  # Added specific locations
            'relationship': ['friend', 'enemy', 'ally', 'companion', 'rival', 'mentor'],
            'mechanics': ['ride', 'mount', 'equip', 'use', 'activate', 'summon'],
            'rarity': ['rare', 'legendary', 'mythical', 'unique', 'special', 'ancient'],
            'world': ['world', 'omniterra', 'continent', 'realm', 'dimension', 'felkyn', 'chaos'],  # Enhanced world concepts
            'lore': ['prophecy', 'legend', 'myth', 'history', 'tale', 'saga']  # Added lore concepts
        }
        
        # Enhanced section detection
        section_markers = {
            'chapter': r'Chapter\s+\d+|Episode\s+\d+',
            'arc': r'Arc\s*\d*:.*|Saga:.*',
            'scene': r'Scene\s*\d*:.*|\*\*\*|\-{3,}',
            'location': r'Location:.*|Setting:.*',
            'character': r'Character:.*|Cast:.*'
        }
        
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check for section markers
            is_section_start = False
            section_type = None
            for marker_type, pattern in section_markers.items():
                if re.match(pattern, line, re.IGNORECASE):
                    is_section_start = True
                    section_type = marker_type
                    break
            
            if is_section_start or line.startswith('#') or line.isupper():
                if current_segment:
                    self._enrich_segment_metadata(current_segment, concept_keywords)
                    segments.append(current_segment)
                
                current_segment = {
                    'title': line.lstrip('#').strip(),
                    'content': '',
                    'characters': [],
                    'locations': [],
                    'concepts': defaultdict(list),
                    'context_tags': set(),
                    'world_concepts': [],
                    'section_type': section_type or 'general'
                }
            elif current_segment:
                current_segment['content'] += line + '\n'
                
                # Enhanced annotation extraction
                if '*' in line:
                    parts = line.split('*')
                    for part in parts[1:]:
                        part = part.strip()
                        if 'Character:' in part:
                            char_name = part.replace('Character:', '').strip()
                            current_segment['characters'].append({
                                'name': char_name,
                                'context': line
                            })
                        elif 'Location:' in part:
                            loc_name = part.replace('Location:', '').strip()
                            current_segment['locations'].append({
                                'name': loc_name,
                                'context': line
                            })
                
                # Enhanced world concept detection
                for concept_type, keywords in concept_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in line.lower():
                            # Get surrounding context
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 3)
                            context = '\n'.join(lines[context_start:context_end])
                            
                            if concept_type == 'world':
                                current_segment['world_concepts'].append({
                                    'concept': keyword,
                                    'context': context
                                })
                            current_segment['concepts'][concept_type].append({
                                'keyword': keyword,
                                'context': context
                            })
                            current_segment['context_tags'].add(concept_type)
            
            i += 1
        
        # Process the last segment
        if current_segment:
            self._enrich_segment_metadata(current_segment, concept_keywords)
            segments.append(current_segment)
        
        return segments
        
    def _enrich_segment_metadata(self, segment: Dict[str, Any], concept_keywords: Dict[str, List[str]]):
        """Enrich segment with metadata and cross-references."""
        content_lower = segment['content'].lower()
        
        # Extract character relationships
        for char in segment['characters']:
            char_name = char['name'].lower()
            for rel_type in ['friend', 'enemy', 'ally', 'rival', 'mentor']:
                if rel_type in content_lower:
                    # Find sentences containing both character and relationship
                    sentences = segment['content'].split('.')
                    for sentence in sentences:
                        if char_name in sentence.lower() and rel_type in sentence.lower():
                            if 'relationships' not in char:
                                char['relationships'] = []
                            char['relationships'].append({
                                'type': rel_type,
                                'context': sentence.strip()
                            })
        
        # Extract location details
        for loc in segment['locations']:
            loc_name = loc['name'].lower()
            # Find descriptive sentences about the location
            sentences = segment['content'].split('.')
            for sentence in sentences:
                if loc_name in sentence.lower():
                    if 'descriptions' not in loc:
                        loc['descriptions'] = []
                    loc['descriptions'].append(sentence.strip())
        
        # Add thematic tags
        segment['themes'] = []
        theme_patterns = {
            'conflict': ['battle', 'fight', 'struggle', 'conflict'],
            'friendship': ['friend', 'together', 'bond', 'trust'],
            'growth': ['learn', 'grow', 'change', 'develop'],
            'destiny': ['fate', 'destiny', 'prophecy', 'chosen'],
            'power': ['strength', 'power', 'ability', 'force']
        }
        
        for theme, patterns in theme_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                segment['themes'].append(theme)
                segment['context_tags'].add(f'theme_{theme}')
        
        # Convert tags to list for JSON serialization
        segment['context_tags'] = list(segment['context_tags'])
        
    def _parse_world_content(self, content: str) -> Dict[str, Any]:
        """Parse world design content."""
        world_data = {
            'elements': {},
            'regions': {},
            'landmarks': [],
            'lore': [],
            'world_concepts': [],  # Added to store high-level world concepts
            'locations': []  # Added to store major locations
        }
        
        current_section = None
        current_data = {}
        
        # Keywords for world concepts
        world_keywords = ['world', 'omniterra', 'continent', 'realm', 'dimension']
        
        # First pass - extract high-level world concepts
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Look for world-level concepts - more flexible matching
            if any(keyword in line.lower() for keyword in world_keywords):
                concept = {
                    'name': line.split(':')[0] if ':' in line else line,
                    'description': '',
                    'type': 'world_concept'
                }
                
                # Gather description from following lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not any(lines[j].strip().endswith(c) for c in [':', '*']):
                    concept['description'] += lines[j].strip() + ' '
                    j += 1
                
                world_data['world_concepts'].append(concept)
                i = j  # Skip the lines we've processed
            else:
                # Handle section headers
                if line.endswith(':'):
                    if current_section and current_data:
                        if current_section == 'Element':
                            world_data['elements'][current_data['name']] = current_data
                        elif current_section == 'Region':
                            world_data['regions'][current_data['name']] = current_data
                    
                    current_section = line[:-1].strip()
                    current_data = {'name': current_section}
                    
                # Handle region content
                elif line.startswith('-'):
                    if current_section == 'Region':
                        if 'locations' not in current_data:
                            current_data['locations'] = []
                        if 'landmarks' not in current_data:
                            current_data['landmarks'] = []
                            
                        location = line[1:].strip()
                        if any(keyword in location.lower() for keyword in 
                            ['temple', 'fortress', 'cave', 'palace', 'tower', 'shrine']):
                            current_data['landmarks'].append(location)
                            world_data['landmarks'].append(location)
                        else:
                            current_data['locations'].append(location)
                            world_data['locations'].append(location)
                            
                # Handle key-value pairs
                elif ':' in line:
                    key, value = line.split(':', 1)
                    current_data[key.strip().lower()] = value.strip()
                
                # Handle lore content
                elif line.startswith('THEMES') or line.startswith('ABOUT'):
                    world_data['lore'].append(line)
                
                i += 1
                
        # Add final section
        if current_section and current_data:
            if current_section == 'Element':
                world_data['elements'][current_data['name']] = current_data
            elif current_section == 'Region':
                world_data['regions'][current_data['name']] = current_data
                
        return world_data

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
    
    def load_all_data(self) -> List[LoreEntity]:
        """Load and combine all data sources."""
        entities = []  # Initialize list at the start
        logger.debug("Starting data load process")
        
        # Load all data sources first
        self._load_monster_data()
        self._load_item_data()
        self._load_story_data()
        self._load_world_data()
        
        # Process monsters
        for source, df in self.monster_data.items():
            try:
                monster_entities = self._process_monsters(df, source)
                entities.extend(monster_entities)
                logger.info(f"Added {len(monster_entities)} monsters from {source}")
            except Exception as e:
                logger.error(f"Failed to process monster data from {source}: {str(e)}", exc_info=True)
                continue
        
        # Process items
        for source, df in self.item_data.items():
            try:
                item_entities = self._process_items(df, source)
                entities.extend(item_entities)
                logger.info(f"Added {len(item_entities)} items from {source}")
            except Exception as e:
                logger.error(f"Failed to process item data from {source}: {str(e)}", exc_info=True)
                continue
        
        # Process story content
        for source, data in self.story_data.items():
            try:
                for i, segment in enumerate(data['segments']):
                    try:
                        # Create rich description including concept contexts
                        rich_description = segment['content']
                        for concept_type, concepts in segment['concepts'].items():
                            if concepts:
                                rich_description += f"\n\n{concept_type.title()} Information:\n"
                                for concept in concepts:
                                    rich_description += f"- {concept['context']}\n"
                        
                        # Add world concepts if present
                        if segment.get('world_concepts'):
                            rich_description += "\n\nWorld Concepts:\n"
                            for concept in segment['world_concepts']:
                                rich_description += f"- {concept['concept']}\n"
                        
                        # Create metadata with semantic information
                        metadata = {
                            'title': segment['title'],
                            'concepts': segment['concepts'],
                            'context_tags': segment['context_tags'],
                            'world_concepts': segment.get('world_concepts', [])  # Add world concepts to metadata
                        }
                        
                        # Create entity with enhanced metadata
                        entity = LoreEntity(
                            id=f"story_segment_{i}",
                            name=segment['title'],
                            entity_type="Story",
                            description=rich_description,
                            metadata=metadata,
                            relationships={
                                'characters': segment['characters'],
                                'locations': segment['locations']
                            },
                            sources=[source]
                        )
                        
                        # Add context tags including world-related ones
                        entity.add_context_tags(segment['context_tags'])
                        if segment.get('world_concepts'):
                            entity.add_context_tags(['world_lore', 'omniterra_lore'])
                        
                        # Add source fragments for better retrieval
                        for concept_type, concepts in segment['concepts'].items():
                            for concept in concepts:
                                entity.add_source_fragment(
                                    text=concept['context'],
                                    source=source,
                                    context_type=concept_type
                                )
                        
                        # Add world concepts as source fragments
                        for concept in segment.get('world_concepts', []):
                            entity.add_source_fragment(
                                text=concept['concept'],
                                source=source,
                                context_type='world_concept'
                            )
                        
                        entities.append(entity)
                        
                    except Exception as e:
                        logger.error(f"Failed to process story segment {i} from {source}: {str(e)}", exc_info=True)
                        continue
            except Exception as e:
                logger.error(f"Failed to process story data from {source}: {str(e)}", exc_info=True)
                continue
        
        # Process world data with enhanced world concept handling
        for source, data in self.world_data.items():
            try:
                # Process world concepts first
                for concept in data.get('world_concepts', []):
                    try:
                        entity = LoreEntity(
                            id=f"world_concept_{concept['name'].lower().replace(' ', '_')}",
                            name=concept['name'],
                            entity_type="WorldConcept",
                            description=concept['description'],
                            metadata={'type': concept['type']},
                            sources=[source]
                        )
                        entities.append(entity)
                    except Exception as e:
                        logger.error(f"Failed to process world concept {concept['name']}: {str(e)}", exc_info=True)
                        continue
                
                # Process elements
                for element_name, element_data in data.get('elements', {}).items():
                    try:
                        description = f"Theme: {element_data.get('theme', '')}\n"
                        if 'motifs' in element_data:
                            description += f"Motifs: {', '.join(element_data['motifs'])}\n"
                        if 'affinity' in element_data:
                            description += f"Affinity: {element_data['affinity']}"
                        
                        entity = LoreEntity(
                            id=f"element_{element_name.lower()}",
                            name=element_name,
                            entity_type="Element",
                            description=description,
                            sources=[source]
                        )
                        entities.append(entity)
                    except Exception as e:
                        logger.error(f"Failed to process element {element_name}: {str(e)}", exc_info=True)
                        continue
                
                # Process regions
                for region_name, region_data in data.get('regions', {}).items():
                    try:
                        description = f"A region containing: {', '.join(region_data.get('locations', []))}"
                        entity = LoreEntity(
                            id=f"region_{region_name.lower().replace(' ', '_')}",
                            name=region_name,
                            entity_type="Region",
                            description=description,
                            relationships={
                                'landmarks': region_data.get('landmarks', []),
                                'locations': region_data.get('locations', [])
                            },
                            sources=[source]
                        )
                        entities.append(entity)
                    except Exception as e:
                        logger.error(f"Failed to process region {region_name}: {str(e)}", exc_info=True)
                        continue
            except Exception as e:
                logger.error(f"Failed to process world data from {source}: {str(e)}", exc_info=True)
                continue
        
        # Log stats
        if not entities:
            logger.warning("No entities were loaded from any data source")
        else:
            logger.info(f"Successfully loaded {len(entities)} total entities")
            type_counts = {}
            for entity in entities:
                type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
            logger.debug("Entity type breakdown:")
            for entity_type, count in type_counts.items():
                logger.debug(f"  {entity_type}: {count}")
        
        return entities

    def _process_text_content(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Process text content to extract entity mentions and relationships."""
        fragments = []
        
        # Split into manageable chunks
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            # Extract entity mentions
            entity_mentions = self._extract_entity_mentions(para)
            
            if entity_mentions:
                fragments.append({
                    'text': para,
                    'source': source,
                    'mentions': entity_mentions,
                    'context_type': self._determine_context_type(para)
                })
        
        return fragments
    
    def _extract_entity_mentions(self, text: str) -> List[Dict[str, str]]:
        """Extract mentions of known entities from text."""
        mentions = []
        
        # Check for known entities
        for entity in self.all_entities.values():
            if entity.name.lower() in text.lower():
                context = self._extract_context(text, entity.name)
                mentions.append({
                    'entity_id': entity.id,
                    'name': entity.name,
                    'context': context
                })
        
        return mentions
    
    def _determine_context_type(self, text: str) -> str:
        """Determine the type of context for a text fragment."""
        context_indicators = {
            'evolution': ['evolve', 'evolution', 'transform'],
            'ability': ['ability', 'power', 'skill'],
            'location': ['found in', 'located', 'habitat'],
            'relationship': ['friend', 'enemy', 'ally'],
            'lore': ['legend', 'story', 'myth']
        }
        
        for context_type, indicators in context_indicators.items():
            if any(indicator in text.lower() for indicator in indicators):
                return context_type
        
        return 'general'
    
    def _extract_context(self, text: str, entity_name: str) -> str:
        """Extract relevant context around an entity mention."""
        # Find the sentence containing the entity
        sentences = text.split('.')
        for sentence in sentences:
            if entity_name.lower() in sentence.lower():
                return sentence.strip()
        return ""
    
    def _link_entities(self):
        """Create relationships between entities based on mentions."""
        for entity in self.all_entities.values():
            for fragment in entity.source_fragments:
                for mention in fragment.get('mentions', []):
                    if mention['entity_id'] in self.all_entities:
                        target_entity = self.all_entities[mention['entity_id']]
                        
                        # Add cross-references
                        entity.add_reference(
                            target_id=target_entity.id,
                            relationship=fragment['context_type'],
                            context=mention['context'],
                            source=fragment['source']
                        )
                        
                        # Add reverse reference
                        target_entity.mentioned_in.append(entity.id)
                        
                        # Add context tags
                        both_entities = [entity, target_entity]
                        for e in both_entities:
                            e.add_context_tags([
                                fragment['context_type'],
                                f"related_{e.entity_type.lower()}"
                            ]) 