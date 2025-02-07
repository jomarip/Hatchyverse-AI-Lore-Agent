from typing import Dict, List, Any, Optional, Set
import re
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyze queries to extract intent and entities."""
    
    def __init__(self):
        # Define attribute patterns that can be searched
        self.attribute_patterns = {
            'generation': {
                'patterns': [
                    r'gen(?:eration)?\s*(\d+)',
                    r'(?:gen|generation)\s*(one|two|three|1|2|3)',
                    r'(?:first|second|third)\s+generation',
                    r'gen-?(\d+)'  # Added for gen-1 format
                ],
                'value_map': {
                    'one': '1', 'two': '2', 'three': '3',
                    'first': '1', 'second': '2', 'third': '3'
                }
            },
            'element': {
                'patterns': [
                    r'(fire|water|plant|dark|light|void)\s+(?:type|element)?',
                    r'(?:type|element)\s+(fire|water|plant|dark|light|void)',
                    r'(?:fire|water|plant|dark|light|void)-type'  # Added for element-type format
                ],
                'value_map': {},  # Direct mapping
                'emoji_map': {
                    'fire': '🔥', 'water': '💧', 'plant': '🌿',
                    'dark': '🌑', 'light': '✨', 'void': '🌀'
                }
            },
            'size': {
                'patterns': [
                    r'(large|huge|massive|giant|big|enormous|colossal)',
                    r'(?:large|big)\s+enough',
                    r'(?:size|sized)\s+(large|huge|massive)'
                ],
                'attribute': 'size',
                'value': 'large',
                'related_terms': ['large', 'huge', 'massive', 'giant', 'enormous', 'colossal']
            },
            'mountable': {
                'patterns': [
                    r'(?:can\s+be\s+)?(rid(?:e|ing|eable)|mount(?:ed|able)?)',
                    r'(?:can|possible)\s+(?:to\s+)?ride',
                    r'(?:for|suitable\s+for)\s+riding'
                ],
                'attribute': 'mountable',
                'value': True,
                'related_terms': ['can be ridden', 'mountable', 'rideable', 'can ride', 'for riding']
            },
            'habitat': {
                'patterns': [
                    r'(?:live|found|habitat|home)\s+(?:in|at)\s+(\w+)',
                    r'native\s+to\s+(\w+)',
                    r'from\s+(?:the\s+)?(\w+)'
                ],
                'value_map': {}
            },
            'location': {
                'patterns': [
                    r'(?:in|at|from|near)\s+(the\s+)?([\w\s]+(?:city|region|kingdom|temple|village))',
                    r'([\w\s]+(?:city|region|kingdom|temple|village))\s+(?:area|location|place)',
                    r'(?:location|place)\s+(?:called|named)\s+([\w\s]+)'
                ],
                'value_map': {}
            },
            'faction': {
                'patterns': [
                    r'(?:faction|group|army|force)\s+(?:called|named)\s+([\w\s]+)',
                    r'(the\s+)?([\w\s]+)\s+(?:faction|group|army|force)',
                    r'members?\s+of\s+(the\s+)?([\w\s]+)'
                ],
                'value_map': {}
            }
        }
        
        # Define query type patterns with priorities
        self.query_type_patterns = {
            'count': {
                'patterns': [
                    r'how\s+many',
                    r'count\s+(?:of|the)',
                    r'number\s+of',
                    r'total\s+(?:number|count)',
                    r'list\s+(?:all|the)'  # Added for list queries
                ],
                'priority': 1
            },
            'description': {
                'patterns': [
                    r'(?:what|tell\s+me)\s+about',
                    r'describe',
                    r'who\s+is',
                    r'what\s+is',
                    r'information\s+(?:about|on)'  # Added for information queries
                ],
                'priority': 2
            },
            'location': {
                'patterns': [
                    r'where\s+is',
                    r'location\s+of',
                    r'find\s+(?:the\s+)?(?:place|location)',
                    r'(?:in|at)\s+what\s+(?:place|location)'
                ],
                'priority': 2
            },
            'relationship': {
                'patterns': [
                    r'(?:how|what)\s+is\s+(?:the\s+)?relationship',
                    r'connected\s+to',
                    r'related\s+to',
                    r'(?:allies|enemies|friends)\s+(?:of|with)'
                ],
                'priority': 3
            }
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent and attributes with improved accuracy."""
        query_lower = query.lower()
        
        # Initialize results
        results = {
            'query_type': 'standard',
            'filters': {},
            'confidences': {},
            'metadata': {},
            'extracted_entities': []
        }
        
        # Determine query type with priority handling
        highest_priority = 0
        for qtype, type_info in self.query_type_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in type_info['patterns']):
                if type_info['priority'] > highest_priority:
                    results['query_type'] = qtype
                    highest_priority = type_info['priority']
        
        # Extract attributes with confidence scores
        for attr_name, attr_config in self.attribute_patterns.items():
            for pattern in attr_config['patterns']:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    # Extract value and calculate confidence
                    value = match.group(1) if match.groups() else None
                    if value:
                        # Apply value mapping if exists
                        value = attr_config.get('value_map', {}).get(value, value)
                        
                        # Store in filters and calculate confidence
                        results['filters'][attr_name] = value
                        confidence = self._calculate_confidence(match, query_lower)
                        results['confidences'][attr_name] = confidence
                        
                        # Add to extracted entities if it's a named entity
                        if attr_name in ['location', 'faction']:
                            results['extracted_entities'].append({
                                'type': attr_name,
                                'value': value,
                                'confidence': confidence
                            })
        
        # Add metadata for response formatting
        results['metadata'] = {
            'emojis': {
                k: v for k, v in self.attribute_patterns.get('element', {}).get('emoji_map', {}).items()
                if k in str(results['filters'].get('element', '')).lower()
            }
        }
        
        return results

    def _calculate_confidence(self, match: re.Match, query: str) -> float:
        """Calculate confidence score for an extracted attribute."""
        # Base confidence for regex match
        confidence = 0.8
        
        # Adjust based on match position (earlier matches are slightly more confident)
        position_factor = 1.0 - (match.start() / len(query) * 0.2)
        confidence *= position_factor
        
        # Adjust based on surrounding context
        context_start = max(0, match.start() - 20)
        context_end = min(len(query), match.end() + 20)
        context = query[context_start:context_end]
        
        # Boost confidence for clear contextual markers
        clarity_markers = ['is', 'the', 'in', 'of', 'from']
        for marker in clarity_markers:
            if f" {marker} " in context:
                confidence += 0.05
        
        # Cap confidence at 1.0
        return min(1.0, confidence)

class ContextualRetriever:
    """Retrieves relevant context using vector search and knowledge graph."""
    
    def __init__(
        self,
        knowledge_graph: HatchyKnowledgeGraph,
        vector_store: VectorStore,
        max_results: int = 5,
        max_relationship_depth: int = 2
    ):
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.max_results = max_results
        self.max_relationship_depth = max_relationship_depth
        self.query_analyzer = QueryAnalyzer()

    def get_context(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant context for a query with improved comprehensive search."""
        query_analysis = self.query_analyzer.analyze(query)
        logger.debug(f"Query analysis: {query_analysis}")
        
        # Initialize combined context with source tracking
        all_context = []
        source_tracker = {
            'structured': set(),  # Track structured data sources used
            'unstructured': set(),  # Track unstructured data sources used
            'entities': set()  # Track entities found
        }
        
        # 1. First try direct entity lookup for exact matches
        for entity in query_analysis['extracted_entities']:
            entity_name = entity['value']
            entity_type = entity['type']
            
            # Try exact entity match first
            found_entity = self.knowledge_graph.find_entity_by_name(
                entity_name,
                fuzzy_match=True,
                entity_type=entity_type
            )
            
            if found_entity:
                source_tracker['entities'].add(found_entity['id'])
                source_tracker['structured'].add(found_entity.get('source', 'unknown'))
                
                # Get entity context with relationships
                entity_context = self._get_entity_with_relationships(found_entity)
                all_context.extend(entity_context)
        
        # 2. Search CSV data based on query type and filters
        if query_analysis['query_type'] in ['count', 'description']:
            csv_results = self._search_structured_data(
                query,
                query_analysis['filters'],
                query_analysis['query_type']
            )
            for result in csv_results:
                source = result.get('metadata', {}).get('source', 'unknown')
                source_tracker['structured'].add(source)
                all_context.append(result)
        
        # 3. Search unstructured text data
        text_results = self._search_unstructured_data(
            query,
            query_analysis['filters'],
            query_analysis['query_type']
        )
        for result in text_results:
            source = result.get('metadata', {}).get('source', 'unknown')
            source_tracker['unstructured'].add(source)
            all_context.append(result)
        
        # 4. Add specialized context based on query type
        specialized_context = self._get_specialized_context(
            query_analysis['query_type'],
            query_analysis['filters'],
            query
        )
        all_context.extend(specialized_context)
        
        # 5. Validate and enhance context
        enhanced_context = self._validate_and_enhance_context(
            all_context,
            query_analysis,
            source_tracker
        )
        
        # Log context sources for debugging
        logger.debug(f"Structured sources used: {source_tracker['structured']}")
        logger.debug(f"Unstructured sources used: {source_tracker['unstructured']}")
        logger.debug(f"Entities found: {source_tracker['entities']}")
        
        return enhanced_context

    def _search_structured_data(
        self,
        query: str,
        filters: Dict[str, Any],
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Search through structured data sources with improved filtering."""
        results = []
        
        # Define source priority based on query type
        source_priority = {
            'monster': [
                "Hatchy - Monster Data - gen 1.csv",
                "Hatchy - Monster Data - gen 2.csv",
                "Hatchipedia - monsters.csv"
            ],
            'location': [
                "Hatchipedia - nations and politics.csv",
                "Hatchipedia - Factions and groups.csv"
            ],
            'faction': [
                "Hatchipedia - Factions and groups.csv",
                "Hatchipedia - nations and politics.csv"
            ]
        }
        
        # Determine which sources to search based on filters
        search_sources = []
        if 'generation' in filters:
            search_sources.extend(source_priority['monster'])
        if 'location' in filters:
            search_sources.extend(source_priority['location'])
        if 'faction' in filters:
            search_sources.extend(source_priority['faction'])
        
        # If no specific sources determined, search all
        if not search_sources:
            search_sources = list(set(sum(source_priority.values(), [])))
        
        # Search each source with proper filtering
        for source in search_sources:
            try:
                entities = self.knowledge_graph.search_entities(
                    query,
                    file_filter=source,
                    filters=filters
                )
                
                for entity in entities:
                    # Format entity data
                    content = self._format_entity_content(entity)
                    
                    # Add to results with source tracking
                    results.append({
                        'content': content,
                        'metadata': {
                            'source': source,
                            'type': entity.get('entity_type'),
                            'element': entity.get('element'),
                            'generation': entity.get('generation'),
                            'file_name': source,
                            'data_type': 'structured'
                        }
                    })
            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
                continue
        
        return results

    def _search_unstructured_data(
        self,
        query: str,
        filters: Dict[str, Any],
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Search through unstructured text data with improved context awareness."""
        results = []
        
        # Define text file sources and their priorities
        text_sources = {
            'lore': [
                "Hatchy World Comic_ Chaos saga.txt",
                "Hatchy World _ world design.txt",
                "HWCS - Simplified main arc and arc suggestions.txt"
            ],
            'location': [
                "Hatchy World _ world design.txt",
                "Hatchyverse Eco Presentation v3.txt"
            ],
            'faction': [
                "Hatchy World Comic_ Chaos saga.txt",
                "HWCS - Simplified main arc and arc suggestions.txt"
            ]
        }
        
        # Determine search strategy based on query type
        search_config = {
            'description': {'k': 3, 'sources': text_sources['lore']},
            'location': {'k': 2, 'sources': text_sources['location']},
            'relationship': {'k': 3, 'sources': text_sources['faction']},
            'standard': {'k': 2, 'sources': sum(text_sources.values(), [])}
        }
        
        config = search_config.get(query_type, search_config['standard'])
        
        # Search each source
        for source in config['sources']:
            try:
                # Construct search query with filters
                search_query = query
                if filters:
                    filter_terms = []
                    for key, value in filters.items():
                        if key in ['element', 'generation', 'location', 'faction']:
                            filter_terms.append(f"{key}:{value}")
                    if filter_terms:
                        search_query = f"{query} {' '.join(filter_terms)}"
                
                # Perform semantic search
                results_for_source = self.vector_store.similarity_search(
                    search_query,
                    k=config['k'],
                    filter={"source": source}
                )
                
                # Process and add results
                for doc in results_for_source:
                    processed_doc = self._process_text_content(doc)
                    if processed_doc:
                        results.append(processed_doc)
            
            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
        
        return results

    def _validate_and_enhance_context(
        self,
        context: List[Dict[str, Any]],
        query_analysis: Dict[str, Any],
        source_tracker: Dict[str, Set[str]]
    ) -> List[Dict[str, Any]]:
        """Validate and enhance retrieved context."""
        enhanced_context = []
        
        # Track which aspects of the query have been addressed
        coverage = {
            'entity_mentions': set(),
            'attributes_covered': set(),
            'relationships_found': set()
        }
        
        for ctx in context:
            # Validate content relevance
            if not self._is_content_relevant(ctx, query_analysis):
                continue
                        
            # Track coverage
            self._update_coverage_tracking(ctx, coverage, query_analysis)
            
            # Add confidence score
            ctx['metadata']['confidence'] = self._calculate_context_confidence(
                ctx, query_analysis, coverage
            )
            
            enhanced_context.append(ctx)
            
        # Sort by confidence
        enhanced_context.sort(
            key=lambda x: x.get('metadata', {}).get('confidence', 0),
            reverse=True
        )
        
        # Add coverage metadata
        if enhanced_context:
            enhanced_context[0]['metadata']['coverage_stats'] = {
                'entity_coverage': len(coverage['entity_mentions']) / max(1, len(query_analysis['extracted_entities'])),
                'attribute_coverage': len(coverage['attributes_covered']) / max(1, len(query_analysis['filters'])),
                'source_diversity': len(source_tracker['structured']) + len(source_tracker['unstructured'])
            }
        
        return enhanced_context

    def _is_content_relevant(
        self,
        context: Dict[str, Any],
        query_analysis: Dict[str, Any]
    ) -> bool:
        """Check if content is relevant to the query."""
        # Check if content matches query filters
        metadata = context.get('metadata', {})
        filters = query_analysis['filters']
        
        for key, value in filters.items():
            if key in metadata and str(metadata[key]).lower() != str(value).lower():
                return False
        
        # Check if content type matches query type
        query_type = query_analysis['query_type']
        content_type = metadata.get('type', '')
        
        if query_type == 'count' and 'count' not in metadata:
            return False
        
        if query_type == 'location' and content_type not in ['location', 'text']:
            return False
        
        return True

    def _update_coverage_tracking(
        self,
        context: Dict[str, Any],
        coverage: Dict[str, Set[str]],
        query_analysis: Dict[str, Any]
    ):
        """Update tracking of query coverage."""
        # Track entity mentions
        for entity in query_analysis['extracted_entities']:
            if entity['value'].lower() in context.get('content', '').lower():
                coverage['entity_mentions'].add(entity['value'])
        
        # Track attribute coverage
        metadata = context.get('metadata', {})
        for attr in query_analysis['filters']:
            if attr in metadata:
                coverage['attributes_covered'].add(attr)
        
        # Track relationships
        if 'relationships' in context.get('entity_context', {}):
            for rel in context['entity_context']['relationships']:
                coverage['relationships_found'].add(
                    f"{rel.get('source', '')}-{rel.get('type', '')}-{rel.get('target', '')}"
                )

    def _calculate_context_confidence(
        self,
        context: Dict[str, Any],
        query_analysis: Dict[str, Any],
        coverage: Dict[str, Set[str]]
    ) -> float:
        """Calculate confidence score for context relevance."""
        base_confidence = 0.5
        
        # Boost for direct entity matches
        for entity in query_analysis['extracted_entities']:
            if entity['value'].lower() in context.get('content', '').lower():
                base_confidence += 0.2
        
        # Boost for attribute matches
        metadata = context.get('metadata', {})
        for attr, value in query_analysis['filters'].items():
            if attr in metadata and str(metadata[attr]).lower() == str(value).lower():
                base_confidence += 0.1
        
        # Boost for relationship coverage
        if 'relationships' in context.get('entity_context', {}):
            base_confidence += 0.1
        
        # Adjust based on source type
        if metadata.get('data_type') == 'structured':
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _get_entity_with_relationships(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get entity context with relationships."""
        context = []
        
        # Get entity and its relationships
        entity_context = {
            'content': self._format_entity_content(entity),
            'metadata': {
                'source': entity.get('source', 'unknown'),
                'type': entity.get('entity_type'),
                'element': entity.get('element'),
                'generation': entity.get('generation'),
                'file_name': entity.get('source', '').split('/')[-1] if entity.get('source') else 'unknown'
            }
        }
        context.append(entity_context)
        
        # Get related entities
        related_context = self._get_related_entities([entity['id']])
        context.extend(related_context)
        
        return context

    def _get_related_entities(self, entity_ids: List[str]) -> List[Dict]:
        """Get 2nd-degree relationships from knowledge graph."""
        related = []
        seen = set()
        
        for eid in entity_ids:
            if eid in seen:
                continue
            
            seen.add(eid)
            
            # Get direct relationships
            direct_rels = self.knowledge_graph.get_entity_relationships(eid)
            for rel in direct_rels:
                target_id = rel['target']
                if target_id not in seen:
                    target_entity = self.knowledge_graph.get_entity(target_id)
                    if target_entity:
                        related.append({
                            'content': self._format_entity_content(target_entity),
                            'metadata': {
                                'source': target_entity.get('source', 'unknown'),
                                'type': target_entity.get('entity_type'),
                                'element': target_entity.get('element'),
                                'generation': target_entity.get('generation'),
                                'relationship': rel['type'],
                                'file_name': target_entity.get('source', '').split('/')[-1] if target_entity.get('source') else 'unknown'
                            }
                        })
                        
                        # Get second-degree relationships if within depth limit
                        if len(seen) < self.max_relationship_depth:
                            second_rels = self.knowledge_graph.get_entity_relationships(target_id)
                            for second_rel in second_rels:
                                second_target = second_rel['target']
                                if second_target not in seen:
                                    second_entity = self.knowledge_graph.get_entity(second_target)
                                    if second_entity:
                                        related.append({
                                            'content': self._format_entity_content(second_entity),
                                            'metadata': {
                                                'source': second_entity.get('source', 'unknown'),
                                                'type': second_entity.get('entity_type'),
                                                'element': second_entity.get('element'),
                                                'generation': second_entity.get('generation'),
                                                'relationship': f"{rel['type']}->{second_rel['type']}",
                                                'file_name': second_entity.get('source', '').split('/')[-1] if second_entity.get('source') else 'unknown'
                                            }
                                        })
        
        return related

    def _format_entity_content(self, entity: Dict[str, Any]) -> str:
        """Format entity data into readable content."""
        content_parts = []
        
        # Add name and description
        content_parts.append(f"Name: {entity['name']}")
        if entity.get('description'):
            content_parts.append(f"Description: {entity['description']}")
        
        # Add all relevant attributes
        for key, value in entity.get('attributes', {}).items():
            if key not in ['name', 'description'] and value:
                content_parts.append(f"{key}: {value}")
        
        # Add relationship information if available
        if entity.get('relationships'):
            rel_parts = []
            for rel in entity['relationships']:
                rel_parts.append(f"- {rel['type']} {rel.get('target_name', '')}")
            if rel_parts:
                content_parts.append("Relationships:")
                content_parts.extend(rel_parts)
        
        return '\n'.join(content_parts)

    def _get_specialized_context(
        self,
        query_type: str,
        filters: Dict[str, Any],
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get specialized context based on query type."""
        if query_type == 'element':
            element = filters.get('element')
            if element:
                return self._get_element_context(element)
        elif query_type == 'location' and query:
            location_terms = self._extract_location_terms(query)
            return [self._get_location_context(term) for term in location_terms]
        elif query_type == 'generation':
            generation = filters.get('generation')
            if generation:
                return self._get_generation_context(generation)
        return []

    def _get_element_context(self, element: str) -> List[Dict[str, Any]]:
        """Get comprehensive context for an element."""
        context = []
        
        # Get monsters of that element
        monsters = self.knowledge_graph.get_entities(entity_type="monster", element=element)
        context.extend([{
            'content': f"{monster['name']}: {monster.get('description', '')}",
            'metadata': {
                'source': monster.get('source', 'unknown'),
                'type': 'monster',
                'element': monster.get('element'),
                'generation': monster.get('generation')
            }
        } for monster in monsters])
        
        # Get element lore from text files
        lore_results = self.vector_store.similarity_search(
            f"{element} element lore",
            k=3
        )
        context.extend(self._process_docs(lore_results))
        
        return context

    def _get_location_context(self, location: str) -> List[Dict[str, Any]]:
        """Get comprehensive context for a location."""
        context = []
        
        # 1. Search in all text files
        text_files = [
            "Hatchy World Comic_ Chaos saga.txt",
            "Hatchy World _ world design.txt",
            "HWCS - Simplified main arc and arc suggestions.txt",
            "Hatchyverse Eco Presentation v3.txt"
        ]
        
        for file in text_files:
            try:
                # Search for exact location name
                results = self.vector_store.similarity_search(
                    location,
                    k=3,
                    filter={"source": file}
                )
                context.extend(self._process_docs(results))
                
                # Also search for location with "region", "city", etc.
                for location_type in ["region", "city", "kingdom", "area", "location"]:
                    type_results = self.vector_store.similarity_search(
                        f"{location} {location_type}",
                        k=2,
                        filter={"source": file}
                    )
                    context.extend(self._process_docs(type_results))
            except Exception as e:
                logger.error(f"Error searching {file}: {str(e)}")
        
        # 2. Search in CSV data for location references
        csv_files = [
            "Hatchipedia - nations and politics.csv",
            "Hatchipedia - Factions and groups.csv"
        ]
        
        for file in csv_files:
            try:
                entities = self.knowledge_graph.search_entities(location, file_filter=file)
                for entity in entities:
                    context.append({
                        'content': f"Name: {entity['name']}\nDescription: {entity.get('description', '')}\n" +
                                  '\n'.join(f"{k}: {v}" for k, v in entity.get('attributes', {}).items() 
                                  if k not in ['name', 'description']),
                        'metadata': {
                            'source': file,
                            'type': entity.get('entity_type'),
                            'file_name': file
                        }
                    })
            except Exception as e:
                logger.error(f"Error searching {file}: {str(e)}")
        
        return context

    def _get_generation_context(self, generation: str) -> List[Dict[str, Any]]:
        """Get comprehensive context for a generation."""
        context = []
        
        # Get all monsters from that generation
        monsters = self.knowledge_graph.get_entities_by_generation(generation)
        for monster in monsters:
            context.append({
                'content': f"Name: {monster['name']}\nDescription: {monster.get('description', '')}\n" +
                          '\n'.join(f"{k}: {v}" for k, v in monster.get('attributes', {}).items() 
                          if k not in ['name', 'description']),
                'metadata': {
                    'source': monster.get('source', 'unknown'),
                    'type': 'monster',
                    'element': monster.get('element'),
                    'generation': generation,
                    'file_name': monster.get('source', '').split('/')[-1] if monster.get('source') else 'unknown'
                }
            })
        
        # Get generation-specific lore
        gen_results = self.vector_store.similarity_search(
            f"generation {generation}",
            k=3
        )
        context.extend(self._process_docs(gen_results))
        
        return context

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words and split into terms
        common_words = {'what', 'where', 'who', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
        terms = query.lower().split()
        return [term for term in terms if term not in common_words]

    def _search_csv_data(self, keyword: str) -> List[Dict[str, Any]]:
        """Search through CSV data for matches."""
        results = []
        
        # Search through all loaded CSV files
        csv_files = [
            "Hatchy - Monster Data - gen 1.csv",
            "Hatchy - Monster Data - gen 2.csv",
            "Hatchipedia - monsters.csv",
            "Hatchipedia - nations and politics.csv",
            "Hatchipedia - Factions and groups.csv"
        ]
        
        for file in csv_files:
            try:
                entities = self.knowledge_graph.search_entities(keyword, file_filter=file)
                for entity in entities:
                    # Create detailed content with all available information
                    content_parts = []
                    content_parts.append(f"Name: {entity['name']}")
                    if entity.get('description'):
                        content_parts.append(f"Description: {entity['description']}")
                    
                    # Add all relevant attributes
                    for key, value in entity.items():
                        if key not in ['name', 'description', 'id', '_metadata'] and value:
                            content_parts.append(f"{key}: {value}")
                    
                    results.append({
                        'content': '\n'.join(content_parts),
                        'metadata': {
                            'source': file,
                            'type': entity.get('entity_type'),
                            'element': entity.get('element'),
                            'generation': entity.get('generation'),
                            'file_name': file
                        }
                    })
            except Exception as e:
                logger.error(f"Error searching {file}: {str(e)}")
                continue
        
        return results

    def _search_text_files(self, keyword: str) -> List[Dict[str, Any]]:
        """Search through text files for matches."""
        text_files = [
            "Hatchy World Comic_ Chaos saga.txt",
            "Hatchy World _ world design.txt",
            "HWCS - Simplified main arc and arc suggestions.txt"
        ]
        
        results = []
        for file in text_files:
            try:
                # Use vector store to search text content
                file_results = self.vector_store.similarity_search(
                    keyword,
                    k=3,
                    filter={"source": file}
                )
                results.extend(self._process_docs(file_results))
            except Exception as e:
                logger.error(f"Error searching {file}: {str(e)}")
        
        return results

    def _process_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Process retrieved documents and format for LLM consumption."""
        processed_results = []
        
        for doc in docs:
            # Extract entity information if available
            entity_id = doc.metadata.get('entity_id')
            entity_context = None
            if entity_id:
                entity_context = self.get_entity_context(entity_id)
            
            # Create formatted result with source information
            result = {
                'text_content': doc.page_content,
                'metadata': {
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': doc.metadata.get('type', 'text'),
                    'generation': doc.metadata.get('generation'),
                    'element': doc.metadata.get('element'),
                    'file_name': doc.metadata.get('source', '').split('/')[-1] if doc.metadata.get('source') else 'unknown'
                }
            }
            
            # Add entity context if available
            if entity_context:
                result['entity_context'] = entity_context
            
            processed_results.append(result)
        
        return processed_results

    def _extract_location_terms(self, query: str) -> List[str]:
        """Extract location terms from query."""
        # Known location terms from the world data
        known_locations = {
            'ixor': 'city',
            'leaf city': 'city',
            'water kingdom': 'kingdom',
            'fire region': 'region',
            'dark region': 'region',
            'light region': 'region',
            'void temple': 'temple',
            'fire temple': 'temple'
        }
        
        query_lower = query.lower()
        found_terms = []
        
        # Check for known locations
        for loc in known_locations:
            if loc in query_lower:
                found_terms.append(loc)
        
        # Check for location type words
        location_types = ['city', 'town', 'region', 'kingdom', 'temple', 'village']
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in location_types and i > 0:
                found_terms.append(f"{words[i-1]} {word}")
        
        return found_terms

    def get_entity_context(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a specific entity."""
        try:
            entity = self.knowledge_graph.get_entity(entity_id)
            if not entity:
                return None
            
            context = {
                'entity': entity,
                'relationships': self.knowledge_graph.get_entity_relationships(entity_id),
                'similar_entities': self._get_similar_entities(entity)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting entity context: {str(e)}")
            return None
    
    def _get_similar_entities(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get similar entities based on vector similarity."""
        try:
            # Create a query from entity attributes
            query_parts = [
                entity['name'],
                entity.get('description', ''),
                ' '.join(f"{k}: {v}" for k, v in entity.get('attributes', {}).items())
            ]
            query = ' '.join(query_parts)
            
            # Get similar documents from vector store
            similar_docs = self.vector_store.similarity_search(
                query,
                k=self.max_results
            )
            
            # Convert documents to entities
            similar_entities = []
            for doc in similar_docs:
                entity_id = doc.metadata.get('entity_id')
                if entity_id and entity_id != entity['id']:
                    similar_entity = self.knowledge_graph.get_entity(entity_id)
                    if similar_entity:
                        similar_entities.append(similar_entity)
            
            return similar_entities
            
        except Exception as e:
            logger.error(f"Error getting similar entities: {str(e)}")
            return []

    def get_world_context(self, query: str) -> str:
        """Specialized retrieval for world concepts"""
        return self.vector_store.similarity_search(
            query=query,
            k=5,
            filter={"source": "world_design"},
            score_threshold=0.7
        )

    def _process_text_content(self, doc: Document) -> Dict[str, Any]:
        """Process text content and format for LLM consumption."""
        try:
            # Extract entity information if available
            entity_id = doc.metadata.get('entity_id')
            entity_context = None
            if entity_id:
                entity_context = self.get_entity_context(entity_id)
            
            # Create formatted result with source information
            result = {
                'text_content': doc.page_content,
                'metadata': {
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': doc.metadata.get('type', 'text'),
                    'generation': doc.metadata.get('generation'),
                    'element': doc.metadata.get('element'),
                    'file_name': doc.metadata.get('source', '').split('/')[-1] if doc.metadata.get('source') else 'unknown'
                }
            }
            
            # Add entity context if available
            if entity_context:
                result['entity_context'] = entity_context
            
            return result
        except Exception as e:
            logger.error(f"Error processing text content: {str(e)}")
            return {
                'text_content': '',
                'metadata': {
                    'source': 'unknown',
                    'type': 'text',
                    'error': str(e)
                }
            } 