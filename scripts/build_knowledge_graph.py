"""Script to build and initialize the Hatchyverse knowledge graph."""

import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.knowledge_graph import HatchyKnowledgeGraph
from src.models.enhanced_loader import EnhancedDataLoader

def build_knowledge_graph(data_dir: Path, output_dir: Path) -> None:
    """Build and save the knowledge graph."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing knowledge graph...")
        knowledge_graph = HatchyKnowledgeGraph()
        data_loader = EnhancedDataLoader(knowledge_graph)
        
        # Set data directory
        data_loader.set_data_directory(data_dir)
        
        # Load all data
        logger.info("Loading data into knowledge graph...")
        loaded_entities = data_loader.load_all_data()
        
        # Get statistics
        stats = knowledge_graph.get_statistics()
        logger.info(f"Knowledge graph statistics: {json.dumps(stats, indent=2)}")
        
        # Export graph
        logger.info("Exporting knowledge graph...")
        graph_data = knowledge_graph.export_to_dict()
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save graph data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"knowledge_graph_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        # Create symlink to latest
        latest_link = output_dir / "knowledge_graph_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_file)
        
        logger.info(f"Knowledge graph saved to {output_file}")
        logger.info(f"Latest symlink updated: {latest_link}")
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}")
        raise

if __name__ == '__main__':
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "knowledge_graphs"
    
    # Build graph
    build_knowledge_graph(data_dir, output_dir) 