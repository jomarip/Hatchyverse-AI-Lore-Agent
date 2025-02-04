from typing import List, Dict, Any, Optional
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from .lore_validator import LoreValidator
import logging

logger = logging.getLogger(__name__)

class FilteredRetriever(BaseRetriever):
    """Custom retriever that filters and prioritizes items based on query type."""
    
    base_retriever: BaseRetriever = Field(description="Base retriever to filter and enhance")
    vector_store: Any = Field(description="Vector store for additional filtering")
    
    class Config:
        arbitrary_types_allowed = True
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async implementation - required by BaseRetriever."""
        raise NotImplementedError("Async retrieval not implemented")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents, prioritizing items for item-related queries."""
        try:
            # Check if this is an item-related query
            item_keywords = ["item", "items", "equipment", "available", "use", "using"]
            ability_keywords = ["attack", "power", "ability", "abilities", "skill", "skills"]
            element_keywords = {
                "fire": ["fire", "flame", "burning", "heat", "volcanic"],
                "water": ["water", "aqua", "ice", "frost", "crystal"],
                "dark": ["dark", "shadow", "night", "mysterious"]
            }
            
            # Add specific item names to search
            item_names = {
                "fire": ["Flame Essence", "flame gems", "fire essence crystals"],
                "water": ["Frost Crystal", "ice shard"],
                "dark": ["Shadow Essence", "dark crystal"]
            }
            
            query_lower = query.lower()
            is_item_query = any(kw in query_lower for kw in item_keywords)
            is_ability_query = any(kw in query_lower for kw in ability_keywords)
            
            # Detect element type from query
            element_type = None
            for element, keywords in element_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    element_type = element
                    break
            
            all_docs = []
            seen_contents = set()
            search_kwargs = {"k": 5}
            
            try:
                # Get item-specific results if it's an item query
                if is_item_query:
                    # Search for items using type metadata
                    item_filter = {"metadata": {"type": "Item"}}
                    
                    # Create combined query with item names if we know the element
                    if element_type and element_type in item_names:
                        item_query = f"{query} {' '.join(item_names[element_type])}"
                    else:
                        item_query = query
                    
                    item_docs = self.vector_store.similarity_search(
                        item_query,
                        filter=item_filter,
                        **search_kwargs
                    )
                    
                    # Add unique item documents
                    for doc in item_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                
                # Get ability-related results if it's an ability query or item query
                if is_ability_query or is_item_query:
                    ability_query = f"{query} abilities attacks powers"
                    if element_type:
                        ability_query = f"{ability_query} {element_type} element"
                    
                    ability_docs = self.vector_store.similarity_search(
                        ability_query,
                        **search_kwargs
                    )
                    
                    # Add unique ability documents
                    for doc in ability_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                
                # Get element-specific results without filtering
                if element_type:
                    element_query = f"{query} {element_type} {' '.join(element_keywords[element_type])}"
                    element_docs = self.vector_store.similarity_search(
                        element_query,
                        **search_kwargs
                    )
                    
                    # Add unique element documents
                    for doc in element_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                
                # If we have results, return them
                if all_docs:
                    return all_docs[:4]  # Limit to top 4 results
                
            except Exception as e:
                logger.warning(f"Error in specialized search, falling back to general search: {e}")
            
            # Fallback to base retriever with enhanced query
            enhanced_query = query
            if element_type:
                if element_type in item_names:
                    enhanced_query = f"{query} {' '.join(item_names[element_type])}"
                enhanced_query = f"{enhanced_query} {' '.join(element_keywords[element_type])}"
            return self.base_retriever.get_relevant_documents(enhanced_query)
            
        except Exception as e:
            logger.error(f"Error in FilteredRetriever: {e}")
            return self.base_retriever.get_relevant_documents(query)

class LoreChatbot:
    """Manages conversations and lore interactions with users."""
    
    def __init__(
        self,
        validator: LoreValidator,
        model_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        self.validator = validator
        self.model_provider = model_provider
        self.primary_model_name = model_name
        self.fallback_model_name = os.getenv("OPENAI_FALLBACK_MODEL")
        self.temperature = temperature
        
        # Initialize the appropriate LLM based on provider
        if model_provider == "openai":
            self.primary_llm = self._create_llm(self.primary_model_name)
            if self.fallback_model_name:
                self.fallback_llm = self._create_llm(self.fallback_model_name)
            else:
                self.fallback_llm = None
        elif model_provider == "anthropic":
            self.primary_llm = ChatAnthropic(
                model_name=model_name or "claude-3-sonnet-20240229",
                temperature=temperature
            )
            self.fallback_llm = None
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Define the base prompt template
        self.base_prompt = ChatPromptTemplate.from_template(
            """You are a knowledgeable Hatchyverse Lorekeeper. Your role is to help users explore and 
            understand the Hatchyverse world while ensuring all new contributions maintain consistency 
            with existing lore.

            Use this context from the knowledge base:
            {context}

            Current conversation history:
            {chat_history}

            User's message: {question}

            Remember:
            - Be friendly and enthusiastic about Hatchyverse lore
            - ALWAYS mention specific items, locations, and creatures from the knowledge base by their EXACT names
            - When discussing items or equipment:
              * Start by listing ALL relevant items with their EXACT names
              * Describe each item's abilities, powers, and attack capabilities
              * Explain how these items enhance combat and abilities
            - When discussing elements or types:
              * Describe their unique abilities and attack styles
              * Mention specific locations where they train or gather
              * Include details about their powers and combat capabilities
            - NEVER make up names - only use items/locations/creatures explicitly mentioned in the knowledge base
            - Include specific details about abilities, attacks, and powers
            - Be comprehensive in covering all relevant aspects (items, abilities, locations)
            - Use exact terminology from the knowledge base

            Format your response with clear sections:
            1. Direct answer to the query
            2. Available items/equipment (if relevant)
            3. Related abilities and powers
            4. Important locations
            5. Additional relevant details

            Assistant:"""
        )
        
        # Initialize the QA chain with primary model
        self.qa_chain = self._create_qa_chain(self.primary_llm)
        
    def _create_llm(self, model_name: str) -> ChatOpenAI:
        """Creates a ChatOpenAI instance with retry logic for rate limits."""
        @retry(
            stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", 3))),
            wait=wait_exponential(
                multiplier=float(os.getenv("INITIAL_RETRY_DELAY", 1)),
                max=float(os.getenv("MAX_RETRY_DELAY", 60))
            )
        )
        def create_with_retry():
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature
            )
        return create_with_retry()
    
    def _create_qa_chain(self, llm: Any) -> ConversationalRetrievalChain:
        """Creates a ConversationalRetrievalChain with the specified LLM."""
        try:
            # Create the filtered retriever
            base_retriever = self.validator.vector_store.as_retriever()
            filtered_retriever = FilteredRetriever(
                base_retriever=base_retriever,
                vector_store=self.validator.vector_store
            )
            
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=filtered_retriever,
                combine_docs_chain_kwargs={"prompt": self.base_prompt}
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            # Fallback to simple retriever if filtered retriever fails
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.validator.vector_store.as_retriever(),
                combine_docs_chain_kwargs={"prompt": self.base_prompt}
            )
    
    @retry(
        stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", 3))),
        wait=wait_exponential(
            multiplier=float(os.getenv("INITIAL_RETRY_DELAY", 1)),
            max=float(os.getenv("MAX_RETRY_DELAY", 60))
        )
    )
    def _process_with_fallback(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Processes a request with fallback to GPT-4 if needed."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() and self.fallback_llm:
                logger.info("Rate limit hit, falling back to GPT-4")
                # Temporarily swap to fallback model
                original_chain = self.qa_chain
                self.qa_chain = self._create_qa_chain(self.fallback_llm)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Restore primary model
                    self.qa_chain = original_chain
            raise
            
    def _process_lore_submission(self, submission: str) -> Dict[str, Any]:
        """
        Processes a new lore submission for validation.
        
        Args:
            submission: The new lore content to validate
            
        Returns:
            Dict containing validation results and feedback
        """
        try:
            # Check for conflicts
            validation = self.validator.check_conflict(submission)
            
            # Generate appropriate response based on validation results
            if validation["conflicts"]:
                response = (
                    "⚠️ I've detected some potential conflicts with existing lore:\n\n"
                )
                
                for conflict in validation["conflicts"]:
                    # Handle conflicts with or without metadata
                    if 'metadata' in conflict and conflict['metadata'].get('name'):
                        response += f"- Conflicts with: {conflict['metadata']['name']}\n"
                    else:
                        response += f"- Conflict: {conflict['reason']}\n"
                        
                    if conflict.get('content'):
                        response += f"  Details: {conflict['content']}\n\n"
                    
                response += "\nSuggested improvements:\n"
                # Use LLM to generate improvement suggestions
                suggestions_prompt = f"""
                Given this new lore submission:
                {submission}

                And these conflicts with existing lore:
                {validation['conflicts']}

                Provide 2-3 specific suggestions for modifying the submission to better align with canon.
                Focus on maintaining the core idea while resolving conflicts.
                """
                suggestions = self.primary_llm.invoke(suggestions_prompt).content
                response += suggestions
                
            else:
                response = "✅ Your submission appears to align well with existing lore!\n\n"
                if validation["similar_concepts"]:
                    response += "Related existing elements you might want to reference:\n"
                    for concept in validation["similar_concepts"]:
                        if 'metadata' in concept and concept['metadata'].get('name'):
                            response += f"- {concept['metadata']['name']}: {concept['content']}\n"
                        else:
                            response += f"- Related: {concept['content']}\n"
            
            return {
                "response": response,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"Error processing lore submission: {str(e)}")
            raise
            
    def generate_response(
        self,
        user_input: str,
        chat_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response to user input, handling both queries and submissions.
        
        Args:
            user_input: The user's message
            chat_history: Optional list of previous messages
            
        Returns:
            Dict containing response and any validation results
        """
        try:
            # Handle lore submissions (prefixed with [SUBMIT])
            if user_input.startswith("[SUBMIT]"):
                submission_content = user_input[8:].strip()  # Remove [SUBMIT] prefix
                return self._process_with_fallback(
                    self._process_lore_submission,
                    submission_content
                )
            
            # Handle normal queries
            return self._process_with_fallback(
                lambda: {
                    "response": self.qa_chain.invoke({
                        "question": user_input,
                        "chat_history": chat_history or []
                    })["answer"],
                    "validation": None
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 