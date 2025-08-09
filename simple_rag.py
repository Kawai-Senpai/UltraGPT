"""
Simple Search-based RAG System for UltraGPT
A lightweight RAG system that uses keyword matching instead of vector search.
"""

import re
import string
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import hashlib

class SimpleRAG:
    """
    A simple search-based RAG system that uses keyword matching for retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the SimpleRAG system.
        
        Args:
            chunk_size (int): Maximum size of each text chunk
            overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = {}  # {doc_id: {"label": str, "chunks": List[str], "keywords": List[set]}}
        self.label_index = defaultdict(list)  # {label: [doc_ids]}
        self.keyword_index = defaultdict(set)  # {keyword: {doc_ids}}
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and removing punctuation."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text by breaking down into words."""
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'
        }
        
        keywords = set()
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.add(word)
                # Add word stems (simple version - just remove common suffixes)
                if word.endswith('ing') and len(word) > 6:
                    keywords.add(word[:-3])
                elif word.endswith('ed') and len(word) > 5:
                    keywords.add(word[:-2])
                elif word.endswith('s') and len(word) > 3:
                    keywords.add(word[:-1])
        
        return keywords
    
    def _chunk_text(self, text: str) -> List[str]:
        """Break text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look back for a space
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
                
        return chunks
    
    def add_document(self, content: str, label: str, doc_id: Optional[str] = None) -> str:
        """
        Add a document to the RAG system.
        
        Args:
            content (str): The text content to add
            label (str): Label/category for this document
            doc_id (Optional[str]): Unique ID for document, auto-generated if None
            
        Returns:
            str: The document ID
        """
        if doc_id is None:
            # Generate unique doc_id based on content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            doc_id = f"{label}_{content_hash[:8]}"
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        # Extract keywords for each chunk
        chunk_keywords = []
        for chunk in chunks:
            keywords = self._extract_keywords(chunk)
            chunk_keywords.append(keywords)
            
            # Update keyword index
            for keyword in keywords:
                self.keyword_index[keyword].add(doc_id)
        
        # Store document
        self.documents[doc_id] = {
            "label": label,
            "content": content,
            "chunks": chunks,
            "keywords": chunk_keywords
        }
        
        # Update label index
        self.label_index[label].append(doc_id)
        
        return doc_id
    
    def add_documents_from_list(self, documents: List[str], label: str) -> List[str]:
        """
        Add multiple documents from a list.
        
        Args:
            documents (List[str]): List of text documents
            label (str): Label for all documents
            
        Returns:
            List[str]: List of document IDs
        """
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = self.add_document(doc, label, f"{label}_doc_{i}")
            doc_ids.append(doc_id)
        return doc_ids
    
    def search(self, query: str, top_k: int = 5, labels: Optional[List[str]] = None) -> List[Tuple[str, float, str, str]]:
        """
        Search for relevant chunks based on query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            labels (Optional[List[str]]): Filter by specific labels
            
        Returns:
            List[Tuple[str, float, str, str]]: List of (chunk_text, score, doc_id, label)
        """
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Filter documents by labels if specified
        candidate_docs = set()
        if labels:
            for label in labels:
                candidate_docs.update(self.label_index.get(label, []))
        else:
            candidate_docs = set(self.documents.keys())
        
        # Score each chunk
        scored_chunks = []
        
        for doc_id in candidate_docs:
            if doc_id not in self.documents:
                continue
                
            doc = self.documents[doc_id]
            chunks = doc["chunks"]
            chunk_keywords_list = doc["keywords"]
            
            for i, (chunk, chunk_keywords) in enumerate(zip(chunks, chunk_keywords_list)):
                # Calculate keyword overlap score
                common_keywords = query_keywords.intersection(chunk_keywords)
                if not common_keywords:
                    continue
                
                # Simple scoring: (number of matching keywords) / (total unique keywords)
                total_keywords = len(query_keywords.union(chunk_keywords))
                score = len(common_keywords) / total_keywords if total_keywords > 0 else 0
                
                # Boost score based on keyword frequency in chunk
                chunk_lower = chunk.lower()
                frequency_boost = 0
                for keyword in common_keywords:
                    frequency_boost += chunk_lower.count(keyword)
                
                final_score = score + (frequency_boost * 0.1)  # Small boost for frequency
                
                scored_chunks.append((chunk, final_score, doc_id, doc["label"]))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def get_relevant_context(self, query: str, system_message: str = "", top_k: int = 3, labels: Optional[List[str]] = None) -> str:
        """
        Get relevant context for a query, formatted for appending to system message.
        
        Args:
            query (str): The user query
            system_message (str): Current system message content
            top_k (int): Number of relevant chunks to include
            labels (Optional[List[str]]): Filter by specific labels
            
        Returns:
            str: Formatted context to append to system message
        """
        # Combine system message and query for better matching
        combined_query = f"{system_message} {query}"
        
        results = self.search(combined_query, top_k=top_k, labels=labels)
        
        if not results:
            return ""
        
        context_parts = []
        context_parts.append("RELEVANT CONTEXT FROM KNOWLEDGE BASE:")
        context_parts.append("")
        
        for i, (chunk, score, doc_id, label) in enumerate(results, 1):
            context_parts.append(f"[Context {i} - Label: {label} - Relevance: {score:.2f}]")
            context_parts.append(chunk)
            context_parts.append("")
        
        context_parts.append("Use the above context to inform your response when relevant.")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system."""
        total_chunks = sum(len(doc["chunks"]) for doc in self.documents.values())
        labels = list(self.label_index.keys())
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_keywords": len(self.keyword_index),
            "labels": labels,
            "documents_per_label": {label: len(docs) for label, docs in self.label_index.items()}
        }
    
    def clear(self):
        """Clear all documents from the RAG system."""
        self.documents.clear()
        self.label_index.clear()
        self.keyword_index.clear()
    
    def remove_label(self, label: str):
        """Remove all documents with a specific label."""
        if label not in self.label_index:
            return
        
        doc_ids_to_remove = self.label_index[label].copy()
        
        for doc_id in doc_ids_to_remove:
            if doc_id in self.documents:
                # Remove from keyword index
                for chunk_keywords in self.documents[doc_id]["keywords"]:
                    for keyword in chunk_keywords:
                        self.keyword_index[keyword].discard(doc_id)
                        if not self.keyword_index[keyword]:
                            del self.keyword_index[keyword]
                
                # Remove document
                del self.documents[doc_id]
        
        # Remove label
        del self.label_index[label]
