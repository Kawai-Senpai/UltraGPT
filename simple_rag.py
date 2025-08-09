"""
Simple Search-based RAG System for UltraGPT
A lightweight RAG system that uses keyword matching instead of vector search.
Now with persistent file-based storage for efficient reuse.
"""

import re
import string
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import hashlib

class SimpleRAG:
    """
    A simple search-based RAG system that uses keyword matching for retrieval.
    Features persistent file-based storage for efficient reuse.
    """
    
    def __init__(self, storage_dir: str, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the SimpleRAG system with persistent storage.
        
        Args:
            storage_dir (str): Directory to store chunked data and indices
            chunk_size (int): Maximum size of each text chunk
            overlap (int): Number of characters to overlap between chunks
        """
        self.storage_dir = Path(storage_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for persistent storage
        self.documents_file = self.storage_dir / "documents.json"
        self.label_index_file = self.storage_dir / "label_index.json"
        self.keyword_index_file = self.storage_dir / "keyword_index.json"
        self.config_file = self.storage_dir / "config.json"
        
        # Initialize data structures
        self.documents = {}  # {doc_id: {"label": str, "chunks": List[str], "keywords": List[set]}}
        self.label_index = defaultdict(list)  # {label: [doc_ids]}
        self.keyword_index = defaultdict(set)  # {keyword: {doc_ids}}
        
        # Load existing data if available
        self._load_from_disk()
        
    def _save_to_disk(self):
        """Save all data structures to disk."""
        try:
            # Save configuration
            config = {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "version": "1.0"
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            # Convert sets to lists for JSON serialization
            documents_serializable = {}
            for doc_id, doc in self.documents.items():
                documents_serializable[doc_id] = {
                    "label": doc["label"],
                    "content": doc["content"],
                    "chunks": doc["chunks"],
                    "keywords": [list(keywords) for keywords in doc["keywords"]]
                }
            
            # Save documents
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(documents_serializable, f, indent=2)
            
            # Save label index
            with open(self.label_index_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.label_index), f, indent=2)
            
            # Save keyword index (convert sets to lists)
            keyword_index_serializable = {}
            for keyword, doc_ids in self.keyword_index.items():
                keyword_index_serializable[keyword] = list(doc_ids)
            
            with open(self.keyword_index_file, 'w', encoding='utf-8') as f:
                json.dump(keyword_index_serializable, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save RAG data to disk: {e}")
    
    def _load_from_disk(self):
        """Load existing data from disk if available."""
        try:
            # Check if files exist
            if not all([f.exists() for f in [self.documents_file, self.label_index_file, self.keyword_index_file]]):
                print(f"No existing RAG data found in {self.storage_dir}. Starting fresh.")
                return
            
            # Load configuration and validate compatibility
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config.get("chunk_size") != self.chunk_size or config.get("overlap") != self.overlap:
                        print(f"Warning: Existing data has different chunk settings. Existing: chunk_size={config.get('chunk_size')}, overlap={config.get('overlap')}. Current: chunk_size={self.chunk_size}, overlap={self.overlap}")
            
            # Load documents
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
                for doc_id, doc in documents_data.items():
                    self.documents[doc_id] = {
                        "label": doc["label"],
                        "content": doc["content"],
                        "chunks": doc["chunks"],
                        "keywords": [set(keywords) for keywords in doc["keywords"]]
                    }
            
            # Load label index
            with open(self.label_index_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                self.label_index = defaultdict(list, label_data)
            
            # Load keyword index
            with open(self.keyword_index_file, 'r', encoding='utf-8') as f:
                keyword_data = json.load(f)
                self.keyword_index = defaultdict(set)
                for keyword, doc_ids in keyword_data.items():
                    self.keyword_index[keyword] = set(doc_ids)
            
            print(f"Loaded existing RAG data: {len(self.documents)} documents, {len(self.label_index)} labels, {len(self.keyword_index)} keywords")
            
        except Exception as e:
            print(f"Warning: Failed to load existing RAG data: {e}. Starting fresh.")
            self.documents = {}
            self.label_index = defaultdict(list)
            self.keyword_index = defaultdict(set)
        
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
        
        # Save to disk after adding new document
        self._save_to_disk()
        
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
        
        # Note: _save_to_disk() is called by add_document() for each document
        # For bulk operations, we could optimize this, but it ensures data consistency
        
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
        """Clear all documents from the RAG system and delete stored files."""
        self.documents.clear()
        self.label_index.clear()
        self.keyword_index.clear()
        
        # Delete storage files
        for file_path in [self.documents_file, self.label_index_file, self.keyword_index_file, self.config_file]:
            if file_path.exists():
                file_path.unlink()
        
        print(f"Cleared all RAG data from {self.storage_dir}")
    
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
        
        # Save changes to disk
        self._save_to_disk()
        
        print(f"Removed {len(doc_ids_to_remove)} documents with label '{label}'")
    
    def add_documents_bulk(self, documents_dict: Dict[str, List[str]], auto_save: bool = True):
        """
        Add multiple documents efficiently with bulk operations.
        
        Args:
            documents_dict (Dict[str, List[str]]): {label: [documents]} format
            auto_save (bool): Whether to save to disk after bulk operation
        """
        total_added = 0
        for label, documents in documents_dict.items():
            for i, doc in enumerate(documents):
                doc_id = f"{label}_doc_{len(self.label_index[label]) + i}"
                
                # Chunk the content
                chunks = self._chunk_text(doc)
                
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
                    "content": doc,
                    "chunks": chunks,
                    "keywords": chunk_keywords
                }
                
                # Update label index
                self.label_index[label].append(doc_id)
                total_added += 1
        
        if auto_save:
            self._save_to_disk()
        
        print(f"Added {total_added} documents in bulk operation")
        return total_added
    
    def get_storage_info(self) -> Dict:
        """Get information about the storage directory and files."""
        storage_info = {
            "storage_dir": str(self.storage_dir),
            "storage_exists": self.storage_dir.exists(),
            "files": {}
        }
        
        for file_name, file_path in [
            ("documents", self.documents_file),
            ("label_index", self.label_index_file),
            ("keyword_index", self.keyword_index_file),
            ("config", self.config_file)
        ]:
            storage_info["files"][file_name] = {
                "exists": file_path.exists(),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }
        
        return storage_info
