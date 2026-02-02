import os
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from .models import LayoutChunk, Document, QueryResult, LayoutType


class VectorStore:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "document_layouts",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Cache for document metadata
        self.document_metadata_file = os.path.join(persist_directory, "documents.json")
        self.documents_cache = self._load_document_cache()
    
    def add_document(self, document: Document) -> bool:
        if not document.chunks:
            print(f"No chunks to add for document {document.document_id}")
            return False
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in document.chunks:
            if not chunk.text or chunk.text.strip() == "":
                continue
            
            chunk_id = f"{document.document_id}_{chunk.layout_id}"
            ids.append(chunk_id)
            
            text_with_context = self._create_context_text(chunk, document)
            embedding = self.embedding_model.encode(text_with_context).tolist()
            embeddings.append(embedding)
            
            # Store the actual text
            documents.append(chunk.text)
            
            # Create metadata
            metadata = {
                "document_id": document.document_id,
                "layout_id": chunk.layout_id,
                "layout_type": chunk.layout_type.value,
                "page_number": chunk.page_number,
                "reading_order": chunk.reading_order,
                "confidence": chunk.confidence,
                "ocr_confidence": chunk.ocr_confidence,
                "is_handwritten": chunk.is_handwritten,
                "parent_id": chunk.parent_id or "",
                "children_ids": ",".join(chunk.children_ids),
                "bbox": json.dumps(chunk.bbox.to_dict()) if chunk.bbox else "",
            }
            metadatas.append(metadata)
        
        if not ids:
            print(f"No valid chunks with text for document {document.document_id}")
            return False
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # Save document metadata
            self.documents_cache[document.document_id] = document.to_dict()
            self._save_document_cache()
            
            print(f"Added {len(ids)} chunks from document {document.document_id}")
            return True
            
        except Exception as e:
            print(f"Error adding document to vector store: {str(e)}")
            return False
    
    def query(self, 
              query_text: str, 
              n_results: int = 5,
              filter_dict: Optional[Dict] = None,
              include_context: bool = True) -> List[QueryResult]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"layout_type": "text"})
            include_context: Whether to include surrounding chunks for context
            
        Returns:
            List of QueryResult objects with similarity scores and context
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Parse results
        query_results = []
        
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            similarity_score = 1 - distance  # Convert distance to similarity
            
            metadata = results['metadatas'][0][i]
            text = results['documents'][0][i]
            
            # Reconstruct chunk
            chunk = self._reconstruct_chunk(metadata, text)
            
            # Get context chunks if requested
            context_chunks = []
            if include_context:
                context_chunks = self._get_context_chunks(
                    metadata['document_id'],
                    chunk.page_number,
                    chunk.reading_order
                )
            
            query_result = QueryResult(
                chunk=chunk,
                similarity_score=similarity_score,
                document_id=metadata['document_id'],
                context_chunks=context_chunks
            )
            
            query_results.append(query_result)
        
        return query_results
    
    def query_by_layout_type(self, 
                            query_text: str,
                            layout_types: List[LayoutType],
                            n_results: int = 5) -> List[QueryResult]:
        """
        Query for specific layout types only.
        
        Args:
            query_text: Query text
            layout_types: List of layout types to search
            n_results: Number of results
            
        Returns:
            List of QueryResult objects
        """
        # Create filter for layout types
        type_values = [lt.value for lt in layout_types]
        
        # ChromaDB $in operator
        filter_dict = {
            "layout_type": {"$in": type_values}
        }
        
        return self.query(query_text, n_results, filter_dict)
    
    def get_document_chunks(self, document_id: str, page_number: Optional[int] = None) -> List[LayoutChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID
            page_number: Optional page number filter
            
        Returns:
            List of LayoutChunk objects
        """
        # Build filter based on ChromaDB new API requirements
        if page_number is not None:
            filter_dict = {
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"page_number": {"$eq": page_number}}
                ]
            }
        else:
            filter_dict = {"document_id": {"$eq": document_id}}
        
        # Get all chunks (use large limit)
        results = self.collection.get(
            where=filter_dict,
            limit=1000
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            text = results['documents'][i]
            chunk = self._reconstruct_chunk(metadata, text)
            chunks.append(chunk)
        
        # Sort by reading order
        chunks.sort(key=lambda x: (x.page_number, x.reading_order))
        
        return chunks
    
    def _create_context_text(self, chunk: LayoutChunk, document: Document) -> str:
        """
        Create enriched text with layout context for better embeddings.
        Includes layout type, position, and hierarchical information.
        """
        context_parts = []
        
        # Add layout type context
        context_parts.append(f"[{chunk.layout_type.value.upper()}]")
        
        # Add parent context if exists
        if chunk.parent_id:
            parent_chunk = document.get_chunk_by_id(chunk.parent_id)
            if parent_chunk and parent_chunk.text:
                context_parts.append(f"Related to: {parent_chunk.text[:100]}")
        
        # Add the main text
        context_parts.append(chunk.text)
        
        # Add children context for figures/tables
        if chunk.layout_type in [LayoutType.FIGURE, LayoutType.TABLE] and chunk.children_ids:
            for child_id in chunk.children_ids[:2]:  # Limit to 2 children
                child_chunk = document.get_chunk_by_id(child_id)
                if child_chunk and child_chunk.text:
                    context_parts.append(f"Caption: {child_chunk.text[:100]}")
        
        return " ".join(context_parts)
    
    def _get_context_chunks(self, 
                           document_id: str,
                           page_number: int,
                           reading_order: int,
                           window: int = 2) -> List[LayoutChunk]:
        """
        Get surrounding chunks for context.
        
        Args:
            document_id: Document ID
            page_number: Page number
            reading_order: Reading order of the query chunk
            window: Number of chunks before and after to include
            
        Returns:
            List of context chunks
        """
        # Get all chunks from the same page
        page_chunks = self.get_document_chunks(document_id, page_number)
        
        # Find chunks within the window
        context_chunks = []
        for chunk in page_chunks:
            order_diff = abs(chunk.reading_order - reading_order)
            if 0 < order_diff <= window:
                context_chunks.append(chunk)
        
        # Sort by reading order
        context_chunks.sort(key=lambda x: x.reading_order)
        
        return context_chunks
    
    def _reconstruct_chunk(self, metadata: Dict, text: str) -> LayoutChunk:
        """Reconstruct LayoutChunk from metadata and text"""
        from .models import BoundingBox
        
        # Parse bbox if exists
        bbox = None
        if metadata.get('bbox'):
            try:
                bbox_dict = json.loads(metadata['bbox'])
                bbox = BoundingBox(**bbox_dict)
            except:
                pass
        
        # Parse children IDs
        children_ids = []
        if metadata.get('children_ids'):
            children_ids = metadata['children_ids'].split(',')
            children_ids = [cid for cid in children_ids if cid]
        
        chunk = LayoutChunk(
            layout_id=metadata['layout_id'],
            layout_type=LayoutType(metadata['layout_type']),
            bbox=bbox,
            page_number=metadata['page_number'],
            confidence=metadata['confidence'],
            parent_id=metadata['parent_id'] if metadata['parent_id'] else None,
            children_ids=children_ids,
            reading_order=metadata['reading_order'],
            text=text,
            ocr_confidence=metadata['ocr_confidence'],
            is_handwritten=metadata['is_handwritten']
        )
        
        return chunk
    
    def _load_document_cache(self) -> Dict:
        """Load document metadata cache"""
        if os.path.exists(self.document_metadata_file):
            try:
                with open(self.document_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_document_cache(self):
        """Save document metadata cache"""
        os.makedirs(os.path.dirname(self.document_metadata_file), exist_ok=True)
        with open(self.document_metadata_file, 'w') as f:
            json.dump(self.documents_cache, f, indent=2)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                
                # Remove from cache
                if document_id in self.documents_cache:
                    del self.documents_cache[document_id]
                    self._save_document_cache()
                
                print(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                print(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the vector store"""
        total_chunks = self.collection.count()
        num_documents = len(self.documents_cache)
        
        # Get layout type distribution
        layout_types = {}
        try:
            for doc_data in self.documents_cache.values():
                for chunk_data in doc_data.get('chunks', []):
                    lt = chunk_data.get('layout_type', 'unknown')
                    layout_types[lt] = layout_types.get(lt, 0) + 1
        except:
            pass
        
        return {
            "total_chunks": total_chunks,
            "total_documents": num_documents,
            "layout_type_distribution": layout_types,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name
        }


class HybridVectorStore(VectorStore):
    """
    Enhanced vector store with hybrid search capabilities.
    Combines semantic search with keyword matching for better retrieval.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Build keyword index
        self.keyword_index = {}
    
    def add_document(self, document: Document) -> bool:
        """Add document with keyword indexing"""
        success = super().add_document(document)
        
        if success:
            # Build keyword index for this document
            self._index_keywords(document)
        
        return success
    
    def hybrid_query(self,
                    query_text: str,
                    n_results: int = 5,
                    semantic_weight: float = 0.7) -> List[QueryResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query_text: Query text
            n_results: Number of results
            semantic_weight: Weight for semantic similarity (0-1)
            
        Returns:
            List of QueryResult objects
        """
        # Get semantic results
        semantic_results = self.query(query_text, n_results * 2, include_context=False)
        
        # Get keyword matches
        keyword_scores = self._keyword_search(query_text)
        
        # Combine scores
        combined_scores = {}
        
        for result in semantic_results:
            chunk_id = f"{result.document_id}_{result.chunk.layout_id}"
            combined_scores[chunk_id] = {
                'result': result,
                'score': result.similarity_score * semantic_weight
            }
        
        # Add keyword scores
        for chunk_id, kw_score in keyword_scores.items():
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['score'] += kw_score * (1 - semantic_weight)
            else:
                # Get this chunk and add it
                pass  # Could add pure keyword matches here
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:n_results]
        
        # Add context to final results
        final_results = []
        for item in sorted_results:
            result = item['result']
            result.similarity_score = item['score']
            result.context_chunks = self._get_context_chunks(
                result.document_id,
                result.chunk.page_number,
                result.chunk.reading_order
            )
            final_results.append(result)
        
        return final_results
    
    def _index_keywords(self, document: Document):
        """Build keyword index for a document"""
        for chunk in document.chunks:
            if not chunk.text:
                continue
            
            chunk_id = f"{document.document_id}_{chunk.layout_id}"
            
            # Simple keyword extraction (can be enhanced with NLP)
            words = chunk.text.lower().split()
            
            for word in words:
                if len(word) > 3:  # Skip short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(chunk_id)
    
    def _keyword_search(self, query_text: str) -> Dict[str, float]:
        """Search keyword index"""
        query_words = query_text.lower().split()
        
        chunk_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for chunk_id in self.keyword_index[word]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1
        
        # Normalize scores
        if chunk_scores:
            max_score = max(chunk_scores.values())
            chunk_scores = {k: v / max_score for k, v in chunk_scores.items()}
        
        return chunk_scores
