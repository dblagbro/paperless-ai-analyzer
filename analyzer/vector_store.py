"""
Vector Store for Document Embeddings using Cohere + ChromaDB

Implements RAG (Retrieval Augmented Generation) for semantic document search.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import cohere
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and semantic search using Cohere + ChromaDB."""

    def __init__(self, cohere_api_key: Optional[str] = None, persist_directory: str = '/app/data/chroma'):
        """
        Initialize vector store.

        Args:
            cohere_api_key: Cohere API key for embeddings
            persist_directory: Where to persist the ChromaDB database
        """
        self.cohere_api_key = cohere_api_key or os.environ.get('COHERE_API_KEY', '')

        if not self.cohere_api_key:
            logger.warning("No Cohere API key provided - vector store disabled")
            self.enabled = False
            return

        try:
            # Initialize Cohere client
            self.cohere_client = cohere.Client(self.cohere_api_key)

            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"description": "Financial document embeddings"}
            )

            self.enabled = True
            logger.info(f"Vector store initialized with {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.enabled = False

    def embed_document(self, document_id: int, title: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Embed a document and store in vector database.

        Args:
            document_id: Paperless document ID
            title: Document title
            content: Document content (AI analysis, transactions, etc.)
            metadata: Additional metadata (anomalies, risk score, document_type, etc.)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            # Prepare text for embedding (combine title + content)
            text_to_embed = f"Title: {title}\n\nContent:\n{content}"

            # Generate embedding with Cohere
            response = self.cohere_client.embed(
                texts=[text_to_embed],
                model="embed-english-v3.0",
                input_type="search_document"
            )

            embedding = response.embeddings[0]

            # Store in ChromaDB - recreate collection if it doesn't exist
            try:
                self.collection.upsert(
                    ids=[str(document_id)],
                    embeddings=[embedding],
                    documents=[text_to_embed],
                    metadatas=[{
                        'document_id': document_id,
                        'title': title,
                        'risk_score': metadata.get('risk_score', 0),
                        'anomalies': ','.join(metadata.get('anomalies', [])),
                        'timestamp': metadata.get('timestamp', ''),
                        'document_type': metadata.get('document_type', 'unknown')
                    }]
                )
            except Exception as collection_error:
                # Collection might have been deleted - recreate it
                if "does not exist" in str(collection_error):
                    logger.warning(f"Collection was deleted, recreating it")
                    self.collection = self.chroma_client.get_or_create_collection(
                        name="documents",
                        metadata={"description": "Financial document embeddings"}
                    )
                    # Retry the upsert
                    self.collection.upsert(
                        ids=[str(document_id)],
                        embeddings=[embedding],
                        documents=[text_to_embed],
                        metadatas=[{
                            'document_id': document_id,
                            'title': title,
                            'risk_score': metadata.get('risk_score', 0),
                            'anomalies': ','.join(metadata.get('anomalies', [])),
                            'timestamp': metadata.get('timestamp', ''),
                            'document_type': metadata.get('document_type', 'unknown')
                        }]
                    )
                else:
                    raise

            logger.info(f"Embedded document {document_id}: {title} (type: {metadata.get('document_type', 'unknown')})")
            return True

        except Exception as e:
            logger.error(f"Failed to embed document {document_id}: {e}")
            return False

    def search(self, query: str, n_results: int = 10, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents.

        Args:
            query: User's search query
            n_results: Number of results to return
            document_type: Optional filter by document type (e.g., 'bank_statement', 'invoice')

        Returns:
            List of relevant documents with metadata
        """
        if not self.enabled:
            return []

        try:
            # Generate query embedding
            response = self.cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )

            query_embedding = response.embeddings[0]

            # Build query parameters
            query_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }

            # Add document type filter if specified
            if document_type:
                query_params['where'] = {'document_type': document_type}
                logger.info(f"Filtering search by document_type: {document_type}")

            # Search ChromaDB - recreate collection if it doesn't exist
            try:
                results = self.collection.query(**query_params)
            except Exception as collection_error:
                if "does not exist" in str(collection_error):
                    logger.warning(f"Collection was deleted, recreating it")
                    self.collection = self.chroma_client.get_or_create_collection(
                        name="documents",
                        metadata={"description": "Financial document embeddings"}
                    )
                    # Return empty results since collection was just recreated
                    return []
                else:
                    raise

            # Format results
            documents = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    doc = {
                        'document_id': results['metadatas'][0][i]['document_id'],
                        'title': results['metadatas'][0][i]['title'],
                        'content': results['documents'][0][i],
                        'risk_score': results['metadatas'][0][i].get('risk_score', 0),
                        'anomalies': results['metadatas'][0][i].get('anomalies', '').split(',') if results['metadatas'][0][i].get('anomalies') else [],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'timestamp': results['metadatas'][0][i].get('timestamp', ''),
                        'document_type': results['metadatas'][0][i].get('document_type', 'unknown')
                    }
                    documents.append(doc)

            logger.info(f"Found {len(documents)} relevant documents for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics including breakdown by document type."""
        if not self.enabled:
            return {'enabled': False}

        try:
            # Recreate collection if it doesn't exist
            try:
                count = self.collection.count()
            except Exception as collection_error:
                if "does not exist" in str(collection_error):
                    logger.warning(f"Collection was deleted, recreating it")
                    self.collection = self.chroma_client.get_or_create_collection(
                        name="documents",
                        metadata={"description": "Financial document embeddings"}
                    )
                    count = 0
                else:
                    raise

            # Get breakdown by document type
            type_breakdown = {}
            if count > 0:
                all_docs = self.collection.get(include=['metadatas'])
                for metadata in all_docs['metadatas']:
                    doc_type = metadata.get('document_type', 'unknown')
                    type_breakdown[doc_type] = type_breakdown.get(doc_type, 0) + 1

            return {
                'enabled': True,
                'total_documents': count,
                'embedding_model': 'cohere/embed-english-v3.0',
                'vector_db': 'ChromaDB',
                'by_type': type_breakdown
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'enabled': False, 'error': str(e)}

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a specific document from vector store.

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            self.collection.delete(ids=[str(document_id)])
            logger.info(f"Deleted document {document_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def delete_by_type(self, document_type: str) -> int:
        """
        Delete all documents of a specific type.

        Args:
            document_type: Document type to delete (e.g., 'bank_statement')

        Returns:
            Number of documents deleted
        """
        if not self.enabled:
            return 0

        try:
            # Get all documents of this type
            results = self.collection.get(
                where={'document_type': document_type},
                include=['metadatas']
            )

            if results['ids']:
                count = len(results['ids'])
                self.collection.delete(where={'document_type': document_type})
                logger.info(f"Deleted {count} documents of type '{document_type}'")
                return count
            else:
                logger.info(f"No documents found with type '{document_type}'")
                return 0

        except Exception as e:
            logger.error(f"Failed to delete documents by type '{document_type}': {e}")
            return 0

    def get_document_types(self) -> List[str]:
        """
        Get list of all unique document types in the vector store.

        Returns:
            List of document types
        """
        if not self.enabled:
            return []

        try:
            all_docs = self.collection.get(include=['metadatas'])
            types = set()
            for metadata in all_docs['metadatas']:
                doc_type = metadata.get('document_type', 'unknown')
                types.add(doc_type)

            return sorted(list(types))

        except Exception as e:
            logger.error(f"Failed to get document types: {e}")
            return []

    def clear(self) -> bool:
        """Clear all embeddings from vector store."""
        if not self.enabled:
            return False

        try:
            self.chroma_client.delete_collection("documents")
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"description": "Financial document embeddings"}
            )
            logger.info("Vector store cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            return False
