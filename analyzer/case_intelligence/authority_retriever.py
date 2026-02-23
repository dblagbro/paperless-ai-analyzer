"""
Authority Retriever — searches the shared authority_corpus ChromaDB collection.

The authority_corpus collection contains embeddings of legal authorities
(statutes, regulations, case law) fetched by AuthorityIngester.
Retrieval is filtered by jurisdiction profile.
"""

import logging
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

AUTHORITY_COLLECTION_NAME = 'authority_corpus'


class AuthorityRetriever:
    """
    Searches the shared authority_corpus ChromaDB collection.

    Uses Cohere embed-english-v3.0 with input_type='search_query' for retrieval.
    Filters results by the run's JurisdictionProfile.authority_jurisdictions.
    """

    def __init__(self, cohere_api_key: Optional[str] = None,
                 persist_directory: str = '/app/data/chroma'):
        self.cohere_api_key = cohere_api_key
        self.persist_directory = persist_directory
        self.enabled = False
        self.collection = None
        self.cohere_client = None

        if not cohere_api_key:
            logger.warning("AuthorityRetriever: no Cohere API key — authority search disabled")
            return

        try:
            import cohere
            import chromadb
            from chromadb.config import Settings

            self.cohere_client = cohere.Client(cohere_api_key)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name=AUTHORITY_COLLECTION_NAME,
                metadata={
                    'description': 'Shared legal authority corpus (statutes, regulations, case law)',
                },
            )
            self.enabled = True
            logger.info(
                f"AuthorityRetriever initialized: {self.collection.count()} authorities in corpus"
            )
        except Exception as e:
            logger.error(f"AuthorityRetriever init failed: {e}")

    def embed_pending_authorities(self, batch_size: int = 50) -> int:
        """
        Embed any authorities in ci_authority_corpus that haven't been embedded yet.
        Returns count of newly embedded documents.
        """
        if not self.enabled:
            return 0

        from analyzer.case_intelligence.db import get_unembedded_authorities, mark_authority_embedded

        pending = get_unembedded_authorities(limit=batch_size)
        if not pending:
            return 0

        count = 0
        for authority in pending:
            try:
                content = authority['content_text'] or ''
                if not content.strip():
                    continue

                # Chunk if too long
                chunks = self._chunk_text(content, max_tokens=400)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{authority['id']}_chunk_{i}"
                    embedding = self._embed_text(chunk)
                    if embedding:
                        self.collection.upsert(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                'corpus_id': authority['id'],
                                'citation': authority['citation'],
                                'source': authority['source'],
                                'jurisdiction': authority['jurisdiction'],
                                'authority_type': authority['authority_type'],
                                'reliability': authority['reliability'],
                                'title': authority['title'] or '',
                                'source_url': authority['source_url'] or '',
                            }],
                        )

                mark_authority_embedded(authority['id'])
                count += 1

            except Exception as e:
                logger.error(f"AuthorityRetriever: failed to embed {authority['id']}: {e}")

        logger.info(f"AuthorityRetriever: embedded {count} authorities")
        return count

    def search(self, query: str, jurisdiction_jurisdictions: List[str] = None,
               n_results: int = 8, authority_type: str = None) -> List[Dict[str, Any]]:
        """
        Search the authority corpus for relevant authorities.

        Args:
            query: Natural language search query
            jurisdiction_jurisdictions: List of jurisdiction codes to filter by
            n_results: Number of results to return
            authority_type: Optional filter (statute|regulation|case_law|rule)

        Returns:
            List of authority dicts with citation, excerpt, source, relevance score.
        """
        if not self.enabled or self.collection.count() == 0:
            return []

        try:
            query_embedding = self._embed_text(query, input_type='search_query')
            if not query_embedding:
                return []

            # Build where filter
            where = {}
            if jurisdiction_jurisdictions:
                where['jurisdiction'] = {'$in': jurisdiction_jurisdictions}
            if authority_type:
                where['authority_type'] = authority_type

            query_params = {
                'query_embeddings': [query_embedding],
                'n_results': min(n_results, max(1, self.collection.count())),
                'include': ['documents', 'metadatas', 'distances'],
            }
            if where:
                query_params['where'] = where

            results = self.collection.query(**query_params)

            authorities = []
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i in range(len(ids)):
                meta = metas[i]
                authorities.append({
                    'citation': meta.get('citation', ''),
                    'title': meta.get('title', ''),
                    'source': meta.get('source', ''),
                    'jurisdiction': meta.get('jurisdiction', ''),
                    'authority_type': meta.get('authority_type', ''),
                    'reliability': meta.get('reliability', 'official'),
                    'source_url': meta.get('source_url', ''),
                    'excerpt': docs[i][:500],
                    'relevance_score': round(1.0 - distances[i], 4),
                })

            # Sort by relevance, highest first
            authorities.sort(key=lambda x: x['relevance_score'], reverse=True)
            return authorities

        except Exception as e:
            logger.error(f"AuthorityRetriever.search failed: {e}")
            return []

    def _embed_text(self, text: str,
                    input_type: str = 'search_document') -> Optional[list]:
        """Generate a Cohere embedding for the given text."""
        if not self.cohere_client:
            return None
        try:
            response = self.cohere_client.embed(
                texts=[text[:2048]],
                model='embed-english-v3.0',
                input_type=input_type,
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"AuthorityRetriever embed failed: {e}")
            return None

    def _chunk_text(self, text: str, max_tokens: int = 400) -> List[str]:
        """Split text into ~max_tokens chunks (approximate by characters)."""
        # Rough approximation: ~4 chars per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]

        chunks = []
        words = text.split()
        current = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= max_chars:
                chunks.append(' '.join(current))
                current = []
                current_len = 0

        if current:
            chunks.append(' '.join(current))

        return chunks

    def get_corpus_stats(self) -> dict:
        """Return stats about the authority corpus."""
        if not self.enabled:
            return {'enabled': False, 'count': 0}
        try:
            return {
                'enabled': True,
                'count': self.collection.count(),
                'collection_name': AUTHORITY_COLLECTION_NAME,
            }
        except Exception:
            return {'enabled': True, 'count': 0}
