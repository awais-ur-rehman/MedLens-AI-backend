"""Vertex AI Search tool — queries the MedLens medical knowledge base."""

import logging
from typing import Any

from google.cloud import discoveryengine_v1 as discoveryengine

from app.config import settings

logger = logging.getLogger(__name__)


def search_medical_knowledge(query: str) -> dict[str, Any]:
    """Search the MedLens first-aid knowledge base for relevant protocols.

    Uses Vertex AI Search (Discovery Engine) to query the medical document
    corpus.  Returns the top 3 results with snippets.

    Args:
        query: Natural language search query, e.g. "how to treat a burn".

    Returns:
        A dict with 'results' (list of dicts with title, snippet, link)
        and 'total_size'.
    """
    try:
        client = discoveryengine.SearchServiceClient()

        serving_config = (
            f"projects/{settings.google_cloud_project}"
            f"/locations/global"
            f"/collections/default_collection"
            f"/engines/{settings.vertex_search_app}"
            f"/servingConfigs/default_search"
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=3,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                ),
            ),
        )

        response = client.search(request)

        results = []
        for result in response.results:
            doc = result.document
            struct_data = doc.derived_struct_data

            title = ""
            snippet = ""
            link = ""

            if struct_data:
                title = struct_data.get("title", "")
                link = struct_data.get("link", "")
                snippets = struct_data.get("snippets", [])
                if snippets:
                    snippet = snippets[0].get("snippet", "")

            results.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "document_id": doc.id,
            })

        return {
            "results": results,
            "total_size": response.total_size,
            "query": query,
        }

    except Exception as e:
        logger.error("Vertex AI Search failed: %s", e)
        return {
            "results": [],
            "total_size": 0,
            "query": query,
            "error": str(e),
        }
