"""RAG tool — Vertex AI Search (Discovery Engine) for first-aid protocols.

Standard tier only: supports snippets and basic search.
Does NOT use extractive answers/segments or summaries (Enterprise-only).

Project: medlens-489020
Engine:  medlens-search-app
Store:   medlens-first-aid-docs
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.cloud import discoveryengine_v1 as discoveryengine

from app.config import settings

logger = logging.getLogger(__name__)


# ======================================================================
#  Low-level RAGTool class
# ======================================================================


class RAGTool:
    """Wraps the Discovery Engine SearchServiceClient.

    For the global location no special endpoint is needed. For regional
    locations (us/eu) you would set::

        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
    """

    def __init__(
        self,
        project_id: str,
        engine_id: str,
        location: str = "global",
    ) -> None:
        self.client = discoveryengine.SearchServiceClient()

        # Use the ENGINE path (not datastore path) for search-app queries.
        self.serving_config = (
            f"projects/{project_id}"
            f"/locations/{location}"
            f"/collections/default_collection"
            f"/engines/{engine_id}"
            f"/servingConfigs/default_search"
        )

    # ------------------------------------------------------------------

    async def search(self, query: str, page_size: int = 5) -> list[dict[str, Any]]:
        """Search the first-aid protocol documents.

        Returns a list of result dicts sorted by confidence (descending).
        """
        request = discoveryengine.SearchRequest(
            serving_config=self.serving_config,
            query=query,
            page_size=page_size,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                # ONLY snippet_spec on Standard tier.
                # Do NOT use summary_spec or extractive_content_spec —
                # they require Enterprise edition.
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                ),
            ),
        )

        # Discovery Engine client is synchronous — wrap for async.
        response = await asyncio.to_thread(self._do_search, request)

        results: list[dict[str, Any]] = []
        for result in response.results:
            doc = result.document
            struct_data = (
                dict(doc.derived_struct_data) if doc.derived_struct_data else {}
            )

            # Collect snippets from multiple possible locations.
            snippets: list[str] = []
            if hasattr(result, "snippet") and result.snippet:
                snippets.append(result.snippet.snippet)

            if "snippets" in struct_data:
                for s in struct_data["snippets"]:
                    if isinstance(s, dict) and "snippet" in s:
                        snippets.append(s["snippet"])

            results.append({
                "source": struct_data.get(
                    "title", struct_data.get("link", "Unknown Source")
                ),
                "snippet": " ".join(snippets) if snippets else struct_data.get("snippet", ""),
                "url": struct_data.get("link", ""),
                "document_id": doc.id,
                "confidence": getattr(result, "relevance_score", 0.0) or 0.0,
            })

        # Best results first.
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _do_search(self, request: discoveryengine.SearchRequest):
        """Synchronous search call (runs in a thread via ``asyncio.to_thread``)."""
        return self.client.search(request=request)


# ======================================================================
#  Singleton instance
# ======================================================================

rag_tool = RAGTool(
    project_id=settings.google_cloud_project or "medlens-489020",
    engine_id=settings.vertex_search_app or "medlens-search-app",
    location="global",
)


# ======================================================================
#  ADK-compatible function tool
# ======================================================================


async def search_first_aid_protocols(query: str) -> str:
    """Search verified first-aid protocol documents for treatment guidance.

    Use this tool whenever you need to find the correct first-aid procedure
    for an injury or medical situation.  Returns the top 3 matching results
    from the MedLens verified document corpus (WHO, Red Cross, etc.).

    Args:
        query: Natural language search query, e.g. "how to treat a burn".

    Returns:
        Formatted text with source, content snippet, and confidence for
        each matching document.
    """
    try:
        results = await rag_tool.search(query)
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return f"Search failed: {e}"

    if not results:
        return "No matching protocols found in the verified document database."

    formatted: list[str] = []
    for r in results[:3]:
        formatted.append(
            f"Source: {r['source']}\n"
            f"Content: {r['snippet']}\n"
            f"Confidence: {r['confidence']:.2f}"
        )

    return "\n---\n".join(formatted)
