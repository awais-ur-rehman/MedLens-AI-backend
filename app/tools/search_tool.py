"""Google Search grounding metadata parser.

Google Search grounding in Gemini is NOT a separate API call — it is a
built-in tool enabled in the model config.  For the Live API we already
configure it in ``gemini_live_client.py`` as::

    tools=[types.Tool(google_search=types.GoogleSearch())]

The model autonomously decides when to search.  This module provides a
parser that extracts structured citations from the grounding metadata
returned in Gemini responses.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GoogleSearchGroundingParser:
    """Parses grounding metadata from Gemini responses that used Google Search."""

    @staticmethod
    def extract_citations(grounding_metadata: Any) -> list[dict[str, Any]]:
        """Extract citation sources from Gemini grounding metadata.

        Args:
            grounding_metadata: The ``grounding_metadata`` attribute from a
                Gemini response candidate.  Contains ``search_entry_point``,
                ``grounding_chunks``, and ``grounding_supports``.

        Returns:
            A list of citation dicts with ``source``, ``url``, ``section``,
            and ``confidence`` keys.
        """
        citations: list[dict[str, Any]] = []

        if not grounding_metadata:
            return citations

        # --- grounding_chunks → web citations ---
        if hasattr(grounding_metadata, "grounding_chunks"):
            for chunk in grounding_metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    citations.append({
                        "source": chunk.web.title or "Web Source",
                        "url": chunk.web.uri or "",
                        "section": "",
                        "confidence": 1.0,
                    })

        return citations

    @staticmethod
    def extract_search_suggestions(grounding_metadata: Any) -> list[str]:
        """Extract follow-up search suggestions if available.

        Args:
            grounding_metadata: The grounding metadata from a Gemini response.

        Returns:
            A list of suggested search queries.
        """
        suggestions: list[str] = []

        if not grounding_metadata:
            return suggestions

        if hasattr(grounding_metadata, "web_search_queries"):
            for query in grounding_metadata.web_search_queries:
                suggestions.append(str(query))

        return suggestions

    @staticmethod
    def has_grounding(grounding_metadata: Any) -> bool:
        """Check whether the response actually used Google Search grounding.

        Args:
            grounding_metadata: The grounding metadata from a Gemini response.

        Returns:
            ``True`` if grounding chunks are present.
        """
        if not grounding_metadata:
            return False
        if hasattr(grounding_metadata, "grounding_chunks"):
            return bool(grounding_metadata.grounding_chunks)
        return False
