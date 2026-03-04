"""Emergency services locator tool (hackathon version).

For the hackathon we skip the Google Maps/Places API complexity and
instead provide an ADK FunctionTool that gives actionable guidance.
The Live API's built-in Google Search grounding will provide real-time
location data when the model uses this context.

To add real Google Maps/Places API later you would need:
1. Enable the Places API in GCP console
2. Create an API key restricted to Places API
3. Use the ``googlemaps`` Python package
"""

from __future__ import annotations


async def find_nearest_emergency_services(location_description: str) -> str:
    """Find the nearest hospital or emergency room based on the user's location.

    Use when the user asks about nearby medical facilities or when an injury
    requires professional medical attention.

    Args:
        location_description: The user's described location or area,
            e.g. "I'm in downtown Lahore" or "near Times Square, New York".

    Returns:
        Actionable guidance for reaching emergency services.
    """
    return (
        f"Based on the user's location ({location_description}): "
        "Recommend they call their local emergency number "
        "(911 in the US, 999 in the UK, 112 in Europe, 1122 in Pakistan). "
        "They can also search 'nearest emergency room' on Google Maps on "
        "their phone. If the situation is life-threatening, always prioritize "
        "calling emergency services first."
    )
