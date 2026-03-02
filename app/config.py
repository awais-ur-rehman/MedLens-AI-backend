"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """MedLens AI service settings.

    Values are loaded from environment variables and/or a `.env` file
    located in the backend root directory.
    """

    # Google Cloud
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    # Gemini models
    gemini_model: str = "gemini-live-2.5-flash-native-audio"
    gemini_flash_model: str = "gemini-2.5-flash"

    # Vertex AI Search
    vertex_search_datastore: str = ""
    vertex_search_app: str = ""

    # Cloud Storage
    gcs_bucket: str = "medlens-sessions"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
