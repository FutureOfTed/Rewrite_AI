import pytest
from app.services.backend_client import get_dataset_links

@pytest.mark.asyncio
async def test_get_dataset_links():
    links = await get_dataset_links("job_123")
    assert isinstance(links, list)
    # Mocking httpx responses would be added here
