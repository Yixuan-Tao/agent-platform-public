# Test values are synthetic fixtures for software behavior.
import pytest

from core.model_gateway import ModelGateway, GatewayConfigurationError


def test_gateway_accepts_configured_allowlist():
    gateway = ModelGateway("https://models.example.com/v1", "key", ["models.example.com"])
    assert gateway.base_url == "https://models.example.com/v1"


def test_gateway_rejects_untrusted_host():
    with pytest.raises(GatewayConfigurationError):
        ModelGateway("https://public.example/v1", "key", ["models.example.com"])


def test_gateway_creates_client_without_exposing_key():
    gateway = ModelGateway("https://models.example.com/v1", "secret", ["models.example.com"])
    client = gateway.create_chat_client(timeout=10, max_retries=1)
    assert str(client.base_url) == "https://models.example.com/v1/"


def test_gateway_builds_validated_endpoint_and_headers():
    gateway = ModelGateway("https://models.example.com/v1", "secret", ["models.example.com"])
    assert gateway.endpoint("/v1/embeddings") == "https://models.example.com/v1/embeddings"
    assert gateway.headers() == {"Authorization": "Bearer secret", "Content-Type": "application/json"}
