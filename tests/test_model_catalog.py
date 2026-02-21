from __future__ import annotations

from poecoder.services.model_catalog import ModelCatalog


def test_model_catalog_refreshes_openai_models(monkeypatch) -> None:
    class Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def get(self, url: str, headers: dict[str, str]):
            del headers
            if url.endswith("/v1/models"):
                return Resp(
                    {
                        "data": [
                            {"id": "gpt-5.1"},
                            {"id": "o4-mini"},
                            {"id": "text-embedding-3-small"},
                        ]
                    }
                )
            return Resp({"data": []})

    import poecoder.services.model_catalog as module

    monkeypatch.setattr(module.httpx, "Client", Client)
    catalog = ModelCatalog(
        supported_models=["assistant"],
        api_key=None,
        openai_api_key="oa-key",
        openai_api_url="https://api.openai.com/v1",
    )
    models = catalog.list_models(refresh=True)
    assert "openai/gpt-5.1" in models
    assert "openai/o4-mini" in models
    assert "openai/text-embedding-3-small" not in models


def test_model_catalog_openai_update_changes_base_url() -> None:
    catalog = ModelCatalog(
        supported_models=["assistant"],
        api_key=None,
        openai_api_key=None,
        openai_api_url="https://api.openai.com/v1",
    )
    catalog.update_openai(api_key="oa-key", base_url="https://proxy.openai.local/v1/")
    assert catalog.openai_api_key == "oa-key"
    assert catalog.openai_api_url == "https://proxy.openai.local/v1"


def test_model_catalog_provider_status_has_key_info() -> None:
    catalog = ModelCatalog(
        supported_models=["assistant", "openai/gpt-5"],
        api_key="poe-key",
        openai_api_key="oa-key",
        openai_api_url="https://api.openai.com/v1",
    )
    status = catalog.provider_status()
    assert status["poe"]["api_key_configured"] is True
    assert status["openai"]["api_key_configured"] is True
    assert status["openai"]["model_count"] == 1
