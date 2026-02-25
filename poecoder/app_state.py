from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poecoder.backend.agent_runtime import AgentRuntime
from poecoder.backend.engine import BackendEngine
from poecoder.backend.provider_secrets import ProviderSecretStore
from poecoder.backend.shell_runtime import ShellRuntime
from poecoder.backend.store import AgentStore
from poecoder.config import Settings, get_settings
from poecoder.services.model_clients import PoeModelClient
from poecoder.tools.web_tools import WebTools


@dataclass(slots=True)
class AppState:
    settings: Settings
    store: AgentStore
    model_client: PoeModelClient
    shell: ShellRuntime
    web_tools: WebTools
    provider_secrets: ProviderSecretStore
    runtime: AgentRuntime
    engine: BackendEngine


def build_app_state(workspace_root: Path | None = None) -> AppState:
    settings = get_settings()
    root = (workspace_root or Path.cwd()).resolve()
    store = AgentStore(db_path=settings.db_path)
    model_client = PoeModelClient(
        api_url=settings.poe_api_url,
        api_key=settings.poe_api_key,
        openai_api_url=settings.openai_api_url,
        openai_api_key=settings.openai_api_key,
        openai_models=settings.openai_models,
    )
    shell = ShellRuntime(workspace_root=root)
    web_tools = WebTools(download_root=root / ".poecoder_downloads")
    provider_secrets = ProviderSecretStore(path=settings.home_dir / "provider_secrets.enc.json")
    runtime = AgentRuntime(
        store=store,
        model_client=model_client,
        shell=shell,
        default_model=settings.default_large_model,
    )
    engine = BackendEngine(store=store, runtime=runtime)
    engine.bootstrap_defaults()
    return AppState(
        settings=settings,
        store=store,
        model_client=model_client,
        shell=shell,
        web_tools=web_tools,
        provider_secrets=provider_secrets,
        runtime=runtime,
        engine=engine,
    )
