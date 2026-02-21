from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poecoder.config import Settings, get_settings
from poecoder.db import Database
from poecoder.policy import PolicyEngine
from poecoder.router import ModelRouter
from poecoder.services.audit_service import AuditService
from poecoder.services.command_service import CommandService
from poecoder.services.leader_service import LeaderService
from poecoder.services.memory_service import MemoryService
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.model_profile_service import ModelProfileService
from poecoder.services.provider_secret_service import ProviderSecretService
from poecoder.services.review_service import ReviewService
from poecoder.services.session_service import SessionService
from poecoder.services.shell_service import ShellService
from poecoder.services.subagent_service import SubagentService
from poecoder.services.task_service import TaskService
from poecoder.services.turn_service import TurnService
from poecoder.services.usage_service import UsageService
from poecoder.services.wiki_service import WikiService
from poecoder.tools.code_tools import CodeTools
from poecoder.tools.runtime import ToolRuntime
from poecoder.tools.web_tools import WebTools


@dataclass(slots=True)
class AppState:
    settings: Settings
    db: Database
    sessions: SessionService
    memories: MemoryService
    wiki: WikiService
    commands: CommandService
    subagents: SubagentService
    shell: ShellService
    audit: AuditService
    model_catalog: ModelCatalog
    model_profiles: ModelProfileService
    usage: UsageService
    reviews: ReviewService
    provider_secrets: ProviderSecretService
    tools: ToolRuntime
    turns: TurnService
    tasks: TaskService
    leader: LeaderService



def build_app_state(workspace_root: Path | None = None) -> AppState:
    settings = get_settings()
    db = Database(settings.db_path)

    policy = PolicyEngine()
    model_client = PoeModelClient(
        api_url=settings.poe_api_url,
        api_key=settings.poe_api_key,
        openai_api_url=settings.openai_api_url,
        openai_api_key=settings.openai_api_key,
        openai_models=settings.openai_models,
    )
    router = ModelRouter(settings.default_small_model, settings.default_large_model)
    model_catalog = ModelCatalog(settings.supported_models, api_key=settings.poe_api_key)
    model_profiles = ModelProfileService(db=db)
    model_profiles.ensure_seeded(model_catalog.list_models(refresh=False))

    sessions = SessionService(db=db, default_model=settings.default_large_model)
    memories = MemoryService(db=db)
    wiki = WikiService(db=db)
    commands = CommandService(db=db, policy=policy)
    audit = AuditService(db=db)
    shell = ShellService(policy=policy, sessions=sessions)
    usage = UsageService(api_key=settings.poe_api_key)
    reviews = ReviewService(
        sessions=sessions,
        model_client=model_client,
        model_catalog=model_catalog,
        default_model=settings.reviewer_model,
        default_thinking_level=settings.reviewer_thinking_level,
        default_thinking_budget=settings.reviewer_thinking_budget,
        command_catalog_provider=lambda: [],
    )
    provider_secrets = ProviderSecretService(path=settings.home_dir / "provider_secrets.enc.json")

    root = workspace_root or Path.cwd()
    code_tools = CodeTools(root=root.resolve())
    web_tools = WebTools(download_root=(root / ".poecoder_downloads"))
    subagents = SubagentService(db=db, model_client=model_client, session_service=sessions, model_catalog=model_catalog)

    tools = ToolRuntime(
        code_tools=code_tools,
        memory_service=memories,
        wiki_service=wiki,
        command_service=commands,
        subagent_service=subagents,
        shell_service=shell,
        audit_service=audit,
        sessions=sessions,
        model_catalog=model_catalog,
        web_tools=web_tools,
        usage_service=usage,
        settings=settings,
        model_client=model_client,
        review_service=reviews,
    )
    reviews.command_catalog_provider = tools.command_catalog
    turns = TurnService(
        sessions=sessions,
        memories=memories,
        model_client=model_client,
        router=router,
        tools=tools,
        model_catalog=model_catalog,
        model_profiles=model_profiles,
    )
    tasks = TaskService(db=db, turns=turns, subagents=subagents)
    leader = LeaderService(
        db=db,
        sessions=sessions,
        tasks=tasks,
        shell=shell,
        model_client=model_client,
        model_catalog=model_catalog,
    )
    tools.task_service = tasks
    tools.leader_service = leader
    return AppState(
        settings=settings,
        db=db,
        sessions=sessions,
        memories=memories,
        wiki=wiki,
        commands=commands,
        subagents=subagents,
        shell=shell,
        audit=audit,
        model_catalog=model_catalog,
        model_profiles=model_profiles,
        usage=usage,
        reviews=reviews,
        provider_secrets=provider_secrets,
        tools=tools,
        turns=turns,
        tasks=tasks,
        leader=leader,
    )
