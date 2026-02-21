from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_LANG = "en"

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "cli.title": "PoeCoder CLI",
        "cli.type_help": "(type /help)",
        "cli.prompt": "poecoder> ",
        "cli.help": """
Commands:
  /help                               Show this help
  /quit                               Exit CLI
  /system <text>                      Set system message
  /mode <coding|chat|planning>        Start new backend session in mode
  /plan                               Switch to planning mode + planning system message
  /lang <en|zh-cn>                    Switch CLI language
  /image <path|url>                   Attach one image for next model/subagent request
  /images                             Show pending images
  /clearimages                        Clear pending images
  /listmodels                         List supported models
  /changemodel <name|auto>            Change active main model
  /balance                            Fetch current Poe point balance
  /context <key> <json>               Store context key/value in backend session
  /memory <scope> <text>              Write memory entry (session|project|global)
  /wiki <topic> <text>                Add project wiki note
  /subagent <model> <perm> <prompt>   Start subagent
  /bgturn <prompt>                    Start background turn task
  /bgsubagent <model> <perm> <prompt> Start background subagent task
  /tasks                               List background tasks
  /task <task_id>                      Read task detail
  /readtaskoutput <task_id>            Read task output/result only
  /canceltask <task_id>                Cancel background task
  /shell <danger> <command>           Run shell command through policy engine
""".strip(),
        "msg.system_updated": "System message updated.",
        "msg.mode_set": "Mode set to {mode}",
        "msg.plan_mode_enabled": "Planning mode enabled with planning system message.",
        "msg.image_added": "Image queued. pending={count}",
        "msg.image_not_found": "Image file not found: {path}",
        "msg.images_empty": "No pending images.",
        "msg.images_cleared": "Pending images cleared.",
        "msg.models_header": "models ({count})",
        "msg.direct_model_set": "Direct model set to {model}",
        "msg.session_model_changed": "Session model changed to {model}",
        "msg.balance_backend_only": "Balance requires backend mode.",
        "msg.current_balance": "Current balance: {points} points",
        "msg.context_json_invalid": "Context value must be valid JSON",
        "msg.local_context_updated": "Local context updated.",
        "msg.context_stored": "Context stored.",
        "msg.memory_backend_only": "Memory API requires backend mode.",
        "msg.memory_stored": "Memory stored.",
        "msg.wiki_backend_only": "Wiki API requires backend mode.",
        "msg.wiki_updated": "Wiki updated.",
        "msg.subagent_backend_only": "Subagent API requires backend mode.",
        "msg.subagent_started": "Subagent started: {id}",
        "msg.task_backend_only": "Background tasks require backend mode.",
        "msg.task_started": "Task started: {id}",
        "msg.task_list_header": "tasks ({count})",
        "msg.no_tasks": "No tasks.",
        "msg.task_detail": "task {id}",
        "msg.task_output_header": "task output {id} state={state}",
        "msg.task_output_pending": "Task has no output yet.",
        "msg.models_empty": "No models available.",
        "msg.current_model": "current={model}",
        "msg.table_empty": "(empty)",
        "msg.shell_backend_only": "Shell API requires backend mode.",
        "msg.unknown_command": "Unknown command. Use /help",
        "msg.session_id": "session={session_id}",
        "table.no": "no",
        "table.current": "current",
        "table.model": "model",
        "table.id": "id",
        "table.type": "type",
        "table.state": "state",
        "table.updated": "updated",
        "table.summary": "summary",
        "table.field": "field",
        "table.value": "value",
        "table.created": "created",
        "table.payload": "payload",
        "table.result": "result",
        "table.error": "error",
        "table.image": "image",
        "stream.model": "model={model}",
        "stream.tool": "tool:{name}",
        "stream.assistant": "assistant> ",
        "stream.done": "done model={model} tools={tools}",
        "msg.lang_switched": "Language switched to {lang}",
        "msg.lang_invalid": "Unsupported language: {lang}",
        "status.routing": "routing",
        "status.tools": "tools",
        "status.responding": "responding",
        "lang.en": "English",
        "lang.zh-cn": "Simplified Chinese",
    },
    "zh-cn": {
        "cli.title": "PoeCoder CLI",
        "cli.type_help": "（输入 /help 查看帮助）",
        "cli.prompt": "poecoder> ",
        "cli.help": """
命令：
  /help                               显示帮助
  /quit                               退出 CLI
  /system <text>                      设置系统提示词
  /mode <coding|chat|planning>        以指定模式创建后端会话
  /plan                               切换到规划模式并启用规划系统提示词
  /lang <en|zh-cn>                    切换 CLI 语言
  /image <path|url>                   为下一次模型/子代理请求添加一张图片
  /images                             查看待发送图片
  /clearimages                        清空待发送图片
  /listmodels                         列出可用模型
  /changemodel <name|auto>            切换主模型
  /balance                            查询 Poe 当前点数余额
  /context <key> <json>               向后端会话保存上下文键值
  /memory <scope> <text>              写入记忆（session|project|global）
  /wiki <topic> <text>                添加项目 Wiki 记录
  /subagent <model> <perm> <prompt>   启动子代理
  /bgturn <prompt>                    启动后台轮次任务
  /bgsubagent <model> <perm> <prompt> 启动后台子代理任务
  /tasks                               列出后台任务
  /task <task_id>                      查看任务详情
  /readtaskoutput <task_id>            仅查看任务输出结果
  /canceltask <task_id>                取消后台任务
  /shell <danger> <command>           通过策略引擎执行 shell 命令
""".strip(),
        "msg.system_updated": "系统提示词已更新。",
        "msg.mode_set": "模式已切换为 {mode}",
        "msg.plan_mode_enabled": "已切换到规划模式，并启用规划系统提示词。",
        "msg.image_added": "图片已加入队列。待发送={count}",
        "msg.image_not_found": "未找到图片文件：{path}",
        "msg.images_empty": "没有待发送图片。",
        "msg.images_cleared": "待发送图片已清空。",
        "msg.models_header": "模型列表（{count}）",
        "msg.direct_model_set": "直连模式模型已设置为 {model}",
        "msg.session_model_changed": "会话模型已切换为 {model}",
        "msg.balance_backend_only": "余额查询仅支持后端模式。",
        "msg.current_balance": "当前余额：{points} 点",
        "msg.context_json_invalid": "上下文值必须是合法 JSON",
        "msg.local_context_updated": "本地上下文已更新。",
        "msg.context_stored": "上下文已保存。",
        "msg.memory_backend_only": "记忆功能仅支持后端模式。",
        "msg.memory_stored": "记忆已保存。",
        "msg.wiki_backend_only": "Wiki 功能仅支持后端模式。",
        "msg.wiki_updated": "Wiki 已更新。",
        "msg.subagent_backend_only": "子代理功能仅支持后端模式。",
        "msg.subagent_started": "子代理已启动：{id}",
        "msg.task_backend_only": "后台任务仅支持后端模式。",
        "msg.task_started": "任务已启动：{id}",
        "msg.task_list_header": "任务列表（{count}）",
        "msg.no_tasks": "暂无任务。",
        "msg.task_detail": "任务 {id}",
        "msg.task_output_header": "任务输出 {id} 状态={state}",
        "msg.task_output_pending": "任务尚未生成输出。",
        "msg.models_empty": "没有可用模型。",
        "msg.current_model": "当前模型={model}",
        "msg.table_empty": "（空）",
        "msg.shell_backend_only": "Shell 功能仅支持后端模式。",
        "msg.unknown_command": "未知命令，请使用 /help",
        "msg.session_id": "会话={session_id}",
        "table.no": "序号",
        "table.current": "当前",
        "table.model": "模型",
        "table.id": "ID",
        "table.type": "类型",
        "table.state": "状态",
        "table.updated": "更新时间",
        "table.summary": "摘要",
        "table.field": "字段",
        "table.value": "值",
        "table.created": "创建时间",
        "table.payload": "输入",
        "table.result": "结果",
        "table.error": "错误",
        "table.image": "图片",
        "stream.model": "模型={model}",
        "stream.tool": "工具:{name}",
        "stream.assistant": "助手> ",
        "stream.done": "完成 模型={model} 工具={tools}",
        "msg.lang_switched": "语言已切换为 {lang}",
        "msg.lang_invalid": "不支持的语言：{lang}",
        "status.routing": "路由中",
        "status.tools": "工具处理中",
        "status.responding": "生成中",
        "lang.en": "英文",
        "lang.zh-cn": "简体中文",
    },
}

_ALIAS = {
    "zh": "zh-cn",
    "zh_cn": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-hans": "zh-cn",
    "en": "en",
    "en-us": "en",
}


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return DEFAULT_LANG
    key = lang.strip().lower().replace("_", "-")
    return _ALIAS.get(key, DEFAULT_LANG)


@dataclass(slots=True)
class Translator:
    lang: str

    def __post_init__(self) -> None:
        self.lang = normalize_lang(self.lang)

    def set_lang(self, lang: str) -> str:
        self.lang = normalize_lang(lang)
        return self.lang

    def t(self, key: str, **kwargs: Any) -> str:
        base = _TRANSLATIONS.get(self.lang, _TRANSLATIONS[DEFAULT_LANG])
        text = base.get(key)
        if text is None:
            text = _TRANSLATIONS[DEFAULT_LANG].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text


def supported_langs() -> list[str]:
    return ["en", "zh-cn"]


def is_supported_lang(lang: str | None) -> bool:
    if not lang:
        return False
    key = lang.strip().lower().replace("_", "-")
    return key in _ALIAS
