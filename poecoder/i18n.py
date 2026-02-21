from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_LANG = "en"

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "cli.title": "PoeCoder CLI",
        "cli.type_help": "(type /help)",
        "cli.prompt": "poecoder> ",
        "cli.api_key_prompt": "Poe API key: ",
        "cli.help": """
Commands:
  /help                               Show this help
  /quit                               Exit CLI
  /login [api_key]                    Set/update Poe API key (prompt if omitted)
  /system <text>                      Set system message
  /mode <coding|chat|planning|leader> Start new backend session in mode
  /plan                               Switch to planning mode + planning system message
  /thinking <quick|balanced|deep> [budget] Update thinking level and budget
  /commandpolicy <allow|deny> [encourage|noencourage] Set model command autonomy
  /lang <en|zh-cn>                    Switch CLI language
  /image <path|url>                   Attach one image for next model/subagent request
  /images                             Show pending images
  /clearimages                        Clear pending images
  /listmodels                         List supported models
  /modeltable                         Show model strategy table
  /changemodel <name|auto>            Change active main model
  /balance                            Fetch current Poe point balance
  /context <key> <json>               Store context key/value in backend session
  /memory <scope> <text>              Write memory entry (session|project|global)
  /wiki <topic> <text>                Add project wiki note
  /review <prompt>                    Run reviewer role analysis
  /reviewsettings [model level budget] Get or set reviewer defaults
  /subagent <model> <perm> <prompt>   Start subagent
  /bgturn <prompt>                    Start background turn task
  /bgsubagent <model> <perm> <prompt> Start background subagent task
  /leader <goal>                      Start leader-mode orchestration run
  /leaderstatus <run_id>              Read leader run status
  /leaderjobs <run_id>                List leader run jobs
  /leaderwait <run_id> [timeout]      Wait for leader run
  /leadercancel <run_id>              Cancel leader run
  /tasks                               List background tasks
  /task <task_id>                      Read task detail
  /readtaskoutput <task_id>            Read task output/result only
  /canceltask <task_id>                Cancel background task
  /shell <danger> <command>           Run shell command through policy engine
""".strip(),
        "msg.login_cancelled": "API key login cancelled.",
        "msg.login_direct_updated": "Direct Poe API key updated.",
        "msg.login_backend_updated": "Backend Poe API key updated.",
        "msg.system_updated": "System message updated.",
        "msg.mode_set": "Mode set to {mode}",
        "msg.invalid_mode": "Invalid mode: {mode}",
        "msg.thinking_updated": "Thinking updated: level={level} budget={budget}",
        "msg.invalid_number": "Invalid number",
        "msg.invalid_thinking_level": "Invalid thinking level: {level}",
        "msg.command_policy_updated": "Command policy updated: allow={allow} encourage={encourage}",
        "msg.command_policy_usage": "Usage: /commandpolicy <allow|deny> [encourage|noencourage]",
        "msg.plan_mode_enabled": "Planning mode enabled with planning system message.",
        "msg.image_added": "Image queued. pending={count}",
        "msg.image_not_found": "Image file not found: {path}",
        "msg.images_empty": "No pending images.",
        "msg.images_cleared": "Pending images cleared.",
        "msg.models_header": "models ({count})",
        "msg.model_table_empty": "No model profiles.",
        "msg.modeltable_backend_only": "Model table requires backend mode.",
        "msg.direct_model_set": "Direct model set to {model}",
        "msg.session_model_changed": "Session model changed to {model}",
        "msg.current_thinking": "thinking={level} budget={budget}",
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
        "msg.review_backend_only": "Review requires backend mode.",
        "msg.review_header": "review model={model}",
        "msg.review_settings_updated": "Reviewer settings updated.",
        "msg.review_settings_usage": "Usage: /reviewsettings OR /reviewsettings <model> <quick|balanced|deep> <budget>",
        "msg.task_backend_only": "Background tasks require backend mode.",
        "msg.task_started": "Task started: {id}",
        "msg.task_list_header": "tasks ({count})",
        "msg.no_tasks": "No tasks.",
        "msg.task_detail": "task {id}",
        "msg.task_output_header": "task output {id} state={state}",
        "msg.task_output_pending": "Task has no output yet.",
        "msg.leader_backend_only": "Leader mode commands require backend mode.",
        "msg.leader_started": "Leader run started: {id}",
        "msg.leader_run_header": "leader run {id}",
        "msg.leader_jobs_header": "leader jobs ({count})",
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
        "table.goal": "goal",
        "table.name": "name",
        "table.scope": "scope",
        "table.image": "image",
        "table.strategy": "strategy",
        "table.speed": "speed",
        "table.quality": "quality",
        "table.cost": "cost",
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
        "cli.api_key_prompt": "Poe API Key：",
        "cli.help": """
命令：
  /help                               显示帮助
  /quit                               退出 CLI
  /login [api_key]                    设置/更新 Poe API Key（省略则提示输入）
  /system <text>                      设置系统提示词
  /mode <coding|chat|planning|leader> 以指定模式创建后端会话
  /plan                               切换到规划模式并启用规划系统提示词
  /thinking <quick|balanced|deep> [budget] 更新思考等级与预算
  /commandpolicy <allow|deny> [encourage|noencourage] 设置模型命令自治策略
  /lang <en|zh-cn>                    切换 CLI 语言
  /image <path|url>                   为下一次模型/子代理请求添加一张图片
  /images                             查看待发送图片
  /clearimages                        清空待发送图片
  /listmodels                         列出可用模型
  /modeltable                         查看模型策略表
  /changemodel <name|auto>            切换主模型
  /balance                            查询 Poe 当前点数余额
  /context <key> <json>               向后端会话保存上下文键值
  /memory <scope> <text>              写入记忆（session|project|global）
  /wiki <topic> <text>                添加项目 Wiki 记录
  /review <prompt>                    启动审查员角色分析
  /reviewsettings [model level budget] 查看或设置审查默认配置
  /subagent <model> <perm> <prompt>   启动子代理
  /bgturn <prompt>                    启动后台轮次任务
  /bgsubagent <model> <perm> <prompt> 启动后台子代理任务
  /leader <goal>                      启动 leader 编排任务
  /leaderstatus <run_id>              查看 leader 任务状态
  /leaderjobs <run_id>                列出 leader 子任务
  /leaderwait <run_id> [timeout]      等待 leader 任务完成
  /leadercancel <run_id>              取消 leader 任务
  /tasks                               列出后台任务
  /task <task_id>                      查看任务详情
  /readtaskoutput <task_id>            仅查看任务输出结果
  /canceltask <task_id>                取消后台任务
  /shell <danger> <command>           通过策略引擎执行 shell 命令
""".strip(),
        "msg.login_cancelled": "已取消 API Key 登录。",
        "msg.login_direct_updated": "直连 Poe API Key 已更新。",
        "msg.login_backend_updated": "后端 Poe API Key 已更新。",
        "msg.system_updated": "系统提示词已更新。",
        "msg.mode_set": "模式已切换为 {mode}",
        "msg.invalid_mode": "无效模式：{mode}",
        "msg.thinking_updated": "思考设置已更新：level={level} budget={budget}",
        "msg.invalid_number": "数字格式无效",
        "msg.invalid_thinking_level": "无效的思考等级：{level}",
        "msg.command_policy_updated": "命令策略已更新：allow={allow} encourage={encourage}",
        "msg.command_policy_usage": "用法：/commandpolicy <allow|deny> [encourage|noencourage]",
        "msg.plan_mode_enabled": "已切换到规划模式，并启用规划系统提示词。",
        "msg.image_added": "图片已加入队列。待发送={count}",
        "msg.image_not_found": "未找到图片文件：{path}",
        "msg.images_empty": "没有待发送图片。",
        "msg.images_cleared": "待发送图片已清空。",
        "msg.models_header": "模型列表（{count}）",
        "msg.model_table_empty": "暂无模型画像。",
        "msg.modeltable_backend_only": "模型策略表仅支持后端模式。",
        "msg.direct_model_set": "直连模式模型已设置为 {model}",
        "msg.session_model_changed": "会话模型已切换为 {model}",
        "msg.current_thinking": "思考设置 level={level} budget={budget}",
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
        "msg.review_backend_only": "审查功能仅支持后端模式。",
        "msg.review_header": "审查模型={model}",
        "msg.review_settings_updated": "审查默认配置已更新。",
        "msg.review_settings_usage": "用法：/reviewsettings 或 /reviewsettings <model> <quick|balanced|deep> <budget>",
        "msg.task_backend_only": "后台任务仅支持后端模式。",
        "msg.task_started": "任务已启动：{id}",
        "msg.task_list_header": "任务列表（{count}）",
        "msg.no_tasks": "暂无任务。",
        "msg.task_detail": "任务 {id}",
        "msg.task_output_header": "任务输出 {id} 状态={state}",
        "msg.task_output_pending": "任务尚未生成输出。",
        "msg.leader_backend_only": "Leader 命令仅支持后端模式。",
        "msg.leader_started": "Leader 任务已启动：{id}",
        "msg.leader_run_header": "Leader 任务 {id}",
        "msg.leader_jobs_header": "Leader 子任务（{count}）",
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
        "table.goal": "目标",
        "table.name": "名称",
        "table.scope": "范围",
        "table.image": "图片",
        "table.strategy": "策略",
        "table.speed": "速度",
        "table.quality": "质量",
        "table.cost": "成本",
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
