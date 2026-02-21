from __future__ import annotations

from poecoder.i18n import Translator, is_supported_lang, normalize_lang


def test_normalize_lang_variants() -> None:
    assert normalize_lang("zh") == "zh-cn"
    assert normalize_lang("zh_CN") == "zh-cn"
    assert normalize_lang("en-US") == "en"
    assert normalize_lang("unknown") == "en"


def test_translator_zhcn_strings() -> None:
    tr = Translator("zh-cn")
    assert "命令" in tr.t("cli.help")
    assert "/leader" in tr.t("cli.help")
    assert tr.t("msg.current_balance", points=12) == "当前余额：12 点"
    assert tr.t("table.model") == "模型"
    assert tr.t("msg.current_model", model="assistant") == "当前模型=assistant"
    assert tr.t("msg.plan_mode_enabled").startswith("已切换到规划模式")
    assert tr.t("msg.review_header", model="assistant").startswith("审查模型=")


def test_is_supported_lang() -> None:
    assert is_supported_lang("zh-cn")
    assert is_supported_lang("en")
    assert not is_supported_lang("fr")
