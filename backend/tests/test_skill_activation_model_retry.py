from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares import skill_activation_middleware as middleware_module
from deerflow.agents.middlewares.skill_activation_middleware import (
    SkillActivationMiddleware,
    is_slash_skill_activation_reminder,
)
from deerflow.skills.types import Skill, SkillCategory


_SLASH_SOURCE_OWNER_TOKEN = "test-slash-source-owner"


def _make_skill(tmp_path: Path) -> Skill:
    skill_dir = tmp_path / "data-analysis"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("# Data Analysis\nUse pandas.\n", encoding="utf-8")
    return Skill(
        name="data-analysis",
        description="Analyze data",
        license="MIT",
        skill_dir=skill_dir,
        skill_file=skill_file,
        relative_path=Path("data-analysis"),
        category=SkillCategory.CUSTOM,
        enabled=True,
    )


def _make_storage(tmp_path: Path, skill: Skill):
    return SimpleNamespace(
        load_skills=lambda *, enabled_only: [skill] if skill.enabled or not enabled_only else [],
        get_container_root=lambda: "/mnt/skills",
        get_skills_root_path=lambda: tmp_path,
        validate_skill_file_path=lambda skill_file: skill_file.resolve(),
    )


def _make_request(message: HumanMessage, context: dict) -> ModelRequest:
    runtime = SimpleNamespace(context=context)
    return ModelRequest(
        model=object(),
        messages=[message],
        state={"messages": [message]},
        runtime=runtime,
    )


def _contains_activation(messages) -> bool:
    return any(is_slash_skill_activation_reminder(message) for message in messages)


def test_skill_activation_is_reinjected_when_a_sync_model_attempt_fails(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path)
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _make_storage(tmp_path, skill),
    )
    middleware = SkillActivationMiddleware(slash_source_owner_token=_SLASH_SOURCE_OWNER_TOKEN)
    original = HumanMessage(content="/data-analysis inspect uploads/data.csv", id="msg-retry-sync")
    context: dict = {}
    request = _make_request(original, context)
    attempts: list[bool] = []

    def failing_handler(model_request: ModelRequest):
        attempts.append(_contains_activation(model_request.messages))
        raise RuntimeError("synthetic provider failure")

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        middleware.wrap_model_call(request, failing_handler)

    def succeeding_handler(model_request: ModelRequest):
        attempts.append(_contains_activation(model_request.messages))
        return AIMessage(content="ok")

    result = middleware.wrap_model_call(request, succeeding_handler)

    assert result.content == "ok"
    assert attempts == [True, True]


@pytest.mark.anyio
async def test_skill_activation_is_reinjected_when_an_async_model_attempt_fails(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path)
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _make_storage(tmp_path, skill),
    )
    middleware = SkillActivationMiddleware(slash_source_owner_token=_SLASH_SOURCE_OWNER_TOKEN)
    original = HumanMessage(content="/data-analysis inspect uploads/data.csv", id="msg-retry-async")
    context: dict = {}
    request = _make_request(original, context)
    attempts: list[bool] = []

    async def failing_handler(model_request: ModelRequest):
        attempts.append(_contains_activation(model_request.messages))
        raise RuntimeError("synthetic provider failure")

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await middleware.awrap_model_call(request, failing_handler)

    async def succeeding_handler(model_request: ModelRequest):
        attempts.append(_contains_activation(model_request.messages))
        return AIMessage(content="ok")

    result = await middleware.awrap_model_call(request, succeeding_handler)

    assert result.content == "ok"
    assert attempts == [True, True]
