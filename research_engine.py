"""
research_engine.py — Core LLM-powered research logic.

Calls the orchestrator LLM proxy to research a topic and returns a structured
dict with summary, key findings, sections, concepts, and metadata.
All settings are hot-reloadable via update_settings().
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL     = "claude-sonnet-4-6"
_DEFAULT_PROVIDER  = "anthropic"
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_DEPTH     = "standard"
_DEFAULT_LANGUAGE  = "English"

# Depth → (min_findings, max_findings, min_sections, max_sections)
_DEPTH_GUIDELINES: dict[str, tuple[int, int, int, int]] = {
    "brief":    (3,  5,  2, 3),
    "standard": (5,  8,  4, 5),
    "detailed": (8, 12,  6, 8),
}

_SYSTEM_PROMPT = """\
You are an expert research assistant with broad knowledge across all domains.
When given a topic to research, produce a thorough, accurate, and well-structured analysis.

Always respond with a single valid JSON object matching this exact schema — no markdown
fences, no preamble, no trailing text:

{
  "topic":          "the research topic exactly as understood",
  "summary":        "2–3 sentence executive summary of the topic",
  "key_findings":   ["concise finding 1", "concise finding 2", ...],
  "sections": [
    {"title": "Section Title", "content": "Detailed section content (2–4 paragraphs)"}
  ],
  "key_concepts":   ["term or concept", ...],
  "related_topics": ["related topic", ...],
  "confidence":     "high | medium | low",
  "caveats":        "important limitations, knowledge cutoffs, or uncertainties (empty string if none)"
}

Research depth guidelines (follow the requested depth):
  brief    → 3–5 key_findings, 2–3 sections
  standard → 5–8 key_findings, 4–5 sections
  detailed → 8–12 key_findings, 6–8 sections

Be factual, balanced, and cite specific details where possible.
"""


class ResearchEngine:
    """Thin async wrapper around the orchestrator LLM proxy for topic research."""

    def __init__(self, proxy_url: str, agent_id: str) -> None:
        self._proxy_url  = proxy_url
        self._agent_id   = agent_id
        self._model      = _DEFAULT_MODEL
        self._provider   = _DEFAULT_PROVIDER
        self._max_tokens = _DEFAULT_MAX_TOKENS
        self._depth      = _DEFAULT_DEPTH
        self._language   = _DEFAULT_LANGUAGE

    # ------------------------------------------------------------------ settings

    def update_settings(
        self,
        common_settings: dict,
        agent_settings: dict | None = None,
    ) -> None:
        """Apply settings pushed from the orchestrator dashboard (hot-reload)."""
        s = agent_settings or {}

        # Model: agent override → global default → compiled default
        for candidate in (
            str(s.get("research_model", "")).strip(),
            str(common_settings.get("default_model", "")).strip(),
            _DEFAULT_MODEL,
        ):
            if candidate:
                self._model = candidate
                break

        # Provider: agent override → global default → compiled default
        for candidate in (
            str(s.get("research_provider", "")).strip(),
            str(common_settings.get("default_provider", "")).strip(),
            _DEFAULT_PROVIDER,
        ):
            if candidate:
                self._provider = candidate
                break

        try:
            raw = s.get("research_max_tokens") or common_settings.get("research_max_tokens")
            if raw is not None:
                self._max_tokens = max(512, min(8192, int(raw)))
        except (ValueError, TypeError):
            pass

        depth = str(s.get("research_default_depth", "")).strip().lower()
        if depth in _DEPTH_GUIDELINES:
            self._depth = depth

        lang = str(s.get("research_language", "")).strip()
        if lang:
            self._language = lang

        logger.info(
            "ResearchEngine settings updated: model=%s provider=%s "
            "depth=%s max_tokens=%d language=%s",
            self._model, self._provider, self._depth, self._max_tokens, self._language,
        )

    # ------------------------------------------------------------------ research

    async def research(
        self,
        topic: str,
        depth: str | None = None,
        focus_areas: list[str] | None = None,
        output_sections: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Research *topic* and return a structured result dict.

        Args:
            topic:           The subject to research (required).
            depth:           'brief' | 'standard' | 'detailed' — overrides default.
            focus_areas:     Optional list of sub-topics to emphasise.
            output_sections: Optional list of section titles the response must include.
        """
        resolved_depth = (depth or self._depth).lower()
        if resolved_depth not in _DEPTH_GUIDELINES:
            resolved_depth = self._depth

        # ── Build user message ────────────────────────────────────────────────
        parts = [f"Research topic: {topic}", f"Depth: {resolved_depth}"]
        if self._language.lower() not in ("english", "en"):
            parts.append(f"Respond in: {self._language}")
        if focus_areas:
            parts.append(f"Focus areas: {', '.join(focus_areas)}")
        if output_sections:
            parts.append(f"Required section titles: {', '.join(output_sections)}")

        user_msg = "\n".join(parts)

        # ── Call LLM proxy ────────────────────────────────────────────────────
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json={
                    "provider":   self._provider,
                    "model":      self._model,
                    "system":     _SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": user_msg}],
                    "max_tokens": self._max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw_text = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"),
            "",
        )

        result = self._parse_response(raw_text, topic)
        result["research_depth"] = resolved_depth
        result["model_used"]     = self._model
        result["provider_used"]  = self._provider
        return result

    # ------------------------------------------------------------------ helpers

    def _parse_response(self, raw: str, topic: str) -> dict[str, Any]:
        """Parse LLM JSON; fall back to a minimal structure on failure."""
        text = raw.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed.setdefault("topic",          topic)
                parsed.setdefault("summary",         "")
                parsed.setdefault("key_findings",    [])
                parsed.setdefault("sections",        [])
                parsed.setdefault("key_concepts",    [])
                parsed.setdefault("related_topics",  [])
                parsed.setdefault("confidence",      "medium")
                parsed.setdefault("caveats",         "")
                return parsed
        except json.JSONDecodeError:
            pass

        logger.warning("ResearchEngine: LLM returned non-JSON; using fallback structure")
        return {
            "topic":          topic,
            "summary":        text[:500] if text else "Research completed.",
            "key_findings":   [],
            "sections":       [{"title": "Research Output", "content": text}],
            "key_concepts":   [],
            "related_topics": [],
            "confidence":     "low",
            "caveats":        "Response was not in the expected structured format.",
        }
