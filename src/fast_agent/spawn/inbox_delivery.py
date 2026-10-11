"""Preserve durable inbox envelopes across live delivery and process recovery."""

from __future__ import annotations

from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent

from fast_agent.spawn.message_bus import AgentMessage
from fast_agent.types import PromptMessageExtended


def needs_attention(message: AgentMessage) -> bool:
    """FYI copies remain queued until an actual task wakes the recipient."""
    return not (message.context.get("no_reply") or message.context.get("recipient_type") == "cc")


def format_inbox_text(messages: list[AgentMessage]) -> str:
    lines = [f"New inbox messages ({len(messages)}):"]
    for message in messages:
        context = message.context
        mode = "ACTION" if needs_attention(message) else "FYI — Do not reply or acknowledge"
        lines.append(
            f"[{mode}] from {message.from_name} to {message.to_name} "
            f"(id: {message.message_id}, type: {message.message_type}, "
            f"priority: {message.priority})"
        )
        if context.get("subject"):
            lines.append(f"Subject: {context['subject']}")
        if context.get("recipient_type") == "cc":
            lines.append(
                f"Primary recipients: {', '.join(context.get('primary_recipients', []))}. "
                "This recipient has an informational copy, not task ownership."
            )
        lines.append(message.content)
        if message.reply_to:
            lines.append(f"Reply to: {message.reply_to}")
    return "\n\n".join(lines)


def build_inbox_prompt(
    messages: list[AgentMessage], *, text: str | None = None
) -> PromptMessageExtended:
    parts = [
        TextContent(type="text", text=text if text is not None else format_inbox_text(messages))
    ]
    for message in messages:
        for attachment in message.context.get("attachments", []):
            mime = attachment["content_type"]
            if mime.startswith("image/"):
                parts.append(ImageContent(type="image", data=attachment["data_b64"], mimeType=mime))
            else:
                parts.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=f"attachment://{message.message_id}/{attachment['filename']}",
                            mimeType=mime,
                            blob=attachment["data_b64"],
                        ),
                    )
                )
    return PromptMessageExtended(role="user", content=parts)
