from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Callable, TextIO

from mcp.client.stdio import StdioServerParameters, stdio_client

from fast_agent.mcp.transport_tracking import ChannelEvent, EventType

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream
    from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


@asynccontextmanager
async def tracking_stdio_client(
    server_params: StdioServerParameters,
    *,
    channel_hook: ChannelHook | None = None,
    errlog: TextIO | None = None,
) -> AsyncGenerator[
    tuple[ObjectReceiveStream[SessionMessage | Exception], ObjectSendStream[SessionMessage]], None
]:
    """Context manager for stdio client with basic connection tracking."""

    def emit_channel_event(event_type: EventType, detail: str | None = None) -> None:
        if channel_hook is None:
            return
        try:
            channel_hook(
                ChannelEvent(
                    channel="stdio",
                    event_type=event_type,
                    detail=detail,
                )
            )
        except Exception:  # pragma: no cover - hook errors must not break transport
            logger.exception("Channel hook raised an exception")

    try:
        # Emit connection event
        emit_channel_event("connect")

        # Use the original stdio_client without stream interception
        if errlog is None:
            async with stdio_client(server_params) as (read_stream, write_stream):
                yield read_stream, write_stream
        else:
            # MCP's stdio_client uses `errlog` as the subprocess's OS-level stderr
            # (anyio.open_process(stderr=errlog)), so it needs a REAL file
            # descriptor. The `errlog` we get here is an in-memory LoggerTextIO
            # whose fileno() points at /dev/null — so the child's stderr (e.g. a
            # Python traceback for a failed-to-start server) was silently dropped
            # and never reached record_stdio_stderr / the UI.
            #
            # Bridge it with a real OS pipe: hand MCP the write end (a true fd) and
            # pump the read end into `errlog` on a daemon thread. The child's stderr
            # is then captured into the server's stderr buffer and surfaced in the
            # startup-error path (recent_stdio_stderr_lines()).
            read_fd, write_fd = os.pipe()
            write_file = os.fdopen(write_fd, "w", buffering=1, errors="replace")
            read_file = os.fdopen(read_fd, "r", errors="replace")

            def _pump_stderr() -> None:
                # Reads until EOF (all write ends closed: the child's stderr fd on
                # exit + our write_file, closed in finally below).
                try:
                    for line in read_file:
                        errlog.write(line)
                except Exception:  # pragma: no cover - pump must never break transport
                    logger.debug("stderr pump thread failed", exc_info=True)

            pump = threading.Thread(target=_pump_stderr, name="mcp-stderr-pump", daemon=True)
            pump.start()
            try:
                async with stdio_client(server_params, errlog=write_file) as (
                    read_stream,
                    write_stream,
                ):
                    yield read_stream, write_stream
            finally:
                # Close our write end so the child's exit yields EOF to the pump,
                # then drain it BEFORE the transport tears down — the startup-error
                # path reads recent_stdio_stderr_lines() right after this returns.
                try:
                    write_file.close()
                except Exception:  # pragma: no cover
                    pass
                pump.join(timeout=2.0)
                try:
                    read_file.close()
                except Exception:  # pragma: no cover
                    pass

    except Exception as exc:
        # Emit error event
        emit_channel_event("error", detail=str(exc))
        raise
    finally:
        # Emit disconnection event
        emit_channel_event("disconnect")
