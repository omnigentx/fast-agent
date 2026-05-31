# Changes from upstream

This repository is a fork of [evalstate/fast-agent](https://github.com/evalstate/fast-agent),
licensed under the Apache License 2.0. This file lists the categories of
modifications made downstream so redistributors can see at a glance what
diverges from upstream, satisfying Apache 2.0 §4(b)
("You must cause any modified files to carry prominent notices stating
that You changed the files").

For the exact set of changed files and commit-by-commit context, run:

```bash
git log --oneline upstream/main..origin/main
git diff upstream/main..origin/main --stat
```

## Modification categories

The omnigentx fork extends upstream fast-agent to power the
[omnigentx/jarvis](https://github.com/omnigentx/jarvis) project. The
substantive deltas group as follows:

1. **Spawn / team / agent orchestration.** Layered port of an agent
   spawning subsystem (registry, execution engine, TUI display, MCP
   server tools, identity guards, env propagation). Touches `src/`
   under spawn, team, meeting, agent_card, runtime_paths, mcp.

2. **Pause / resume state machine.** New `on_pause_cancel` hook,
   transitional `pausing` / `resuming` states, tool-runner Path B for
   `stop_reason=CANCELLED`, identity-asymmetric Path A/B docs.

3. **Email + meeting-room MCP servers.** New tool surfaces
   (`send_team_message`, `resume_team_tool`, meeting agenda / approval
   flow, inbox watcher, push-model communication).

4. **Security / identity hardening.** Strict caller-identity checks on
   the 5 team-tool surfaces, `TEAM_MY_NAME` bypass closed, refusal of
   impersonation in meeting_room, `JARVIS_RUNTIME_RPC_SOCKET`
   propagation to grandchild MCPs, oversized-tool-result capping.

5. **SQLite persistence.** TeamSession storage migrated to SQLite,
   spawn results persisted to SQLite instead of YAML.

6. **Dependency / CVE bumps.** fastmcp 3.1.1 → 3.2.0
   (CVE: SSRF + path traversal in OpenAPI provider), agent-client-protocol
   capped <0.9 to avoid removed AuthMethod.

7. **Internationalisation.** Replaced Vietnamese agent-name pool with
   gender-neutral ASCII placeholders to keep upstream samples neutral.

8. **CI + lint baseline.** ruff baseline reset, `ty` step set to
   continue-on-error pending separate fix; test isolation for cross-
   module logger pollution.

Each commit on this fork carries enough context in its subject + body to
locate the relevant files. Where individual source files were modified
beyond cosmetic CI fixes, the modification is recoverable via
`git blame` against `upstream/main`.

## Upstream sync policy

`origin/main` periodically merges `upstream/main` (see
`chore/sync-upstream-*` branches and their merge commits). The merges
preserve the changelog at the commit level; this file is the
human-readable summary expected by the Apache 2.0 §4(b) "prominent
notice" requirement.
