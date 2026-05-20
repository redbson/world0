# World 0 Agent TODO

Priority-ordered development backlog for the World 0 Agent.

This backlog is informed by `claw-code`'s strengths in statefulness, session control, recovery, and observability, but it stays inside World 0's boundary:

- World 0 is a concept-world system, not a generic workflow engine
- agent improvements should improve concept activation, projection quality, and cognitive usability
- orchestration features are only justified when they make conceptual work more reliable

## Current Progress

- Done: explicit `AgentState` and `SessionState` models
- Done: initial failure taxonomy and basic recovery policy for search/research flows
- Done: initial session control surface for rename, manual compact, and latest failure inspection
- Done: initial projection-quality feedback loop with feedback capture and lightweight concept/relation adjustment
- Next: deepen recovery policy beyond search, then expose projection feedback directly in the UI

## Priority Rules

Use these rules when choosing the next feature:

1. Prefer features that improve projection usefulness over features that only add storage volume.
2. Prefer explicit state and explainability over hidden agent behavior.
3. Prefer recovery and inspectability over brittle one-shot execution.
4. Prefer concept-quality and relation-quality improvements over generic orchestration features.

## P0: Current Highest Priority

These items should be built first.

### 1. Explicit Agent State Model

Add a first-class `AgentState` model instead of relying on booleans and ad hoc status fields.

Target outcomes:

- explicit states such as `ready`, `running`, `blocked`, `degraded`, `failed`, `recovering`
- a stable state transition model for the web UI, CLI, and future MCP exposure
- state reason fields that explain why the agent is blocked or degraded

Why this is P0:

- this is the clearest lesson from `claw-code`
- it makes the agent inspectable without turning World 0 into a logging system
- it improves trust in research, search, and session resume

### 2. Failure Taxonomy and Recovery Policy

Define structured failure classes and a recovery strategy for each one.

Target outcomes:

- classify failures such as `llm_error`, `provider_auth`, `provider_rate_limit`, `tool_runtime`, `mcp_unavailable`, `search_fetch_failed`, `session_corrupt`
- attach retryability and fallback rules to each class
- automatically retry or degrade where safe before surfacing the error

Why this is P0:

- current turn summaries are useful, but not enough
- World 0 needs reliable cognition support, not brittle tool execution

### 3. Session Control Surface

Turn sessions into explicit controllable units, not just resumable transcripts.

Target outcomes:

- inspect latest turn outcome
- inspect latest failure
- manually compact a session
- rename and annotate sessions
- show unresolved loops or open research threads in session metadata

Why this is P0:

- `claw-code` is strong on session control
- World 0 already has compaction and turn summaries, so this is the natural next step

### 4. Projection Quality Feedback

Add a way to evaluate whether the projected concept view was actually useful.

Target outcomes:

- capture whether a projection helped complete the task
- record missing concepts, noisy concepts, and weak relations
- feed that signal back into relation weighting and concept card refinement

Why this is P0:

- projection is World 0's operational output
- without feedback, the system can grow while projection quality stays flat

## P1: Next Layer

These items become high-value once P0 is stable.

### 5. Editable Concept Cards and Relation Authoring

Upgrade concept cards from inspection views into working cognitive objects.

Target outcomes:

- edit description, aliases, tags, and source notes from the card view
- create and delete typed relations from the card view
- show relation provenance and recent activation history

### 6. Research Backlog and Source Provenance

Turn one-off research runs into reusable conceptual work.

Target outcomes:

- persist research briefs, source lists, and unresolved questions
- distinguish source evidence from concept-world structure
- let users reopen a research thread and continue from its open questions

### 7. Provider Capability Matrix

Make the agent adapt behavior to provider and model capabilities instead of treating all models the same.

Target outcomes:

- track capability flags such as tool use, long context, structured output, vision, and search suitability
- adjust prompts, token parameters, and tool policy based on capabilities
- surface mismatches clearly in settings and status

### 8. MCP and Tool Health View

Expose tool-system health without collapsing the product into orchestration-first design.

Target outcomes:

- per-tool and per-MCP health summaries
- degraded-mode explanations
- last success and last failure snapshots

## P2: Later, If It Supports Cognition

These items are useful, but they are lower priority and should only ship if they clearly improve World 0's cognitive role.

### 9. Background Research Jobs

Support longer-running research tasks with explicit status and resumable outputs.

Constraint:

- do not turn this into a generic job scheduler

### 10. Coding Perspective Adapters

Add optional git, worktree, and test-awareness for coding-oriented perspectives.

Constraint:

- this should remain a perspective layer for concept activation and projection
- it should not become the core identity of World 0

### 11. MCP Server for World 0

Expose World 0 as an external tool surface for systems like Claude Code.

Target outcomes:

- concept and projection APIs exposed through MCP
- structured session and state inspection for external agents
- clear separation between concept-world operations and execution tooling

## Non-Goals for This Backlog

Even when inspired by `claw-code`, World 0 should not drift into these shapes:

- a generic multi-agent orchestration platform
- a workflow scheduler
- a notification router
- a document archive or note database
- a pure coding-agent shell without concept-world structure

## Definition of Good Progress

This backlog is working if it strengthens this chain:

1. the agent can inspect its own current state
2. the agent can recover or degrade safely when a failure happens
3. the system can activate the right concept neighborhood for a task
4. the projection is small, explainable, and useful
5. the user can understand why the agent reached that conceptual view
