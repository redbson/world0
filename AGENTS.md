# AGENTS.md

## Project: World 0

This repository implements **World 0**, a cognitive concept-world system designed for LLM Agents.

World 0 is **not** a traditional Ontology project, **not** a memory system, and **not** a conventional knowledge graph. Its purpose is to provide a structured conceptual framework that allows an Agent to understand a problem space through concepts, relations, context, activation, and projection.

The system should be developed as a **concept-first, relation-first, context-sensitive cognitive structure**.

---

## Core Project Definition

World 0 is a cognitive layer built around the following principles:

- **Concept first**: concepts are the primary unit, not facts
- **Relation first**: relations are central, not taxonomy alone
- **Context sensitive**: relevance changes with task and perspective
- **Cognition-oriented**: the system supports understanding, not archival storage
- **LLM-operable**: structures should be readable, writable, and extendable by LLMs
- **Projection-oriented**: the system should generate useful local conceptual views instead of exposing the whole structure at once

World 0 should be treated as a system that helps an Agent answer:

- What concepts are active in the current task?
- How are these concepts related?
- What conceptual boundaries matter here?
- What local conceptual projection should be presented to an Agent?

---

## Development Philosophy

When developing World 0, always preserve the distinction between:

- **conceptual structure**
- **memory**
- **facts**
- **workflow**
- **tool execution**

Do not collapse these layers into one system.

World 0 is specifically responsible for conceptual organization and cognitive projection.

This project should avoid drifting into:

- a generic note-taking or card database
- an encyclopedia or documentation vault
- a rigid formal ontology with little operational value
- a pure embedding search layer without explicit conceptual structure
- a workflow engine disguised as a cognitive system

---

## Top-Level System Concepts

All implementation work should remain aligned with these top-level concepts:

### Concept
A semantic unit that can be linked, activated, referenced, and projected.

### Concept Card
The raw record form of a concept. This is the initial, editable unit from which the concept-world structure is formed.

### Relation
The connective structure between concepts. Relations are first-class objects in the system.

### Context
The active condition that determines which concepts and relations matter now.

### Activation
The mechanism that determines which concepts become foregrounded under a given input or task.

### Projection
A task-relevant local conceptual view generated from the broader concept-world.

### Perspective
A role- or task-conditioned way of viewing the same underlying conceptual structure.

---

## Product Goal

The product goal is **not** to build the largest possible concept graph.

The goal is to build a system that can:

1. represent concepts clearly
2. connect concepts meaningfully
3. activate relevant concepts under task context
4. generate a local conceptual projection for an Agent
5. evolve over time without collapsing into unstructured semantic sprawl

If a proposed feature improves storage volume but does not improve conceptual clarity, activation quality, or projection usefulness, it should be treated as lower priority.

---

## Engineering Priorities

Prioritize work in the following order:

### 1. Concept integrity
Ensure concepts are stable, distinguishable, and minimally coherent.

### 2. Relation quality
Ensure relations are meaningful, typed, and explainable.

### 3. Context handling
Ensure the system can adapt concept relevance based on task conditions.

### 4. Projection quality
Ensure the system can produce useful, compact conceptual views for downstream Agent use.

### 5. Evolvability
Ensure the system can grow incrementally without requiring complete redesign.

Do **not** prioritize breadth of coverage over structural usefulness.

---

## Recommended Initial Architecture

The project should begin with a minimal architecture, rather than a complete world model.

Suggested foundational modules:

- `concepts/`
  - concept definitions
  - concept normalization
  - alias handling
  - concept card management

- `relations/`
  - relation types
  - relation creation and validation
  - relation weighting
  - relation traversal rules

- `context/`
  - context model
  - contextual relevance rules
  - task-conditioned weighting

- `activation/`
  - concept activation logic
  - propagation rules
  - neighborhood expansion

- `projection/`
  - local conceptual view generation
  - ranking and filtering
  - projection shaping for different perspectives

- `perspectives/`
  - role-specific projection strategies
  - perspective profiles

- `schemas/`
  - concept card schema
  - relation schema
  - context schema
  - projection schema

- `tests/`
  - concept integrity tests
  - relation semantics tests
  - context and activation tests
  - projection behavior tests

---

## Minimum Viable System

The first usable version of World 0 should support:

1. creation of concept cards
2. typed relations between concepts
3. contextual weighting of concepts and relations
4. activation of a concept set from an input task
5. projection of a small local concept-world for downstream use

A good MVP is not one with many concepts.
A good MVP is one where a small number of concepts can produce a useful projection consistently.

---

## Design Rules for Codex

When making design or implementation decisions, follow these rules:

### Rule 1: Prefer explicit conceptual structure over hidden semantic behavior
If the system relies only on embeddings or latent similarity without explicit conceptual representation, it is moving away from the purpose of World 0.

### Rule 2: Keep cards simple, but semantically meaningful
Concept cards should remain compact enough to edit and reason over, while still containing enough structure to support relation-building and projection.

### Rule 3: Relations must be typed
Avoid reducing everything to a generic `related_to` unless it is explicitly a temporary fallback.

### Rule 4: Context changes relevance
The system should never assume that concept importance is globally fixed.

### Rule 5: Projection is the operational output
The system is only useful if it can turn a larger concept-world into a smaller task-relevant view.

### Rule 6: Preserve boundary clarity
Do not add memory responsibilities, workflow logic, or generalized document storage into the core model unless there is a very strong architectural reason.

### Rule 7: Build for iteration
Prefer designs that allow concept cards, relations, and projection logic to evolve gradually.

---

## Expected Repository Style

### Code style
- Prefer clear, typed, readable code
- Favor explicit data models over overly magical abstractions
- Keep conceptual operations easy to inspect and test
- Avoid unnecessary framework complexity in early stages

### Data modeling style
- Prefer structured schemas for core objects
- Keep semantics human-readable
- Make room for LLM-facing representations
- Avoid premature overformalization

### Testing style
- Test conceptual behavior, not only syntax
- Include tests for ambiguity, boundary cases, and projection stability
- Validate that context actually changes outputs in meaningful ways

---

## Early Non-Goals

The following are explicitly out of scope for early versions unless strongly justified:

- complete formal ontology reasoning
- universal world modeling
- exhaustive fact storage
- document archive features
- multi-agent orchestration engine
- workflow scheduling
- full memory system
- generic vector database replacement

These may become adjacent systems later, but they are not the initial purpose of World 0.

---

## What Good Progress Looks Like

Good progress means the repository increasingly supports the following chain:

1. define a concept
2. describe it with a concept card
3. connect it to other concepts through typed relations
4. evaluate relevance under a task context
5. activate a local concept neighborhood
6. project a useful task-level conceptual view

If development is not strengthening this chain, it is probably drifting.

---

## Guidance for Codex

When asked to generate code, architecture, schemas, or refactors for this project:

- preserve the cognitive-structure framing
- preserve the distinction between concept, relation, context, activation, and projection
- keep the system minimal and extensible
- prefer simple foundations over premature scale assumptions
- optimize for conceptual clarity and downstream Agent usability

When uncertain between a broad feature and a structurally coherent one, choose the structurally coherent option.

---

## One-Sentence Project Definition

**World 0 is a cognitive concept-world system for LLM Agents, built to organize concepts and relations into context-sensitive, projectable structures that support understanding rather than storage.**
