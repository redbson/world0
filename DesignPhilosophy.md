# Design Philosophy

## Project Positioning

**World 0** is a cognitive concept-world system designed for LLM Agents.

It draws inspiration from Ontology, but it is not a traditional Ontology project. Its primary purpose is not to define the world through rigid logic, nor to archive objective facts in a complete and static way. Its purpose is to provide a structured conceptual framework that helps an Agent understand, organize, and interpret a problem space.

World 0 should be understood as a **cognitive structure**, not as a memory store, not as a workflow engine, and not as a conventional knowledge graph.

---

## Core Philosophy

### 1. Concept First
World 0 takes **concepts** as its primary unit.

Its focus is not on storing as many facts as possible, but on identifying the core conceptual units through which a task, domain, or problem can be understood. Facts may attach to concepts, but facts are not the foundation of the system.

This means World 0 is fundamentally concerned with:

- what a concept is
- how concepts differ from one another
- how concepts form boundaries
- how concepts support understanding and reasoning

---

### 2. Relation First
The value of World 0 does not come from isolated concepts, but from the **relationships between concepts**.

Traditional ontology systems often emphasize classification hierarchies. World 0 does not reject hierarchy, but it does not treat hierarchy as the center of the system. A parent-child tree is only one possible way to connect meaning.

World 0 gives priority to a broader conceptual network, including relations such as:

- similarity
- contrast
- dependency
- support
- containment
- contextual relevance
- activation linkage

Its structure is therefore better described as a **relational concept network** than as a rigid taxonomy.

---

### 3. Context Sensitive, Not Globally Static
A concept does not have the same weight in every situation.

The meaning, relevance, and usefulness of a concept depend on the task, the active context, and the perspective from which the world is being interpreted. Because of this, World 0 should not be treated as a static world description. It is a dynamic semantic structure whose active region changes depending on use.

This implies that World 0 must support:

- contextual relevance
- selective concept activation
- local conceptual projection
- shifting relation weights under different tasks

---

### 4. Serve Cognition, Not Storage
World 0 is designed to support **cognition**, not archival storage.

Its role is not to preserve everything that has happened, nor to store all available information. That role belongs to memory systems, documents, databases, or event logs. World 0 instead provides the structural lens through which information becomes understandable.

In that sense:

- memory answers: what happened
- storage answers: what exists
- World 0 answers: how this should be understood

---

### 5. LLM-Native Structure
World 0 should be readable, writable, and extendable by LLMs.

Its structures must not be optimized only for formal knowledge engineering or database design. They must also be semantically legible to language models. This means its core objects should remain expressive enough for LLM interpretation, while still being structured enough for computation.

World 0 is therefore not just machine-readable. It should be **LLM-operable**.

---

### 6. Projection Over Exhaustiveness
The practical value of World 0 does not come from having a complete model of the whole world.

Its practical value comes from being able to project the most relevant conceptual region for a given task. An Agent does not need the total world model at once. It needs the right local world-view.

For this reason, World 0 should optimize for:

- useful local views
- high-value conceptual subsets
- task-oriented conceptual projection
- reduction of semantic noise

It should not optimize primarily for maximum coverage.

---

## Top-Level Concepts

### Concept
A **Concept** is the fundamental cognitive unit in World 0.

A concept may represent an object, process, role, state, principle, method, or abstract frame. It is not just a word or label. It is a semantic unit that can be linked, activated, referenced, and projected.

### Concept Card
A **Concept Card** is the raw record of a concept.

It acts as the initial carrier of a concept within the system. It is not the world model itself, but the material from which the world model is formed. It provides a stable unit that both humans and LLMs can read, edit, and extend.

### Relation
A **Relation** is the connective structure between concepts.

Relations give shape to the world of concepts. Without relations, concepts remain isolated. Relations are therefore not secondary metadata; they are part of the primary structure of cognition in World 0.

### Context
A **Context** determines which concepts matter now.

It shapes relevance, weights relations, and controls which part of the conceptual world becomes active. Context prevents World 0 from collapsing into a static graph.

### Activation
**Activation** is the mechanism by which concepts become foregrounded.

When an input enters the system, some concepts become active, some remain dormant, and some expand into neighboring conceptual regions. Activation is what makes World 0 operational rather than merely descriptive.

### Projection
A **Projection** is a local conceptual view generated from the larger structure.

It is the part of World 0 that an Agent actually uses for a task. A projection is not the whole world. It is the selected conceptual structure most relevant to the current objective.

### Perspective
A **Perspective** is a task- or role-conditioned view over the same conceptual world.

Different Agents, or different modes of reasoning, may require different emphasis even when operating over the same underlying structure. Perspective allows one world structure to support multiple cognitive views.

---

## Boundary Principles

### World 0 is not Memory
Memory stores historical content, experiences, and previous interactions.
World 0 organizes conceptual structure.

### World 0 is not a Traditional Knowledge Graph
A knowledge graph usually emphasizes facts and entities.
World 0 emphasizes concepts, relations, relevance, and cognitive projection.

### World 0 is not a Workflow Engine
Workflow systems manage action sequences.
World 0 supports understanding.

### World 0 is not an Encyclopedia
Its goal is not to contain everything.
Its goal is to support structured understanding where it matters.

---

## Design Principles Summary

World 0 is built on the following principles:

1. **Concept first, not fact first**
2. **Relation first, not taxonomy first**
3. **Context sensitive, not globally static**
4. **Built for cognition, not archival storage**
5. **Readable and operable by LLMs**
6. **Projection-oriented, not coverage-oriented**

---

## One-Sentence Definition

**World 0 is a cognitive concept-world structure composed of concepts, relations, context, activation, and projection, designed to provide a stable and computable understanding framework for LLM-driven Agents.**
