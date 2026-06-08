"""Concept identity operations: ``merge`` and ``split``.

These are the heaviest, lowest-frequency mutations on the concept
store.  Both touch the name index, the token index, the dirty set, the
relation graph (in the case of merge) and several derived fields on the
ConceptNode itself.  Keeping them out of ``_manager.py`` keeps the
storage core small and focused on the hot reinforcement path.

Functions take a ``ConceptManager`` as their first argument and reach
into its private state — they are part of the same trust domain as the
manager itself.  This file is **internal**: external callers should
keep using ``ConceptManager.merge`` / ``ConceptManager.split``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.concept import ConceptNode, Maturity

if TYPE_CHECKING:
    from world0.concepts._manager import ConceptManager
    from world0.core import RelationStore


_MATURITY_ORDER = [
    Maturity.FADING,
    Maturity.EMBRYONIC,
    Maturity.DEVELOPING,
    Maturity.ESTABLISHED,
    Maturity.CORE,
]


def merge_concepts(
    manager: ConceptManager,
    keeper_id: str,
    absorbed_id: str,
    relations: RelationStore | None = None,
) -> ConceptNode | None:
    """Merge ``absorbed`` into ``keeper``.

    The keeper retains its canonical name; the absorbed node's name and
    aliases become aliases on the keeper.  Activation counts,
    disconfirmation counts and reinforcement logs are summed.
    Relations touching the absorbed node are rewritten to the keeper
    via the supplied ``relations`` store.

    Returns the keeper node, or None if either id is unknown.
    """
    if keeper_id == absorbed_id:
        return manager._concepts.get(keeper_id)  # type: ignore[attr-defined]

    keeper = manager._concepts.get(keeper_id)  # type: ignore[attr-defined]
    absorbed = manager._concepts.get(absorbed_id)  # type: ignore[attr-defined]
    if not keeper or not absorbed:
        return None

    # Move name + aliases onto the keeper.  The absorbed node's entries
    # in the name index currently point to its own id and would block
    # ``add_alias`` from re-mapping them; clear them first so the alias
    # attachment succeeds.
    absorbed_names = [absorbed.name, *absorbed.aliases]
    for name_variant in absorbed.all_names():
        manager._name_index.remove_if_owned(name_variant, absorbed.id)  # type: ignore[attr-defined]
    for alias in absorbed_names:
        manager.add_alias(keeper.id, alias)

    _merge_descriptive(keeper, absorbed)
    _merge_evidence(keeper, absorbed)
    _merge_domain_profile(keeper, absorbed)
    _merge_source_refs(keeper, absorbed)
    _merge_token_refs(keeper, absorbed)

    if relations is not None:
        relations.migrate_concept(absorbed.id, keeper.id)

    manager._token_index.index_node(keeper)  # type: ignore[attr-defined]
    manager.mark_dirty(keeper.id)

    manager.remove(absorbed.id)
    return keeper


def split_concept(
    manager: ConceptManager,
    concept_id: str,
    new_name: str,
    *,
    aliases_to_move: list[str] | None = None,
    description: str = "",
    domain: str = "",
) -> ConceptNode | None:
    """Detach some aliases of ``concept_id`` into a brand-new node.

    Symmetric inverse of merge for cases where one concept card
    accumulated two distinct meanings.  Relations are *not* moved —
    direction is meaning-sensitive and cannot be inferred mechanically.

    Returns the newly created node, or None if the source id is unknown
    or ``new_name`` is already in use.
    """
    source = manager._concepts.get(concept_id)  # type: ignore[attr-defined]
    if source is None:
        return None
    if manager.resolve(new_name) is not None:
        return None

    moved: list[str] = []
    if aliases_to_move:
        remaining: list[str] = []
        moved_norms = {a.strip().lower() for a in aliases_to_move}
        for alias in source.aliases:
            if alias.strip().lower() in moved_norms:
                moved.append(alias)
                manager._name_index.remove_if_owned(alias, source.id)  # type: ignore[attr-defined]
            else:
                remaining.append(alias)
        source.aliases = remaining

    new_node = ConceptNode(
        name=new_name,
        aliases=moved,
        description=description or "",
        domain=domain or source.domain,
        origin=source.origin,
        confidence=min(0.3, source.confidence),
    )
    manager._concepts[new_node.id] = new_node  # type: ignore[attr-defined]
    manager._identity_index[new_node.ensure_identity_key()] = new_node.id  # type: ignore[attr-defined]
    manager._name_index.index_node(new_node)  # type: ignore[attr-defined]
    manager._token_index.index_node(new_node)  # type: ignore[attr-defined]
    manager._token_index.index_node(source)  # type: ignore[attr-defined]
    manager.mark_dirty(source.id)
    manager.mark_dirty(new_node.id)
    return new_node


# ── merge helpers ────────────────────────────────────────────────────


def _merge_descriptive(keeper: ConceptNode, absorbed: ConceptNode) -> None:
    if absorbed.description and not keeper.description:
        keeper.description = absorbed.description
    if absorbed.domain and not keeper.domain:
        keeper.domain = absorbed.domain
    for tag in absorbed.tags:
        if tag not in keeper.tags:
            keeper.tags.append(tag)


def _merge_evidence(keeper: ConceptNode, absorbed: ConceptNode) -> None:
    keeper.activation_count += absorbed.activation_count
    keeper.disconfirmation_count += absorbed.disconfirmation_count
    keeper.reinforcement_log.extend(absorbed.reinforcement_log)
    if absorbed.last_activated > keeper.last_activated:
        keeper.last_activated = absorbed.last_activated
    if absorbed.last_weakened and (
        keeper.last_weakened is None
        or absorbed.last_weakened > keeper.last_weakened
    ):
        keeper.last_weakened = absorbed.last_weakened
    # Confidence: take the higher — a duplicate-merge should never
    # weaken the kept concept.  Maturity: take the more advanced.
    keeper.confidence = max(keeper.confidence, absorbed.confidence)
    if _MATURITY_ORDER.index(absorbed.maturity) > _MATURITY_ORDER.index(
        keeper.maturity
    ):
        keeper.maturity = absorbed.maturity


def _merge_domain_profile(keeper: ConceptNode, absorbed: ConceptNode) -> None:
    for dom, strength in absorbed.domain_profile.items():
        current = keeper.domain_profile.get(dom, 0.0)
        keeper.domain_profile[dom] = min(1.0, current + strength)


def _merge_source_refs(keeper: ConceptNode, absorbed: ConceptNode) -> None:
    for ref in absorbed.source_refs:
        keeper.record_source_ref(
            source_id=ref.source_id,
            source=ref.source,
            task=ref.task,
            excerpt=ref.excerpt,
        )


def _merge_token_refs(keeper: ConceptNode, absorbed: ConceptNode) -> None:
    for ref in absorbed.token_refs:
        keeper.record_token_ref(
            token=ref.token,
            source_id=ref.source_id,
            source=ref.source,
            task=ref.task,
            excerpt=ref.excerpt,
            role=ref.role,
        )
    keeper.record_token_ref(
        token=absorbed.name,
        source=absorbed.origin,
        role="merged_name",
    )
    for alias in absorbed.aliases:
        keeper.record_token_ref(
            token=alias,
            source=absorbed.origin,
            role="merged_alias",
        )
