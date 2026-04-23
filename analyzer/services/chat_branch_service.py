"""Chat branch-tree computation.

Extracted verbatim from `analyzer/routes/chat.py` during the 2026-04-23
maintainability refactor. Pure function — no Flask, no globals.
"""
from analyzer.db import get_messages, get_active_leaf


def compute_branch_data(session_id: str):
    """
    Returns (active_path, fork_points) for a chat session.

    active_path  — list of message dicts in conversation order, root → leaf,
                   representing the currently active branch.
    fork_points  — list of dicts:
                   { parent_id, variants: [{id, content, leaf_id, is_active}],
                     active_variant_index }
                   One entry per position where the user has edited and branched.

    Falls back to linear order when no branching exists (legacy sessions).
    """
    rows = get_messages(session_id)   # all messages, ordered by id
    if not rows:
        return [], []

    msgs = {r['id']: dict(r) for r in rows}
    active_leaf_id = get_active_leaf(session_id)

    # Legacy / linear sessions: no parent_id set on any message
    has_tree = any(m.get('parent_id') for m in msgs.values())
    if not has_tree or not active_leaf_id:
        return [dict(r) for r in rows], []

    # Build children map  (parent_id → [child_id, ...])
    children = {}
    for mid, msg in msgs.items():
        pid = msg.get('parent_id')
        children.setdefault(pid, []).append(mid)

    # Walk from active_leaf up to root
    active_path, cur, seen = [], active_leaf_id, set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        msg = msgs.get(cur)
        if not msg:
            break
        active_path.append(msg)
        cur = msg.get('parent_id')
    active_path.reverse()
    active_ids = {m['id'] for m in active_path}

    # Find fork points: parents with 2+ children
    def _find_leaf(node_id, depth=0):
        if depth > 100:
            return node_id
        kids = children.get(node_id, [])
        return _find_leaf(max(kids), depth + 1) if kids else node_id

    fork_points = []
    for parent_id, kids in children.items():
        if len(kids) < 2:
            continue
        # parent_id=None means the fork is at the conversation root (first message edited)
        variants, active_idx = [], 0
        for i, kid_id in enumerate(sorted(kids)):
            kid = msgs[kid_id]
            is_active = kid_id in active_ids
            if is_active:
                active_idx = i
            variants.append({
                'id':       kid_id,
                'content':  kid['content'][:100],
                'leaf_id':  _find_leaf(kid_id),
                'is_active': is_active,
            })
        fork_points.append({
            'parent_id':            parent_id,
            'variants':             variants,
            'active_variant_index': active_idx,
        })

    return active_path, fork_points
