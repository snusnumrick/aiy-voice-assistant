# Memory System Implementation Plan

## Summary
This plan evolves the current memory system (SQLite facts + optional embeddings + tiered recall) into a more structured, robust memory architecture with better ranking, temporal consistency, pruning, and user controls. All changes are gated behind config flags for safe rollout.

## Scope Assumptions
- Keep current `$remember:` flow and nightly optimization as baseline.
- Add new features behind config flags and default them to off or conservative values.
- Avoid breaking existing `facts/memory.db` by using additive schema migrations.

## Phases

### Phase 0: Baseline + Measurement
**Goal**: Establish a repeatable evaluation harness and regression tests.

Tasks
1. Add `scripts/memory_eval.py` with a small suite of memory scenarios (multi-session recall, temporal conflict, stale preference).
2. Add `tests/test_memory_eval.py` with deterministic fixtures and expected recall outputs.
3. Add config toggle `memory_eval_mode` to avoid impacting production behavior.

Benefits
- Detect regressions as retrieval and memory logic evolves.
- Faster iteration on ranking strategies.

Tradeoffs
- Time to craft stable scenarios.
- Tests can become brittle if prompts change.

---

### Phase 1: Structured Memory Notes (Schema + Metadata)
**Goal**: Move from flat strings to structured memory entries.

Tasks
1. Extend `facts` schema with additional columns.
2. Add `memory_links` table for future relational connections.
3. Implement schema versioning and migration.

Schema proposal
- `facts` new columns:
  - `tags TEXT`
  - `keywords TEXT`
  - `source TEXT`
  - `confidence REAL`
  - `doc_time TEXT`
  - `event_time TEXT`
  - `last_accessed TEXT`
  - `access_count INTEGER`
  - `ttl_days INTEGER`
  - `priority INTEGER`
- `memory_links`:
  - `id INTEGER PRIMARY KEY`
  - `from_id INTEGER`
  - `to_id INTEGER`
  - `relation TEXT`
  - `strength REAL`

Benefits
- Enables richer ranking (recency, confidence, tags).
- Lays groundwork for versioning and memory evolution.

Tradeoffs
- More complex storage and retrieval.
- Requires careful migration and backfill.

---

### Phase 2: Memory Manager (Paging + Working Set)
**Goal**: Introduce a policy layer that manages what stays in the system prompt.

Tasks
1. Add `src/memory_manager.py`.
2. Move prompt injection logic out of `ConversationManager.get_system_prompt`.
3. Implement a configurable working memory budget.

Behavior
- Always inject a small core set.
- Auto-promote high-value or frequently accessed facts.
- Demote stale facts from core tier.

Benefits
- Reduces prompt bloat while keeping relevant context.
- Approximates paging-style memory management.

Tradeoffs
- Risk of hiding facts that would help in a specific session.
- Requires tuning and sensible defaults.

---

### Phase 3: Temporal Grounding + Conflict Resolution
**Goal**: Resolve stale and conflicting facts using time and versioning.

Tasks
1. Parse timestamps into `doc_time` and `event_time`.
2. Add conflict detection and use `superseded_by` for versioning.
3. Rank newer facts higher when conflicts exist.

Benefits
- Reduces incorrect recall when facts evolve.
- Improves temporal reasoning.

Tradeoffs
- Entity/slot extraction is imperfect.
- False positives can hide valid older facts.

---

### Phase 4: Adaptive Forgetting
**Goal**: Prune old or low-utility memories to reduce noise.

Tasks
1. Add decay scoring to ranking.
2. Add nightly pruning with safe thresholds.
3. Respect pinned or high-priority items.

Config proposal
- `memory_decay_half_life_days`
- `memory_prune_min_score`
- `memory_prune_keep_core`

Benefits
- Keeps memory lean and relevant.
- Improves retrieval precision over time.

Tradeoffs
- Risk of forgetting infrequently used but important facts.
- Requires user trust and transparency.

---

### Phase 5: Retrieval Improvements
**Goal**: Improve relevance beyond keyword + vector cosine.

Tasks
1. Add ranking features: recency, access frequency, tag overlap, tier weight.
2. Add optional MMR-style diversity.
3. Log top candidates and scores for tuning.

Scoring sketch
```
score = w_vec * cosine
      + w_kw * keyword
      + w_time * recency
      + w_access * log(access_count + 1)
      + w_tag * tag_match
      + w_priority * priority
```

Benefits
- Better relevance and less semantic drift.
- More robust in long histories.

Tradeoffs
- Increased complexity and tuning effort.
- Slightly higher latency in recall.

---

### Phase 6: User Controls + Transparency
**Goal**: Let users inspect, pin, and delete memories.

Tasks
1. Add tool `memory_admin` with commands: list, search, pin, delete.
2. Add audit logging for memory changes.

Benefits
- Improves user trust and controllability.
- Aligns with HCI findings.

Tradeoffs
- Exposes more internals to users.
- Additional surface area for mistakes.

## Rollout Order
1. Phase 0 (evaluation) + Phase 1 (schema).
2. Phase 2 (memory manager) + Phase 5 (ranking).
3. Phase 3 (temporal conflicts).
4. Phase 4 (forgetting).
5. Phase 6 (user controls).

## File-Level Changes
- New: `src/memory_manager.py`
- New: `src/memory_admin_tool.py`
- Update: `src/memory_store.py` (schema + CRUD + migration)
- Update: `src/memory_tool.py` (ranking)
- Update: `src/conversation_manager.py` (memory manager integration)
- New: `scripts/memory_eval.py`
- New: `tests/test_memory_eval.py`
- Update: `README.md` (new config keys, new tools)
- Update: `config.json` (new defaults)

## Milestones
1. Baseline tests passing with no behavior change (Phase 0).
2. Schema migration in place without data loss (Phase 1).
3. Memory manager gating core prompt facts (Phase 2).
4. Recall ranking improved and tuned (Phase 5).
5. Temporal conflict resolution enabled (Phase 3).
6. Forgetting enabled with safe defaults (Phase 4).
7. User controls enabled (Phase 6).

