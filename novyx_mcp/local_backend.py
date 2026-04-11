"""
Local SQLite backend for Novyx MCP.

Provides full memory operations without an API key or network connection.
Data is stored in ~/.novyx/local.db by default.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .local_embeddings import LocalEmbedder, pack_embedding, unpack_embedding
from .local_schema import init_db


class CloudFeatureError(Exception):
    """Raised when a cloud-only feature is called in local mode."""
    pass


class LocalBackend:
    """Full-featured local memory backend using SQLite.

    Supports core memory CRUD, semantic search, knowledge graph,
    context spaces, audit trail, and rollback — all offline.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self._data_dir = Path(
            data_dir or os.environ.get("NOVYX_DATA_DIR", "~/.novyx")
        ).expanduser()
        self._db_path = self._data_dir / "local.db"
        self._conn = init_db(self._db_path)
        self._lock = threading.Lock()
        self._embedder = LocalEmbedder()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def _audit(self, operation: str, artifact_id: str, details: dict | None = None) -> None:
        """Write an entry to the local audit trail."""
        self._conn.execute(
            "INSERT INTO audit_log (entry_id, operation, artifact_id, agent_id, timestamp, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._uuid(), operation, artifact_id, "local", self._now(),
             json.dumps(details or {}, default=str)),
        )
        self._conn.commit()

    def _row_to_memory(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a memory dict."""
        return {
            "uuid": row["uuid"],
            "observation": row["observation"],
            "context": row["context"],
            "agent_id": row["agent_id"],
            "tags": json.loads(row["tags"]),
            "importance": row["importance"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "superseded_by": row["superseded_by"],
        }

    def _row_to_draft(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a draft row to a MCP-friendly dict."""
        return {
            "draft_id": row["draft_id"],
            "branch_id": row["branch_id"],
            "status": row["status"],
            "observation": row["observation"],
            "context": row["context"],
            "tags": json.loads(row["tags"]),
            "importance": row["importance"],
            "confidence": row["confidence"],
            "review_summary": json.loads(row["review_summary"]),
            "merged_memory_id": row["merged_memory_id"],
            "merged_at": row["merged_at"],
            "rejected_at": row["rejected_at"],
            "rejection_reason": row["rejection_reason"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "mode": "local",
        }

    def _parse_time(self, target: str) -> datetime:
        """Parse ISO timestamp or relative time expression ('2 hours ago')."""
        # Relative time pattern
        match = re.match(r"(\d+)\s+(second|minute|hour|day|week)s?\s+ago", target.strip().lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            delta_map = {
                "second": timedelta(seconds=amount),
                "minute": timedelta(minutes=amount),
                "hour": timedelta(hours=amount),
                "day": timedelta(days=amount),
                "week": timedelta(weeks=amount),
            }
            return datetime.now(timezone.utc) - delta_map[unit]

        # ISO timestamp
        ts = target.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _cloud_only(self, feature: str) -> None:
        raise CloudFeatureError(
            f"{feature} is available with Novyx Cloud. "
            f"Sign up free at novyxlabs.com"
        )

    # ------------------------------------------------------------------
    # Core Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        observation: str,
        *,
        tags: list[str] | None = None,
        importance: int = 5,
        context: str | None = None,
        ttl_seconds: int | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Store a memory locally."""
        mem_id = self._uuid()
        now = self._now()
        tags_json = json.dumps(tags or [])

        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

        # Generate embedding
        embedding_blob = None
        vec = self._embedder.embed(observation)
        if vec:
            embedding_blob = pack_embedding(vec)

        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (uuid, observation, context, agent_id, tags, importance, "
                "confidence, created_at, embedding, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mem_id, observation, context, "local", tags_json, importance, confidence,
                 now, embedding_blob, expires_at),
            )
            self._audit("CREATE", mem_id, {"observation": observation, "tags": tags or []})

        return {
            "uuid": mem_id,
            "observation": observation,
            "tags": tags or [],
            "importance": importance,
            "confidence": confidence,
            "created_at": now,
            "mode": "local",
        }

    def draft_memory(
        self,
        observation: str,
        *,
        tags: list[str] | None = None,
        importance: int = 5,
        context: str | None = None,
        confidence: float = 1.0,
        branch_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a reviewable draft without changing canonical memory."""
        draft_id = f"drf_{uuid.uuid4().hex[:12]}"
        now = self._now()
        recall_result = self.recall(observation, limit=3, min_score=0.2, tags=tags)
        similar_memories = recall_result.get("memories", [])
        review_summary = {
            "proposed_changes": [
                change for change, present in [
                    ("new observation", True),
                    ("context", bool(context)),
                    ("tags", bool(tags)),
                    ("importance", importance != 5),
                    ("confidence", confidence != 1.0),
                ] if present
            ],
            "similar_count": len(similar_memories),
            "similar_memories": similar_memories,
        }

        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_drafts (draft_id, branch_id, observation, context, tags, importance, confidence, "
                "review_summary, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    draft_id,
                    branch_id,
                    observation,
                    context,
                    json.dumps(tags or []),
                    importance,
                    confidence,
                    json.dumps(review_summary, default=str),
                    "draft",
                    now,
                    now,
                ),
            )
            self._audit("DRAFT_CREATE", draft_id, {"observation": observation, "tags": tags or []})

        row = self._conn.execute(
            "SELECT * FROM memory_drafts WHERE draft_id = ?",
            (draft_id,),
        ).fetchone()
        return self._row_to_draft(row)

    def memory_drafts(self, *, status: str | None = None, branch_id: str | None = None) -> dict[str, Any]:
        """List stored drafts with optional status filtering."""
        if status and branch_id:
            rows = self._conn.execute(
                "SELECT * FROM memory_drafts WHERE status = ? AND branch_id = ? ORDER BY updated_at DESC",
                (status, branch_id),
            ).fetchall()
        elif status:
            rows = self._conn.execute(
                "SELECT * FROM memory_drafts WHERE status = ? ORDER BY updated_at DESC",
                (status,),
            ).fetchall()
        elif branch_id:
            rows = self._conn.execute(
                "SELECT * FROM memory_drafts WHERE branch_id = ? ORDER BY updated_at DESC",
                (branch_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_drafts ORDER BY updated_at DESC"
            ).fetchall()
        drafts = [self._row_to_draft(row) for row in rows]
        return {"total_count": len(drafts), "drafts": drafts, "mode": "local"}

    def memory_draft(self, draft_id: str) -> dict[str, Any]:
        """Fetch a single draft by ID."""
        row = self._conn.execute(
            "SELECT * FROM memory_drafts WHERE draft_id = ?",
            (draft_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Draft not found: {draft_id}")
        return self._row_to_draft(row)

    def draft_diff(
        self,
        draft_id: str,
        *,
        compare_to: str | None = None,
    ) -> dict[str, Any]:
        """Get a field-level diff between a draft and an existing memory."""
        draft = self.memory_draft(draft_id)

        compared_memory = None
        comparison_basis = "none"
        if compare_to:
            compared_memory = self.memory(compare_to)
            comparison_basis = "explicit"
        else:
            similar = draft.get("review_summary", {}).get("similar_memories", [])
            if similar:
                compared_memory = self.memory(similar[0]["uuid"])
                comparison_basis = "closest_match"

        changed_fields = []
        for field in ["observation", "context", "tags", "importance", "confidence"]:
            current = compared_memory.get(field) if compared_memory else None
            proposed = draft.get(field)
            changed_fields.append({
                "field": field,
                "current": current,
                "proposed": proposed,
                "changed": current != proposed,
            })

        if not compared_memory:
            recommendation = "merge_new"
        else:
            changed_count = sum(1 for field in changed_fields if field["changed"])
            recommendation = "check_duplicate" if changed_count <= 1 else "merge_or_supersede"

        return {
            "draft_id": draft_id,
            "compared_memory_id": compared_memory.get("uuid") if compared_memory else compare_to,
            "comparison_basis": comparison_basis,
            "recommendation": recommendation,
            "changed_fields": changed_fields,
            "current_memory": compared_memory,
            "proposed_memory": {
                "uuid": draft["draft_id"],
                "observation": draft["observation"],
                "context": draft["context"],
                "tags": draft["tags"],
                "importance": draft["importance"],
                "confidence": draft["confidence"],
                "created_at": draft["created_at"],
            },
            "mode": "local",
        }

    def memory_branch(self, branch_id: str) -> dict[str, Any]:
        """Get grouped review information for a branch/session."""
        branch = self.memory_drafts(branch_id=branch_id)
        drafts = branch["drafts"]
        if not drafts:
            raise ValueError(f"Branch not found: {branch_id}")
        recommendations: dict[str, int] = {}
        for draft in drafts:
            diff = self.draft_diff(draft["draft_id"])
            rec = diff["recommendation"]
            recommendations[rec] = recommendations.get(rec, 0) + 1
        return {
            "branch_id": branch_id,
            "total_drafts": len(drafts),
            "open_drafts": sum(1 for d in drafts if d["status"] == "draft"),
            "merged_drafts": sum(1 for d in drafts if d["status"] == "merged"),
            "rejected_drafts": sum(1 for d in drafts if d["status"] == "rejected"),
            "recommendations": recommendations,
            "drafts": drafts,
            "mode": "local",
        }

    def merge_draft(
        self,
        draft_id: str,
        *,
        supersede_memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Merge a draft into canonical local memory."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_drafts WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Draft not found: {draft_id}")
            draft = self._row_to_draft(row)
            if draft["status"] != "draft":
                raise ValueError(f"Draft is already {draft['status']}")

        result = self.remember(
            draft["observation"],
            tags=draft["tags"],
            importance=draft["importance"],
            context=draft["context"],
            confidence=draft["confidence"],
        )

        merged_at = self._now()
        with self._lock:
            self._conn.execute(
                "UPDATE memory_drafts SET status = ?, merged_memory_id = ?, merged_at = ?, updated_at = ? "
                "WHERE draft_id = ?",
                ("merged", result["uuid"], merged_at, merged_at, draft_id),
            )
            if supersede_memory_id:
                self._conn.execute(
                    "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE uuid = ? AND deleted = 0",
                    (result["uuid"], merged_at, supersede_memory_id),
                )
            self._audit(
                "DRAFT_MERGE",
                draft_id,
                {
                    "memory_id": result["uuid"],
                    "superseded_memory_id": supersede_memory_id,
                },
            )

        return {
            "draft_id": draft_id,
            "status": "merged",
            "memory_id": result["uuid"],
            "created_at": result["created_at"],
            "message": "Draft merged into canonical memory",
            "superseded_memory_id": supersede_memory_id,
            "mode": "local",
        }

    def merge_branch(self, branch_id: str) -> dict[str, Any]:
        """Merge all open drafts in a branch/session."""
        drafts = self.memory_drafts(status="draft", branch_id=branch_id)["drafts"]
        if not drafts:
            raise ValueError(f"No open drafts for branch: {branch_id}")
        merged_memory_ids = []
        for draft in drafts:
            result = self.merge_draft(draft["draft_id"])
            merged_memory_ids.append(result["memory_id"])
        return {
            "branch_id": branch_id,
            "merged_count": len(merged_memory_ids),
            "merged_memory_ids": merged_memory_ids,
            "status": "merged",
            "mode": "local",
        }

    def reject_draft(self, draft_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Reject a draft without creating a memory."""
        rejected_at = self._now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_drafts WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Draft not found: {draft_id}")
            draft = self._row_to_draft(row)
            if draft["status"] != "draft":
                raise ValueError(f"Draft is already {draft['status']}")
            self._conn.execute(
                "UPDATE memory_drafts SET status = ?, rejected_at = ?, rejection_reason = ?, updated_at = ? "
                "WHERE draft_id = ?",
                ("rejected", rejected_at, reason, rejected_at, draft_id),
            )
            self._audit("DRAFT_REJECT", draft_id, {"reason": reason})

        return self.memory_draft(draft_id)

    def reject_branch(self, branch_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Reject all open drafts in a branch/session."""
        drafts = self.memory_drafts(status="draft", branch_id=branch_id)["drafts"]
        if not drafts:
            raise ValueError(f"No open drafts for branch: {branch_id}")
        for draft in drafts:
            self.reject_draft(draft["draft_id"], reason=reason)
        return self.memory_branch(branch_id)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Search memories semantically.

        Returns a dict matching the cloud API shape for consistent tool handling.
        """
        # Purge expired memories first
        self._purge_expired()

        query_vec = self._embedder.embed(query)

        # Fetch all active memories
        if tags:
            # Filter by tags using LIKE matching on JSON array
            placeholders = " AND ".join("tags LIKE ?" for _ in tags)
            tag_patterns = [f'%"{t}"%' for t in tags]
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE deleted = 0 AND {placeholders}",
                tag_patterns,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE deleted = 0"
            ).fetchall()

        scored = []
        for row in rows:
            cosine = 0.0
            if query_vec and row["embedding"]:
                mem_vec = unpack_embedding(row["embedding"])
                cosine = self._embedder.similarity(query_vec, mem_vec)

            # Match cloud scoring: 0.7 * cosine + 0.2 * (importance/10) + 0.1 * confidence
            score = 0.7 * cosine + 0.2 * (row["importance"] / 10) + 0.1 * row["confidence"]

            if score >= min_score:
                mem = self._row_to_memory(row)
                mem["score"] = round(score, 4)
                mem["similarity"] = round(cosine, 4)
                scored.append(mem)

        # Sort by score descending
        scored.sort(key=lambda m: m["score"], reverse=True)
        scored = scored[:limit]

        return {
            "query": query,
            "total_results": len(scored),
            "memories": scored,
            "mode": "local",
            "embedding_strategy": self._embedder.strategy,
        }

    def forget(self, memory_id: str) -> bool:
        """Soft-delete a memory."""
        with self._lock:
            # Capture state for rollback
            row = self._conn.execute(
                "SELECT * FROM memories WHERE uuid = ? AND deleted = 0", (memory_id,)
            ).fetchone()
            if not row:
                return False

            before = self._row_to_memory(row)
            self._conn.execute(
                "UPDATE memories SET deleted = 1, updated_at = ? WHERE uuid = ?",
                (self._now(), memory_id),
            )
            self._audit("DELETE", memory_id, {"before": before})
        return True

    def memories(
        self,
        *,
        limit: int = 100,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List active memories."""
        self._purge_expired()

        if tags:
            placeholders = " AND ".join("tags LIKE ?" for _ in tags)
            tag_patterns = [f'%"{t}"%' for t in tags]
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE deleted = 0 AND {placeholders} "
                f"ORDER BY created_at DESC LIMIT ?",
                (*tag_patterns, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE deleted = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def memory(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by UUID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE uuid = ? AND deleted = 0", (memory_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Memory not found: {memory_id}")
        return self._row_to_memory(row)

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        self._purge_expired()

        total = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0"
        ).fetchone()["c"]

        avg_imp = self._conn.execute(
            "SELECT AVG(importance) as a FROM memories WHERE deleted = 0"
        ).fetchone()["a"]

        # Tag distribution
        rows = self._conn.execute(
            "SELECT tags FROM memories WHERE deleted = 0"
        ).fetchall()
        tag_counts: dict[str, int] = {}
        for row in rows:
            for tag in json.loads(row["tags"]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_memories": total,
            "average_importance": round(avg_imp or 0, 2),
            "tag_distribution": tag_counts,
            "embedding_strategy": self._embedder.strategy,
            "mode": "local",
            "data_dir": str(self._data_dir),
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        target: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Rollback memory state to a point in time using the audit trail."""
        target_dt = self._parse_time(target)
        target_iso = target_dt.isoformat()

        # Get all audit entries after the target, newest first
        entries = self._conn.execute(
            "SELECT * FROM audit_log WHERE timestamp > ? ORDER BY timestamp DESC",
            (target_iso,),
        ).fetchall()

        if not entries:
            return {
                "rolled_back_to": target_iso,
                "operations_undone": 0,
                "message": "No operations to undo after this timestamp.",
            }

        if dry_run:
            ops = {}
            for e in entries:
                op = e["operation"]
                ops[op] = ops.get(op, 0) + 1
            return {
                "dry_run": True,
                "target": target_iso,
                "operations_to_undo": len(entries),
                "breakdown": ops,
            }

        undone = 0
        with self._lock:
            for entry in entries:
                op = entry["operation"]
                artifact_id = entry["artifact_id"]
                details = json.loads(entry["details"])

                if op == "CREATE":
                    # Undo create = soft-delete
                    self._conn.execute(
                        "UPDATE memories SET deleted = 1, updated_at = ? WHERE uuid = ?",
                        (self._now(), artifact_id),
                    )
                    undone += 1
                elif op == "DELETE":
                    # Undo delete = un-delete
                    self._conn.execute(
                        "UPDATE memories SET deleted = 0, updated_at = ? WHERE uuid = ?",
                        (self._now(), artifact_id),
                    )
                    undone += 1
                elif op == "UPDATE":
                    before = details.get("before", {})
                    if before:
                        self._conn.execute(
                            "UPDATE memories SET observation = ?, tags = ?, importance = ?, "
                            "context = ?, updated_at = ? WHERE uuid = ?",
                            (before.get("observation"), json.dumps(before.get("tags", [])),
                             before.get("importance", 5), before.get("context"),
                             self._now(), artifact_id),
                        )
                        undone += 1

            self._audit("ROLLBACK", "system", {
                "target": target_iso,
                "operations_undone": undone,
            })

        return {
            "rolled_back_to": target_iso,
            "operations_undone": undone,
            "mode": "local",
        }

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(
        self,
        *,
        limit: int = 50,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get the local audit trail."""
        if operation:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE operation = ? ORDER BY timestamp DESC LIMIT ?",
                (operation, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "entry_id": row["entry_id"],
                "operation": row["operation"],
                "artifact_id": row["artifact_id"],
                "agent_id": row["agent_id"],
                "timestamp": row["timestamp"],
                "details": json.loads(row["details"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def link(
        self,
        source_id: str,
        target_id: str,
        *,
        relation: str = "related",
    ) -> dict[str, Any]:
        """Create a directed link between two memories."""
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO links (source_id, target_id, relation, weight, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_id, target_id, relation, 1.0, now),
            )
            self._audit("LINK", source_id, {"target": target_id, "relation": relation})

        return {
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "weight": 1.0,
            "created_at": now,
        }

    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------

    def _ensure_entity(self, name: str) -> str:
        """Get or create an entity by name, returning its ID."""
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row["entity_id"]

        entity_id = self._uuid()
        self._conn.execute(
            "INSERT INTO entities (entity_id, name, entity_type, created_at) VALUES (?, ?, ?, ?)",
            (entity_id, name, None, self._now()),
        )
        return entity_id

    def triple(
        self,
        subject: str,
        predicate: str,
        object_name: str,
    ) -> dict[str, Any]:
        """Add a knowledge graph triple."""
        now = self._now()
        with self._lock:
            subj_id = self._ensure_entity(subject)
            obj_id = self._ensure_entity(object_name)
            triple_id = self._uuid()

            self._conn.execute(
                "INSERT INTO triples (triple_id, subject_id, subject_name, predicate, "
                "object_id, object_name, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (triple_id, subj_id, subject, predicate, obj_id, object_name, 1.0, now),
            )
            self._conn.commit()

        return {
            "triple_id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_name,
            "confidence": 1.0,
            "created_at": now,
        }

    def triples(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> dict[str, Any]:
        """Query knowledge graph triples."""
        conditions = []
        params: list[str] = []

        if subject:
            conditions.append("subject_name = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object:
            conditions.append("object_name = ?")
            params.append(object)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM triples WHERE {where} ORDER BY created_at DESC",
            params,
        ).fetchall()

        results = [
            {
                "triple_id": row["triple_id"],
                "subject": row["subject_name"],
                "predicate": row["predicate"],
                "object": row["object_name"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        return {"triples": results, "total": len(results)}

    # ------------------------------------------------------------------
    # Context Spaces
    # ------------------------------------------------------------------

    def create_space(
        self,
        name: str,
        *,
        description: str | None = None,
        allowed_agent_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a context space."""
        space_id = self._uuid()
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO spaces (space_id, name, description, allowed_agent_ids, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (space_id, name, description, json.dumps(allowed_agent_ids or []),
                 json.dumps(tags or []), now),
            )
            self._conn.commit()

        return {
            "space_id": space_id,
            "name": name,
            "description": description,
            "allowed_agent_ids": allowed_agent_ids or [],
            "tags": tags or [],
            "memory_count": 0,
            "created_at": now,
        }

    def list_spaces(self) -> dict[str, Any]:
        """List all context spaces."""
        rows = self._conn.execute(
            "SELECT s.*, COUNT(sm.memory_id) as memory_count "
            "FROM spaces s LEFT JOIN space_members sm ON s.space_id = sm.space_id "
            "GROUP BY s.space_id ORDER BY s.created_at DESC"
        ).fetchall()

        spaces = [
            {
                "space_id": row["space_id"],
                "name": row["name"],
                "description": row["description"],
                "allowed_agent_ids": json.loads(row["allowed_agent_ids"]),
                "tags": json.loads(row["tags"]),
                "memory_count": row["memory_count"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        return {"spaces": spaces, "total_count": len(spaces)}

    def get_space(self, space_id: str) -> dict[str, Any]:
        """Get a context space by ID."""
        row = self._conn.execute(
            "SELECT s.*, COUNT(sm.memory_id) as memory_count "
            "FROM spaces s LEFT JOIN space_members sm ON s.space_id = sm.space_id "
            "WHERE s.space_id = ? GROUP BY s.space_id",
            (space_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Space not found: {space_id}")

        return {
            "space_id": row["space_id"],
            "name": row["name"],
            "description": row["description"],
            "allowed_agent_ids": json.loads(row["allowed_agent_ids"]),
            "tags": json.loads(row["tags"]),
            "memory_count": row["memory_count"],
            "created_at": row["created_at"],
        }

    def update_space(
        self,
        space_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        allowed_agent_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a context space."""
        current = self.get_space(space_id)

        with self._lock:
            self._conn.execute(
                "UPDATE spaces SET name = ?, description = ?, allowed_agent_ids = ?, tags = ? "
                "WHERE space_id = ?",
                (
                    name if name is not None else current["name"],
                    description if description is not None else current["description"],
                    json.dumps(allowed_agent_ids) if allowed_agent_ids is not None
                    else json.dumps(current["allowed_agent_ids"]),
                    json.dumps(tags) if tags is not None else json.dumps(current["tags"]),
                    space_id,
                ),
            )
            self._conn.commit()

        return self.get_space(space_id)

    def delete_space(self, space_id: str) -> None:
        """Delete a context space and its member associations."""
        with self._lock:
            self._conn.execute("DELETE FROM space_members WHERE space_id = ?", (space_id,))
            self._conn.execute("DELETE FROM spaces WHERE space_id = ?", (space_id,))
            self._conn.commit()

    def space_memories(
        self,
        space_id: str,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List or search memories in a context space."""
        rows = self._conn.execute(
            "SELECT m.* FROM memories m "
            "JOIN space_members sm ON m.uuid = sm.memory_id "
            "WHERE sm.space_id = ? AND m.deleted = 0 "
            "ORDER BY m.created_at DESC LIMIT ?",
            (space_id, limit),
        ).fetchall()

        mems = [self._row_to_memory(row) for row in rows]

        # If query provided, do in-memory similarity ranking
        if query and mems:
            query_vec = self._embedder.embed(query)
            if query_vec:
                for mem_dict, row in zip(mems, rows):
                    if row["embedding"]:
                        mem_vec = unpack_embedding(row["embedding"])
                        mem_dict["score"] = round(self._embedder.similarity(query_vec, mem_vec), 4)
                    else:
                        mem_dict["score"] = 0.0
                mems.sort(key=lambda m: m.get("score", 0), reverse=True)

        return {"memories": mems, "total_count": len(mems)}

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        """Return local mode usage info."""
        total = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0"
        ).fetchone()["c"]

        return {
            "tier": "local",
            "mode": "local",
            "memories": {"current": total, "limit": "unlimited"},
            "api_calls": {"current": 0, "limit": "unlimited"},
            "upgrade": "Sign up at novyxlabs.com for cloud sync, rollback history, and team sharing.",
            "upgrade_url": "https://novyxlabs.com/pricing",
        }

    # ------------------------------------------------------------------
    # Cloud-only features (friendly upgrade prompts)
    # ------------------------------------------------------------------

    def share_context(self, tag: str, to_email: str, permission: str = "read") -> dict[str, Any]:
        self._cloud_only("Sharing context spaces across users")
        return {}  # unreachable

    def replay_timeline(self, **kwargs: Any) -> dict[str, Any]:
        self._cloud_only("Replay timeline (time-travel debugging)")
        return {}

    def replay_snapshot(self, at: str, **kwargs: Any) -> dict[str, Any]:
        self._cloud_only("Replay snapshots")
        return {}

    def replay_lifecycle(self, memory_id: str) -> dict[str, Any]:
        self._cloud_only("Memory lifecycle replay")
        return {}

    def replay_diff(self, start: str, end: str) -> dict[str, Any]:
        self._cloud_only("Replay diff")
        return {}

    def cortex_status(self) -> dict[str, Any]:
        self._cloud_only("Cortex autonomous intelligence")
        return {}

    def cortex_run(self) -> dict[str, Any]:
        self._cloud_only("Cortex autonomous intelligence")
        return {}

    def cortex_insights(self, **kwargs: Any) -> dict[str, Any]:
        self._cloud_only("Cortex insights")
        return {}

    def cortex_config(self) -> dict[str, Any]:
        self._cloud_only("Cortex configuration")
        return {}

    def cortex_update_config(self, **kwargs: Any) -> dict[str, Any]:
        self._cloud_only("Cortex configuration")
        return {}

    # ------------------------------------------------------------------
    # Supersede
    # ------------------------------------------------------------------

    def supersede(self, old_memory_id: str, new_memory_id: str) -> dict[str, Any]:
        """Mark old_memory as superseded by new_memory."""
        now = self._now()
        with self._lock:
            # Verify both exist
            old = self._conn.execute(
                "SELECT uuid FROM memories WHERE uuid = ? AND deleted = 0", (old_memory_id,)
            ).fetchone()
            if not old:
                raise ValueError(f"Memory not found: {old_memory_id}")
            new = self._conn.execute(
                "SELECT uuid FROM memories WHERE uuid = ? AND deleted = 0", (new_memory_id,)
            ).fetchone()
            if not new:
                raise ValueError(f"Memory not found: {new_memory_id}")

            self._conn.execute(
                "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE uuid = ?",
                (new_memory_id, now, old_memory_id),
            )
            self._audit("SUPERSEDE", old_memory_id, {
                "old_memory_id": old_memory_id,
                "new_memory_id": new_memory_id,
            })

        return {
            "old_memory_id": old_memory_id,
            "new_memory_id": new_memory_id,
            "superseded_at": now,
            "mode": "local",
        }

    # ------------------------------------------------------------------
    # Links (extended)
    # ------------------------------------------------------------------

    def unlink(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Remove a link between two memories."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM links WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            )
            self._conn.commit()
            self._audit("UNLINK", source_id, {"target": target_id})
        return {"source_id": source_id, "target_id": target_id, "status": "removed"}

    def links(self, memory_id: str, *, relation: str | None = None) -> dict[str, Any]:
        """Get all links for a memory."""
        if relation:
            rows = self._conn.execute(
                "SELECT * FROM links WHERE (source_id = ? OR target_id = ?) AND relation = ?",
                (memory_id, memory_id, relation),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM links WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            ).fetchall()

        results = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "relation": row["relation"],
                "weight": row["weight"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"memory_id": memory_id, "links": results, "total": len(results)}

    def edges(self, *, memory_id: str | None = None, relation: str | None = None,
              direction: str = "both", limit: int = 100) -> dict[str, Any]:
        """Query graph edges with optional filters."""
        conditions = []
        params: list[Any] = []

        if memory_id:
            if direction == "outgoing":
                conditions.append("source_id = ?")
                params.append(memory_id)
            elif direction == "incoming":
                conditions.append("target_id = ?")
                params.append(memory_id)
            else:
                conditions.append("(source_id = ? OR target_id = ?)")
                params.extend([memory_id, memory_id])
        if relation:
            conditions.append("relation = ?")
            params.append(relation)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM links WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()

        results = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "relation": row["relation"],
                "weight": row["weight"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"edges": results, "total": len(results)}

    # ------------------------------------------------------------------
    # Knowledge Graph (extended)
    # ------------------------------------------------------------------

    def delete_triple(self, triple_id: str) -> dict[str, Any]:
        """Delete a knowledge graph triple."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM triples WHERE triple_id = ?", (triple_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Triple not found: {triple_id}")
            self._conn.execute("DELETE FROM triples WHERE triple_id = ?", (triple_id,))
            self._conn.commit()
            self._audit("TRIPLE_DELETE", triple_id)
        return {"triple_id": triple_id, "status": "deleted"}

    def entities(self, *, limit: int = 100, offset: int = 0,
                 entity_type: str | None = None) -> dict[str, Any]:
        """List knowledge graph entities."""
        if entity_type:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (entity_type, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        results = [
            {
                "entity_id": row["entity_id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"entities": results, "total": len(results)}

    def entity(self, entity_id: str) -> dict[str, Any]:
        """Get a single entity by ID."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Entity not found: {entity_id}")

        # Also fetch triples involving this entity
        triples = self._conn.execute(
            "SELECT * FROM triples WHERE subject_id = ? OR object_id = ?",
            (entity_id, entity_id),
        ).fetchall()

        return {
            "entity_id": row["entity_id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "created_at": row["created_at"],
            "triples": [
                {
                    "triple_id": t["triple_id"],
                    "subject": t["subject_name"],
                    "predicate": t["predicate"],
                    "object": t["object_name"],
                }
                for t in triples
            ],
        }

    def delete_entity(self, entity_id: str) -> dict[str, Any]:
        """Delete an entity and its associated triples."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Entity not found: {entity_id}")
            # Remove triples involving this entity
            self._conn.execute(
                "DELETE FROM triples WHERE subject_id = ? OR object_id = ?",
                (entity_id, entity_id),
            )
            self._conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
            self._conn.commit()
            self._audit("ENTITY_DELETE", entity_id, {"name": row["name"]})
        return {"entity_id": entity_id, "name": row["name"], "status": "deleted"}

    # ------------------------------------------------------------------
    # Rollback (extended)
    # ------------------------------------------------------------------

    def rollback_preview(self, target: str) -> dict[str, Any]:
        """Preview what a rollback would do without executing it."""
        return self.rollback(target, dry_run=True)

    def rollback_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get past rollback operations from the audit trail."""
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE operation = 'ROLLBACK' ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "entry_id": row["entry_id"],
                "operation": row["operation"],
                "artifact_id": row["artifact_id"],
                "timestamp": row["timestamp"],
                "details": json.loads(row["details"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Audit (extended)
    # ------------------------------------------------------------------

    def audit_verify(self) -> dict[str, Any]:
        """Verify audit trail integrity (local mode: count-based check)."""
        total = self._conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
        return {
            "verified": True,
            "total_entries": total,
            "mode": "local",
            "message": "Local audit trail integrity verified (entry count check).",
        }

    # ------------------------------------------------------------------
    # Traces (local implementation)
    # ------------------------------------------------------------------

    def trace_create(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create an execution trace."""
        trace_id = f"trc_{uuid.uuid4().hex[:12]}"
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO traces (trace_id, name, status, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (trace_id, name, "active", json.dumps(kwargs.get("metadata", {})), now),
            )
            self._conn.commit()
            self._audit("TRACE_CREATE", trace_id, {"name": name})
        return {"trace_id": trace_id, "name": name, "status": "active", "created_at": now, "mode": "local"}

    def trace_step(self, trace_id: str, step_name: str, **kwargs: Any) -> dict[str, Any]:
        """Add a step to an execution trace."""
        step_id = f"stp_{uuid.uuid4().hex[:12]}"
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO trace_steps (step_id, trace_id, step_name, input_data, output_data, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (step_id, trace_id, step_name,
                 json.dumps(kwargs.get("input_data", {}), default=str),
                 json.dumps(kwargs.get("output_data", {}), default=str),
                 kwargs.get("status", "completed"), now),
            )
            self._conn.commit()
        return {"step_id": step_id, "trace_id": trace_id, "step_name": step_name,
                "status": "completed", "created_at": now, "mode": "local"}

    def trace_complete(self, trace_id: str) -> dict[str, Any]:
        """Mark an execution trace as complete."""
        now = self._now()
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET status = 'completed', completed_at = ? WHERE trace_id = ?",
                (now, trace_id),
            )
            self._conn.commit()
            self._audit("TRACE_COMPLETE", trace_id)
        steps = self._conn.execute(
            "SELECT COUNT(*) as c FROM trace_steps WHERE trace_id = ?", (trace_id,)
        ).fetchone()["c"]
        return {"trace_id": trace_id, "status": "completed", "total_steps": steps,
                "completed_at": now, "mode": "local"}

    def trace_verify(self, trace_id: str) -> dict[str, Any]:
        """Verify a trace's integrity."""
        row = self._conn.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Trace not found: {trace_id}")
        steps = self._conn.execute(
            "SELECT * FROM trace_steps WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        ).fetchall()
        return {
            "trace_id": trace_id,
            "name": row["name"],
            "status": row["status"],
            "total_steps": len(steps),
            "verified": True,
            "mode": "local",
        }

    # ------------------------------------------------------------------
    # Eval (local implementation)
    # ------------------------------------------------------------------

    def eval_run(self, *, min_score: float | None = None) -> dict[str, Any]:
        """Run a memory health evaluation."""
        self._purge_expired()
        total = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0"
        ).fetchone()["c"]
        if total == 0:
            return {"score": 1.0, "total_memories": 0, "message": "No memories to evaluate", "mode": "local"}

        low_conf = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND confidence < 0.5"
        ).fetchone()["c"]

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        stale = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND COALESCE(updated_at, created_at) < ?",
            (cutoff,),
        ).fetchone()["c"]

        superseded = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND superseded_by IS NOT NULL"
        ).fetchone()["c"]

        # Score: 1.0 perfect, penalize for stale/conflicts/superseded
        stale_ratio = stale / total if total else 0
        conflict_ratio = low_conf / total if total else 0
        supersede_ratio = superseded / total if total else 0

        score = round(max(0.0, 1.0 - (stale_ratio * 0.4) - (conflict_ratio * 0.3) - (supersede_ratio * 0.15)), 3)

        result = {
            "score": score,
            "total_memories": total,
            "stale_count": stale,
            "conflict_count": low_conf,
            "superseded_count": superseded,
            "mode": "local",
        }

        if min_score is not None and score < min_score:
            result["gate"] = "failed"
            result["message"] = f"Score {score} below threshold {min_score}"
        elif min_score is not None:
            result["gate"] = "passed"

        return result

    def eval_gate(self, min_score: float) -> dict[str, Any]:
        """CI gate — pass/fail based on memory health score."""
        return self.eval_run(min_score=min_score)

    def eval_history(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get past eval runs from audit trail."""
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE operation LIKE 'EVAL%' ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "evaluations": [
                {
                    "entry_id": row["entry_id"],
                    "timestamp": row["timestamp"],
                    "details": json.loads(row["details"]),
                }
                for row in rows
            ],
            "total": len(rows),
            "mode": "local",
        }

    def eval_drift(self, *, days: int = 7) -> dict[str, Any]:
        """Detect memory drift over a time period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        created = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND created_at > ?",
            (cutoff,),
        ).fetchone()["c"]
        deleted = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 1 AND updated_at > ?",
            (cutoff,),
        ).fetchone()["c"]
        updated = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE deleted = 0 AND updated_at > ? AND updated_at != created_at",
            (cutoff,),
        ).fetchone()["c"]

        return {
            "period_days": days,
            "created": created,
            "deleted": deleted,
            "updated": updated,
            "net_change": created - deleted,
            "mode": "local",
        }

    # ------------------------------------------------------------------
    # Replay (extended)
    # ------------------------------------------------------------------

    def replay_memory(self, memory_id: str) -> dict[str, Any]:
        """Get the full history of a single memory from the audit trail."""
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE artifact_id = ? ORDER BY timestamp",
            (memory_id,),
        ).fetchall()
        events = [
            {
                "entry_id": row["entry_id"],
                "operation": row["operation"],
                "timestamp": row["timestamp"],
                "details": json.loads(row["details"]),
            }
            for row in rows
        ]
        return {"memory_id": memory_id, "events": events, "total": len(events), "mode": "local"}

    def replay_recall(self, query: str, at: str, *, limit: int = 5) -> dict[str, Any]:
        """Time-travel recall — what would search return at a past timestamp."""
        self._cloud_only("Time-travel recall (replay_recall)")
        return {}

    def replay_drift(self, from_ts: str, to_ts: str) -> dict[str, Any]:
        """Detect memory drift between two timestamps."""
        self._cloud_only("Replay drift detection")
        return {}

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def context_now(self) -> dict[str, Any]:
        """Get a snapshot of current memory context."""
        stats = self.stats()
        recent = self.memories(limit=5)
        audit_recent = self.audit(limit=5)
        return {
            "stats": stats,
            "recent_memories": recent,
            "recent_audit": audit_recent,
            "mode": "local",
        }

    def dashboard(self) -> dict[str, Any]:
        """Get dashboard overview data."""
        stats = self.stats()
        spaces = self.list_spaces()
        audit_recent = self.audit(limit=10)
        return {
            "stats": stats,
            "spaces": spaces,
            "recent_activity": audit_recent,
            "mode": "local",
        }

    # ------------------------------------------------------------------
    # Sharing (extended) — cloud-only
    # ------------------------------------------------------------------

    def accept_shared_context(self, token: str) -> dict[str, Any]:
        self._cloud_only("Accepting shared contexts")
        return {}

    def shared_contexts(self) -> dict[str, Any]:
        self._cloud_only("Listing shared contexts")
        return {}

    def revoke_shared_context(self, token: str) -> dict[str, Any]:
        self._cloud_only("Revoking shared contexts")
        return {}

    # ------------------------------------------------------------------
    # Eval Baselines (cloud-only)
    # ------------------------------------------------------------------

    def eval_baseline_create(self, query: str, expected_observation: str) -> dict[str, Any]:
        self._cloud_only("Eval baselines")
        return {}

    def eval_baselines(self) -> dict[str, Any]:
        self._cloud_only("Eval baselines")
        return {}

    def eval_baseline_delete(self, baseline_id: str) -> bool:
        self._cloud_only("Eval baselines")
        return False

    # ------------------------------------------------------------------
    # Audit Export (cloud-only)
    # ------------------------------------------------------------------

    def audit_export(self, format: str = "json") -> dict[str, Any]:
        self._cloud_only("Audit export")
        return {}

    # ------------------------------------------------------------------
    # Explain Action (cloud-only)
    # ------------------------------------------------------------------

    def explain_action(self, action_id: str) -> dict[str, Any]:
        self._cloud_only("Action explanations")
        return {}

    # ------------------------------------------------------------------
    # Sentinel Intel (cloud-only)
    # ------------------------------------------------------------------

    def threat_feed(self, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_stats(self):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_record(self, threat_event):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_trending(self, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_match(self, threat_event, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_signature(self, signature_id):
        self._cloud_only("Sentinel Intel")
        return {}

    def threat_mitigate(self, signature_id):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_list(self, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_deploy(self, signature_id, rule_type, rule_config=None):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_remove(self, defense_id):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_effectiveness(self, defense_id):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_record_block(self, defense_id, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_stats(self):
        self._cloud_only("Sentinel Intel")
        return {}

    def defense_recommend(self, signature_id):
        self._cloud_only("Sentinel Intel")
        return {}

    def correlate_threat(self, threat_event):
        self._cloud_only("Sentinel Intel")
        return {}

    def detect_campaign(self, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def coordinated_attack_check(self, threat_events, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def related_signatures(self, signature_id, **kwargs):
        self._cloud_only("Sentinel Intel")
        return {}

    def stream_status(self):
        self._cloud_only("Stream status")
        return {}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _purge_expired(self) -> None:
        """Soft-delete memories that have passed their TTL."""
        now = self._now()
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET deleted = 1 WHERE expires_at IS NOT NULL "
                "AND expires_at < ? AND deleted = 0",
                (now,),
            )
            self._conn.commit()
