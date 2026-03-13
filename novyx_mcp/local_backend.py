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
from typing import Any, Optional

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
                (mem_id, observation, context, "local", tags_json, importance, 1.0,
                 now, embedding_blob, expires_at),
            )
            self._audit("CREATE", mem_id, {"observation": observation, "tags": tags or []})

        return {
            "uuid": mem_id,
            "observation": observation,
            "tags": tags or [],
            "importance": importance,
            "created_at": now,
            "mode": "local",
        }

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
            placeholders = " AND ".join(f"tags LIKE ?" for _ in tags)
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
            placeholders = " AND ".join(f"tags LIKE ?" for _ in tags)
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
