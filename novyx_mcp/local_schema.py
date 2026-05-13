"""
SQLite schema and database initialization for Novyx local mode.

Stores memories, links, knowledge graph, context spaces, and audit trail
locally with no external dependencies.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5

SCHEMA_SQL = """
-- Memory store
CREATE TABLE IF NOT EXISTS memories (
    uuid          TEXT PRIMARY KEY,
    observation   TEXT NOT NULL,
    context       TEXT,
    agent_id      TEXT DEFAULT 'local',
    tags          TEXT DEFAULT '[]',   -- JSON array
    importance    INTEGER DEFAULT 5,
    confidence    REAL DEFAULT 1.0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT,
    embedding     BLOB,                -- packed float32 array
    expires_at    TEXT,
    superseded_by TEXT,
    deleted       INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

-- Memory links (directed graph edges)
CREATE TABLE IF NOT EXISTS links (
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    relation   TEXT NOT NULL DEFAULT 'related',
    weight     REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation)
);

-- Knowledge graph entities
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    created_at  TEXT NOT NULL
);

-- Knowledge graph triples
CREATE TABLE IF NOT EXISTS triples (
    triple_id        TEXT PRIMARY KEY,
    subject_id       TEXT NOT NULL REFERENCES entities(entity_id),
    subject_name     TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    object_id        TEXT NOT NULL REFERENCES entities(entity_id),
    object_name      TEXT NOT NULL,
    confidence       REAL DEFAULT 1.0,
    source_memory_id TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_name);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_name);

-- Context spaces
CREATE TABLE IF NOT EXISTS spaces (
    space_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT,
    allowed_agent_ids TEXT DEFAULT '[]',  -- JSON array
    tags              TEXT DEFAULT '[]',  -- JSON array
    created_at        TEXT NOT NULL
);

-- Space-memory associations
CREATE TABLE IF NOT EXISTS space_members (
    space_id  TEXT NOT NULL REFERENCES spaces(space_id),
    memory_id TEXT NOT NULL REFERENCES memories(uuid),
    PRIMARY KEY (space_id, memory_id)
);

-- Audit trail
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id   TEXT PRIMARY KEY,
    operation  TEXT NOT NULL,
    artifact_id TEXT,
    agent_id   TEXT DEFAULT 'local',
    timestamp  TEXT NOT NULL,
    details    TEXT DEFAULT '{}',  -- JSON object with before/after state
    entry_hash TEXT DEFAULT ''     -- SHA-256 hash chain
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation);

-- Memory drafts
CREATE TABLE IF NOT EXISTS memory_drafts (
    draft_id          TEXT PRIMARY KEY,
    branch_id         TEXT,
    observation       TEXT NOT NULL,
    context           TEXT,
    tags              TEXT DEFAULT '[]',
    importance        INTEGER DEFAULT 5,
    confidence        REAL DEFAULT 1.0,
    review_summary    TEXT DEFAULT '{}',
    status            TEXT NOT NULL DEFAULT 'draft',
    merged_memory_id  TEXT,
    merged_at         TEXT,
    rejected_at       TEXT,
    rejection_reason  TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_drafts_status ON memory_drafts(status);
CREATE INDEX IF NOT EXISTS idx_memory_drafts_updated ON memory_drafts(updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_drafts_branch ON memory_drafts(branch_id);

-- Execution traces
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    metadata     TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);

-- Trace steps
CREATE TABLE IF NOT EXISTS trace_steps (
    step_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL REFERENCES traces(trace_id),
    step_name   TEXT NOT NULL,
    input_data  TEXT DEFAULT '{}',
    output_data TEXT DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'completed',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trace_steps_trace ON trace_steps(trace_id);

-- Governance policies
CREATE TABLE IF NOT EXISTS policies (
    name        TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    rules       TEXT DEFAULT '[]',   -- JSON array of rule objects
    enabled     INTEGER DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Governed actions
CREATE TABLE IF NOT EXISTS actions (
    action_id   TEXT PRIMARY KEY,
    action      TEXT NOT NULL,
    params      TEXT DEFAULT '{}',   -- JSON
    status      TEXT NOT NULL,       -- allowed, blocked, pending_review, approved, denied
    policy_result TEXT DEFAULT '{}', -- JSON
    message     TEXT DEFAULT '',
    agent_id    TEXT DEFAULT 'local',
    approver_id TEXT,
    decided_at  TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_created ON actions(created_at);

-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the SQLite database, creating tables if needed.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Check if schema needs initialization
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cursor.fetchone() is None:
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    else:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row and row[0] < SCHEMA_VERSION:
            _migrate(conn, row[0], SCHEMA_VERSION)

    return conn


def _migrate(conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
    """Run schema migrations between versions."""
    if from_version < 2 <= to_version:
        conn.execute("ALTER TABLE memories ADD COLUMN superseded_by TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_drafts (
                draft_id          TEXT PRIMARY KEY,
                observation       TEXT NOT NULL,
                context           TEXT,
                tags              TEXT DEFAULT '[]',
                importance        INTEGER DEFAULT 5,
                confidence        REAL DEFAULT 1.0,
                review_summary    TEXT DEFAULT '{}',
                status            TEXT NOT NULL DEFAULT 'draft',
                merged_memory_id  TEXT,
                merged_at         TEXT,
                rejected_at       TEXT,
                rejection_reason  TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_drafts_status ON memory_drafts(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_drafts_updated ON memory_drafts(updated_at)"
        )
    if from_version < 3 <= to_version:
        conn.execute("ALTER TABLE memory_drafts ADD COLUMN branch_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_drafts_branch ON memory_drafts(branch_id)"
        )
    if from_version < 4 <= to_version:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id     TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'active',
                metadata     TEXT DEFAULT '{}',
                created_at   TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_steps (
                step_id     TEXT PRIMARY KEY,
                trace_id    TEXT NOT NULL REFERENCES traces(trace_id),
                step_name   TEXT NOT NULL,
                input_data  TEXT DEFAULT '{}',
                output_data TEXT DEFAULT '{}',
                status      TEXT NOT NULL DEFAULT 'completed',
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_steps_trace ON trace_steps(trace_id)")
    if from_version < 5 <= to_version:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policies (
                name        TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                rules       TEXT DEFAULT '[]',
                enabled     INTEGER DEFAULT 1,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                action_id   TEXT PRIMARY KEY,
                action      TEXT NOT NULL,
                params      TEXT DEFAULT '{}',
                status      TEXT NOT NULL,
                policy_result TEXT DEFAULT '{}',
                message     TEXT DEFAULT '',
                agent_id    TEXT DEFAULT 'local',
                approver_id TEXT,
                decided_at  TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_created ON actions(created_at)")
        # Add hash chain column to audit_log for cryptographic verification
        try:
            conn.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists
    conn.execute("UPDATE schema_version SET version = ?", (to_version,))
    conn.commit()
