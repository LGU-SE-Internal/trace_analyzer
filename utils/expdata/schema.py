"""SQLite schema initialization and migrations for expdata service."""

import aiosqlite

SCHEMA_VERSION = 1

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    api_token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('eval', 'collection')),
    model TEXT,
    backend TEXT,
    scaffold TEXT,
    dataset TEXT,
    mode TEXT,
    n_samples INTEGER,
    config_json TEXT,
    status TEXT NOT NULL DEFAULT 'created' CHECK(status IN ('created', 'uploading', 'completed', 'failed')),
    summary_json TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    repo TEXT,
    uid TEXT,
    sample_idx INTEGER DEFAULT 0,
    reward REAL,
    termination_reason TEXT,
    extra_json TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    sample_idx INTEGER DEFAULT 0,
    reward REAL,
    termination_reason TEXT,
    n_steps INTEGER,
    messages_json TEXT,
    role TEXT DEFAULT 'primary' CHECK(role IN ('primary', 'chosen', 'rejected')),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fault_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    instance_id TEXT NOT NULL UNIQUE,
    traces_json TEXT,
    n_traces INTEGER DEFAULT 0,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    output_text TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS localization_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL UNIQUE,
    aggregate_json TEXT,
    per_instance_json TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_results_exp ON eval_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_instance ON eval_results(instance_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_exp ON trajectories(experiment_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_instance ON trajectories(instance_id);
CREATE INDEX IF NOT EXISTS idx_fault_traces_exp ON fault_traces(experiment_id);
CREATE INDEX IF NOT EXISTS idx_fault_traces_instance ON fault_traces(instance_id);
CREATE INDEX IF NOT EXISTS idx_test_outputs_exp ON test_outputs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_test_outputs_instance ON test_outputs(instance_id);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(type);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
"""


async def init_db(db_path: str) -> aiosqlite.Connection:
    """Initialize database with schema and WAL mode."""
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.executescript(TABLES_SQL)

    # Check/set schema version
    cursor = await db.execute("SELECT version FROM schema_version")
    row = await cursor.fetchone()
    if row is None:
        await db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    await db.commit()
    return db


async def get_db(db_path: str) -> aiosqlite.Connection:
    """Get a database connection (assumes already initialized)."""
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = aiosqlite.Row
    return db
