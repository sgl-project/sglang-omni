SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_state (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    state TEXT NOT NULL,
    draft INTEGER NOT NULL,
    labels_json TEXT NOT NULL,
    installation_id INTEGER,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (repo, pr_number)
);

CREATE TABLE IF NOT EXISTS queue_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    stage_key TEXT NOT NULL,
    check_name TEXT NOT NULL,
    check_run_id INTEGER,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    enqueue_time TEXT NOT NULL,
    dispatch_time TEXT,
    finish_time TEXT,
    workflow_run_id INTEGER,
    dispatch_id TEXT,
    generated_workflow_path TEXT,
    source_hash TEXT,
    conclusion TEXT,
    installation_id INTEGER,
    updated_at TEXT NOT NULL,
    UNIQUE (repo, pr_number, head_sha, stage_key, attempt)
);

CREATE TABLE IF NOT EXISTS stage_outputs (
    repo TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    stage_key TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    outputs_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (repo, head_sha, stage_key, attempt)
);

CREATE INDEX IF NOT EXISTS idx_queue_status
    ON queue_items (repo, status, head_sha);
CREATE INDEX IF NOT EXISTS idx_queue_pr_sha
    ON queue_items (repo, pr_number, head_sha);
"""

SELECT_PENDING = """
SELECT qi.*, ps.labels_json, ps.state AS pr_state, ps.draft AS pr_draft,
       ps.head_sha AS current_head_sha
FROM queue_items qi
JOIN pr_state ps
  ON ps.repo = qi.repo AND ps.pr_number = qi.pr_number
WHERE qi.status = 'pending'
"""

SELECT_ALL_ITEMS = "SELECT * FROM queue_items ORDER BY id"

SELECT_ITEMS_FOR_PR = """
SELECT * FROM queue_items
WHERE repo = ? AND pr_number = ?
ORDER BY head_sha, attempt, id
"""

SELECT_PR_STATES = "SELECT * FROM pr_state"
