"""Experiment Data Service — FastAPI backend for centralized experiment persistence."""

import hashlib
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from utils.expdata.schema import get_db, init_db

logger = logging.getLogger("expdata")

DB_PATH = os.environ.get("EXPDATA_DB_PATH", "/data/expdata.db")
DASHBOARD_PATH = os.environ.get("EXPDATA_DASHBOARD_PATH", str(Path(__file__).parent.parent.parent / "expdata_dashboard.html"))

_db: Optional[aiosqlite.Connection] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    _db = await init_db(DB_PATH)
    _db.row_factory = aiosqlite.Row
    logger.info(f"Database initialized at {DB_PATH}")
    yield
    if _db:
        await _db.close()


app = FastAPI(title="Experiment Data Service", version="0.1.0", lifespan=lifespan)


def db() -> aiosqlite.Connection:
    assert _db is not None, "Database not initialized"
    return _db


# ---------------------------------------------------------------------------
# Auth — simple username/password, optional for all endpoints
# ---------------------------------------------------------------------------

async def require_user(authorization: Optional[str] = Header(None)) -> str:
    """Require valid Bearer token. Raises 401 if missing or invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required. Login first: POST /api/v1/auth/login")
    token = authorization[7:]
    cursor = await db().execute("SELECT username FROM users WHERE api_token = ?", (token,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(401, "Invalid token")
    return row["username"]


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
async def login(req: LoginRequest):
    """Login with username + password. Returns a Bearer token.
    If the user doesn't exist, auto-registers with the given password."""
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    cursor = await db().execute("SELECT api_token, password_hash FROM users WHERE username = ?", (req.username,))
    row = await cursor.fetchone()
    if row:
        if row["password_hash"] != pw_hash:
            raise HTTPException(401, "Wrong password")
        return {"token": row["api_token"], "username": req.username}
    # Auto-register
    token = secrets.token_hex(32)
    await db().execute(
        "INSERT INTO users (username, password_hash, api_token) VALUES (?, ?, ?)",
        (req.username, pw_hash, token),
    )
    await db().commit()
    return {"token": token, "username": req.username, "created": True}


@app.get("/api/v1/auth/whoami")
async def whoami(user: str = Depends(require_user)):
    return {"username": user}


# ---------------------------------------------------------------------------
# Experiments CRUD
# ---------------------------------------------------------------------------

class CreateExperimentRequest(BaseModel):
    name: str
    type: str  # eval | collection
    model: Optional[str] = None
    backend: Optional[str] = None
    scaffold: Optional[str] = None
    dataset: Optional[str] = None
    mode: Optional[str] = None
    n_samples: Optional[int] = None
    config_json: Optional[str] = None


@app.post("/api/v1/experiments")
async def create_experiment(req: CreateExperimentRequest, user: str = Depends(require_user)):
    if req.type not in ("eval", "collection"):
        raise HTTPException(400, "type must be 'eval' or 'collection'")
    cursor = await db().execute(
        """INSERT INTO experiments (name, type, model, backend, scaffold, dataset, mode, n_samples, config_json, created_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (req.name, req.type, req.model, req.backend, req.scaffold, req.dataset, req.mode, req.n_samples, req.config_json, user),
    )
    await db().commit()
    return {"id": cursor.lastrowid}


@app.get("/api/v1/experiments")
async def list_experiments(
    type: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    user: str = Depends(require_user),
):
    conditions, params = [], []
    if type:
        conditions.append("type = ?")
        params.append(type)
    if model:
        conditions.append("model LIKE ?")
        params.append(f"%{model}%")
    if status:
        conditions.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    cursor = await db().execute(
        f"SELECT * FROM experiments{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    rows = await cursor.fetchall()
    return {"experiments": [dict(r) for r in rows]}


@app.get("/api/v1/experiments/{exp_id}")
async def get_experiment(exp_id: int, user: str = Depends(require_user)):
    cursor = await db().execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Experiment not found")
    result = dict(row)
    # Attach counts
    for table in ("eval_results", "trajectories", "fault_traces", "test_outputs"):
        c = await db().execute(f"SELECT COUNT(*) FROM {table} WHERE experiment_id = ?", (exp_id,))
        count = (await c.fetchone())[0]
        result[f"{table}_count"] = count
    return result


class PatchExperimentRequest(BaseModel):
    status: Optional[str] = None
    summary_json: Optional[str] = None


@app.patch("/api/v1/experiments/{exp_id}")
async def patch_experiment(exp_id: int, req: PatchExperimentRequest, user: str = Depends(require_user)):
    updates, params = [], []
    if req.status:
        updates.append("status = ?")
        params.append(req.status)
    if req.summary_json:
        updates.append("summary_json = ?")
        params.append(req.summary_json)
    if not updates:
        raise HTTPException(400, "No fields to update")
    params.append(exp_id)
    await db().execute(f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?", params)
    await db().commit()
    return {"ok": True}


@app.delete("/api/v1/experiments/{exp_id}")
async def delete_experiment(exp_id: int, user: str = Depends(require_user)):
    await db().execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    await db().commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Bulk Upload Endpoints
# ---------------------------------------------------------------------------

async def _ensure_experiment(exp_id: int):
    cursor = await db().execute("SELECT id FROM experiments WHERE id = ?", (exp_id,))
    if not await cursor.fetchone():
        raise HTTPException(404, "Experiment not found")


@app.post("/api/v1/experiments/{exp_id}/upload/eval-results")
async def upload_eval_results(exp_id: int, request: Request, user: str = Depends(require_user)):
    await _ensure_experiment(exp_id)
    body = await request.body()
    rows = []
    for line in body.decode().strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((
            exp_id, r.get("instance_id", ""), r.get("repo"), r.get("uid"),
            r.get("sample_idx", 0), r.get("reward"), r.get("termination_reason"),
            json.dumps({k: v for k, v in r.items() if k not in ("instance_id", "repo", "uid", "sample_idx", "reward", "termination_reason")}),
        ))
    await db().executemany(
        "INSERT INTO eval_results (experiment_id, instance_id, repo, uid, sample_idx, reward, termination_reason, extra_json) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    await db().commit()
    return {"inserted": len(rows)}


@app.post("/api/v1/experiments/{exp_id}/upload/trajectories")
async def upload_trajectories(exp_id: int, request: Request, user: str = Depends(require_user)):
    await _ensure_experiment(exp_id)
    body = await request.body()
    rows = []
    for line in body.decode().strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((
            exp_id, r.get("instance_id", ""), r.get("sample_idx", 0),
            r.get("reward"), r.get("termination_reason"), r.get("n_steps"),
            json.dumps(r.get("messages", [])), r.get("role", "primary"),
        ))
    await db().executemany(
        "INSERT INTO trajectories (experiment_id, instance_id, sample_idx, reward, termination_reason, n_steps, messages_json, role) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    await db().commit()
    return {"inserted": len(rows)}


@app.post("/api/v1/experiments/{exp_id}/upload/fault-traces")
async def upload_fault_traces(exp_id: int, request: Request, user: str = Depends(require_user)):
    await _ensure_experiment(exp_id)
    body = await request.body()
    data = json.loads(body)  # {instance_id: [traces]}
    rows = []
    for instance_id, traces in data.items():
        rows.append((exp_id, instance_id, json.dumps(traces), len(traces)))
    await db().executemany(
        "INSERT OR REPLACE INTO fault_traces (experiment_id, instance_id, traces_json, n_traces) VALUES (?,?,?,?)",
        rows,
    )
    await db().commit()
    return {"inserted": len(rows)}


@app.post("/api/v1/experiments/{exp_id}/upload/test-outputs")
async def upload_test_outputs(exp_id: int, request: Request, user: str = Depends(require_user)):
    await _ensure_experiment(exp_id)
    body = await request.body()
    data = json.loads(body)  # {instance_id: output_text}
    rows = [(exp_id, iid, text) for iid, text in data.items()]
    await db().executemany(
        "INSERT INTO test_outputs (experiment_id, instance_id, output_text) VALUES (?,?,?)",
        rows,
    )
    await db().commit()
    return {"inserted": len(rows)}


@app.post("/api/v1/experiments/{exp_id}/upload/localization")
async def upload_localization(exp_id: int, request: Request, user: str = Depends(require_user)):
    await _ensure_experiment(exp_id)
    body = await request.body()
    data = json.loads(body)
    await db().execute(
        "INSERT OR REPLACE INTO localization_analyses (experiment_id, aggregate_json, per_instance_json) VALUES (?,?,?)",
        (exp_id, json.dumps(data.get("aggregate", {})), json.dumps(data.get("per_instance", []))),
    )
    await db().commit()
    return {"ok": True}


@app.post("/api/v1/experiments/{exp_id}/upload/collection")
async def upload_collection(exp_id: int, file: UploadFile, user: str = Depends(require_user)):
    """Upload collection parquet — parse and store as trajectories."""
    await _ensure_experiment(exp_id)
    import io
    import pandas as pd
    content = await file.read()
    df = pd.read_parquet(io.BytesIO(content))
    rows = []
    for _, row in df.iterrows():
        messages = row.get("messages", "[]")
        if isinstance(messages, str):
            messages_json = messages
        else:
            messages_json = json.dumps(messages)
        role = "primary"
        if "chosen" in df.columns and "rejected" in df.columns:
            # DPO format — handle differently
            pass
        rows.append((
            exp_id, row.get("instance_id", ""), 0,
            row.get("reward"), row.get("termination_reason"),
            None, messages_json, role,
        ))
    if rows:
        await db().executemany(
            "INSERT INTO trajectories (experiment_id, instance_id, sample_idx, reward, termination_reason, n_steps, messages_json, role) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        await db().commit()
    return {"inserted": len(rows)}


# ---------------------------------------------------------------------------
# Query Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/experiments/{exp_id}/results")
async def get_results(
    exp_id: int,
    instance_id: Optional[str] = None,
    reward: Optional[float] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    user: str = Depends(require_user),
):
    await _ensure_experiment(exp_id)
    conditions = ["experiment_id = ?"]
    params: list = [exp_id]
    if instance_id:
        conditions.append("instance_id LIKE ?")
        params.append(f"%{instance_id}%")
    if reward is not None:
        conditions.append("reward = ?")
        params.append(reward)
    where = " WHERE " + " AND ".join(conditions)
    cursor = await db().execute(
        f"SELECT * FROM eval_results{where} ORDER BY id LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    rows = await cursor.fetchall()
    # Count total
    cnt = await db().execute(f"SELECT COUNT(*) FROM eval_results{where}", params)
    total = (await cnt.fetchone())[0]
    return {"results": [dict(r) for r in rows], "total": total}


@app.get("/api/v1/experiments/{exp_id}/trajectories")
async def list_trajectories(
    exp_id: int,
    instance_id: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    user: str = Depends(require_user),
):
    await _ensure_experiment(exp_id)
    conditions = ["experiment_id = ?"]
    params: list = [exp_id]
    if instance_id:
        conditions.append("instance_id LIKE ?")
        params.append(f"%{instance_id}%")
    if role:
        conditions.append("role = ?")
        params.append(role)
    where = " WHERE " + " AND ".join(conditions)
    # Return metadata only (no messages_json)
    cursor = await db().execute(
        f"SELECT id, experiment_id, instance_id, sample_idx, reward, termination_reason, n_steps, role FROM trajectories{where} ORDER BY id LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    rows = await cursor.fetchall()
    cnt = await db().execute(f"SELECT COUNT(*) FROM trajectories{where}", params)
    total = (await cnt.fetchone())[0]
    return {"trajectories": [dict(r) for r in rows], "total": total}


@app.get("/api/v1/experiments/{exp_id}/trajectories/{tid}")
async def get_trajectory(exp_id: int, tid: int, user: str = Depends(require_user)):
    cursor = await db().execute(
        "SELECT * FROM trajectories WHERE id = ? AND experiment_id = ?", (tid, exp_id)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Trajectory not found")
    result = dict(row)
    if result.get("messages_json"):
        result["messages"] = json.loads(result["messages_json"])
    return result


@app.get("/api/v1/experiments/{exp_id}/fault-traces/{instance_id}")
async def get_fault_traces(exp_id: int, instance_id: str, user: str = Depends(require_user)):
    cursor = await db().execute(
        "SELECT * FROM fault_traces WHERE experiment_id = ? AND instance_id = ?",
        (exp_id, instance_id),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Fault traces not found")
    result = dict(row)
    if result.get("traces_json"):
        result["traces"] = json.loads(result["traces_json"])
    return result


@app.get("/api/v1/experiments/{exp_id}/test-outputs/{instance_id}")
async def get_test_output(exp_id: int, instance_id: str, user: str = Depends(require_user)):
    cursor = await db().execute(
        "SELECT * FROM test_outputs WHERE experiment_id = ? AND instance_id = ?",
        (exp_id, instance_id),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Test output not found")
    return dict(row)


@app.get("/api/v1/experiments/{exp_id}/localization")
async def get_localization(exp_id: int, user: str = Depends(require_user)):
    cursor = await db().execute(
        "SELECT * FROM localization_analyses WHERE experiment_id = ?", (exp_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Localization analysis not found")
    result = dict(row)
    if result.get("aggregate_json"):
        result["aggregate"] = json.loads(result["aggregate_json"])
    if result.get("per_instance_json"):
        result["per_instance"] = json.loads(result["per_instance_json"])
    return result


# ---------------------------------------------------------------------------
# Comparison Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/compare")
async def compare_experiments(ids: str = Query(..., description="Comma-separated experiment IDs"), user: str = Depends(require_user)):
    exp_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
    if not exp_ids or len(exp_ids) > 10:
        raise HTTPException(400, "Provide 1-10 experiment IDs")
    placeholders = ",".join("?" * len(exp_ids))
    cursor = await db().execute(
        f"SELECT * FROM experiments WHERE id IN ({placeholders})", exp_ids
    )
    experiments = [dict(r) for r in await cursor.fetchall()]

    # Get per-experiment stats
    for exp in experiments:
        eid = exp["id"]
        c = await db().execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN reward > 0 THEN 1 ELSE 0 END) as solved FROM eval_results WHERE experiment_id = ?",
            (eid,),
        )
        stats = dict(await c.fetchone())
        exp["total_results"] = stats["total"]
        exp["solved"] = stats["solved"] or 0
        exp["solve_rate"] = (stats["solved"] or 0) / stats["total"] if stats["total"] else 0

    return {"experiments": experiments}


@app.get("/api/v1/instances/{instance_id}/history")
async def instance_history(instance_id: str, user: str = Depends(require_user)):
    cursor = await db().execute(
        """SELECT er.*, e.name as experiment_name, e.model, e.created_at as experiment_date
           FROM eval_results er JOIN experiments e ON er.experiment_id = e.id
           WHERE er.instance_id = ? ORDER BY e.created_at DESC""",
        (instance_id,),
    )
    rows = await cursor.fetchall()
    return {"history": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Health check (no auth required)
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = Path(DASHBOARD_PATH)
    if not path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1><p>Place expdata_dashboard.html alongside the server.</p>", 404)
    return FileResponse(path, media_type="text/html")
