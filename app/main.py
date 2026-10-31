import asyncio, json, sqlite3, threading, time
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app import config as C
from ml.core import FS, N, predict, process_tags, simulate_window
from ml.train import train

lock = threading.Lock()
db = None
bundle = None
sim = {m["tag"]: 0.0 for m in C.MACHINES}
rng = np.random.default_rng()
SCHEMA = """
CREATE TABLE IF NOT EXISTS readings(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tag TEXT, rms REAL, temp REAL, oil REAL,
 rpm REAL, anomaly REAL, fault TEXT, fault_conf REAL, rul REAL, rul_lo REAL, rul_hi REAL, health REAL, stage INT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tag TEXT, stage INT, message TEXT, acked INT DEFAULT 0);
"""


def rows(sql, args=()):
    with lock:
        return [dict(r) for r in db.execute(sql, args).fetchall()]


def record(tag, x, temp, oil, rpm):
    p = predict(bundle, x, temp, oil)
    x = np.asarray(x, float)
    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    with lock:
        prev = db.execute("SELECT stage FROM readings WHERE tag=? ORDER BY id DESC LIMIT 1", (tag,)).fetchone()
        db.execute("INSERT INTO readings(ts,tag,rms,temp,oil,rpm,anomaly,fault,fault_conf,rul,rul_lo,rul_hi,health,stage) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (time.time(), tag, rms, temp, oil, rpm, p["anomaly"], p["fault"], p["fault_conf"], p["rul"],
                    p["rul_lo"], p["rul_hi"], p["health"], p["stage"]))
        if p["stage"] >= 1 and p["stage"] > (prev["stage"] if prev else 0):
            msg = f"{p['fault']} pattern suspected, health {p['health']}, RUL about {p['rul']:.0f} days (model estimate)"
            db.execute("INSERT INTO alerts(ts,tag,stage,message) VALUES(?,?,?,?)", (time.time(), tag, p["stage"], msg))
        db.commit()
    return {**p, "rms": rms}


def tick():
    for m in C.MACHINES:
        tag, s = m["tag"], sim[m["tag"]]
        if m["mode"] == "degrade":
            s += m["rate"]
            if s > 1.05:
                s = 0.0                        # simulates maintenance being performed
            sim[tag] = s
        sev = min(s, 1.0)
        fault = m.get("fault", "normal") if s > 0.05 else "normal"
        t, o = process_tags(sev, rng)
        record(tag, simulate_window(fault, sev, m["rpm"], rng), t, o, m["rpm"])


async def runner():
    while True:
        try:
            await asyncio.to_thread(tick)
        except Exception as e:
            print("simulator error:", e)
        await asyncio.sleep(C.TICK)


@asynccontextmanager
async def lifespan(app):
    global db, bundle
    C.DB.parent.mkdir(exist_ok=True)
    C.MODELS.mkdir(exist_ok=True)
    db = sqlite3.connect(C.DB, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    f = C.MODELS / "bundle.joblib"
    bundle = joblib.load(f) if f.exists() else train()
    task = asyncio.create_task(runner()) if C.SOURCE == "sim" else None
    yield
    if task:
        task.cancel()
    db.close()


app = FastAPI(title="Predictive Maintenance Prototype", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(C.ROOT / "web" / "index.html")


@app.get("/api/machines")
def machines():
    last = {r["tag"]: r for r in rows("SELECT * FROM readings WHERE id IN (SELECT MAX(id) FROM readings GROUP BY tag)")}
    return [{**{k: m[k] for k in ("tag", "name", "plant", "rpm")}, "latest": last.get(m["tag"])} for m in C.MACHINES]


@app.get("/api/machines/{tag}/history")
def history(tag: str, n: int = 120):
    if tag not in sim:
        raise HTTPException(404, "Unknown tag")
    return rows("SELECT * FROM (SELECT * FROM readings WHERE tag=? ORDER BY id DESC LIMIT ?) ORDER BY id", (tag, min(n, 1000)))


@app.get("/api/alerts")
def alerts():
    return rows("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")


@app.post("/api/alerts/{aid}/ack")
def ack(aid: int):
    with lock:
        db.execute("UPDATE alerts SET acked=1 WHERE id=?", (aid,))
        db.commit()
    return {"ok": True}


@app.get("/api/kpis")
def kpis():
    last = rows("SELECT stage FROM readings WHERE id IN (SELECT MAX(id) FROM readings GROUP BY tag)")
    mf = C.MODELS / "metrics.json"
    return {"machines": len(C.MACHINES), "by_stage": [sum(1 for r in last if r["stage"] == i) for i in range(4)],
            "active_alerts": rows("SELECT COUNT(*) c FROM alerts WHERE acked=0")[0]["c"],
            "metrics": json.loads(mf.read_text()) if mf.exists() else {}, "source": C.SOURCE,
            "stages": C.STAGES, "limits": C.STAGE_LIMITS, "fs": FS, "window": N}


class Reading(BaseModel):
    tag: str
    waveform: list[float]
    temp: float
    oil: float
    rpm: float = 1800


@app.post("/api/ingest")
def ingest(r: Reading):
    """Integration point: a read-only gateway (OPC UA / historian export) pushes readings here."""
    if r.tag not in sim:
        raise HTTPException(404, "Unknown tag - add it to MACHINES in app/config.py")
    if len(r.waveform) < N:
        raise HTTPException(422, f"waveform needs at least {N} samples at {FS} Hz")
    return record(r.tag, r.waveform[:N], r.temp, r.oil, r.rpm)
