# Predictive Maintenance Prototype (Engro use case)

Read-only, advisory prototype: simulated plant data -> feature extraction -> three models (anomaly, fault class, RUL) -> health fusion -> alerts -> web console.

## Run
    pip install -r requirements.txt
    python -m ml.train                      # optional, auto-runs on first start
    uvicorn app.main:app --port 8000        # open http://localhost:8000

## Datasets
    python scripts/download_datasets.py        # lists all datasets and links
    python scripts/download_datasets.py ims    # downloads NASA IMS
    python -m ml.ims_demo <extracted test-set folder>   # real-data anomaly check
Only IMS is wired to a runnable script. CWRU, Paderborn, MIMII and C-MAPSS are linked for manual download; their loaders are not included.

## Real vs simulated
- Real: pipeline, API, database, alert logic, UI, IMS check.
- Simulated: sensor data and the training data for all three models (bearing fault simulation at 1800 rpm). Reported accuracy is on synthetic data only.

## Connecting plant data
Set `PM_SOURCE=external` and POST to `/api/ingest` (`tag`, `waveform` >= 2048 samples at 20480 Hz, `temp`, `oil`, `rpm`) from an approved read-only gateway. Add tags in `app/config.py`. No OPC UA client is included, and the app never writes to a PLC/DCS.

## Not included
Login/roles, PostgreSQL/TimescaleDB (SQLite used), MQTT/Kafka, MLflow, maintenance planner and work orders, email/SMS delivery. Retrain on Engro data and validate with plant engineers before any real use.
