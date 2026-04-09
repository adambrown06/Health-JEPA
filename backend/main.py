"""
Cohort Compass — FastAPI application.

Serves both the existing mock/demo endpoints AND the new production
async inference pipeline.
"""

import asyncio
import random
import uuid
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from schemas.inference import (
    AnalysisResult,
    InferenceJobAck,
    InferenceJobRequest,
    InferenceJobStatus,
    JobStatus,
)

app = FastAPI(
    title="Cohort Compass — Causal AI Healthcare Platform",
    version="0.2.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Production async inference endpoints
# ======================================================================

@app.post(
    f"{settings.api_v1_prefix}/analyze",
    response_model=InferenceJobAck,
    status_code=202,
    tags=["inference"],
)
async def analyze(request: InferenceJobRequest):
    """Accept clinical data, enqueue a Celery task, return immediately."""
    from worker.tasks import run_full_analysis

    job_id = str(uuid.uuid4())
    payload = request.model_dump(mode="json")
    payload["callback_job_id"] = job_id

    run_full_analysis.apply_async(
        args=[payload],
        task_id=job_id,
        queue="inference",
    )

    return InferenceJobAck(
        job_id=job_id,
        status="queued",
        poll_url=f"{settings.api_v1_prefix}/jobs/{job_id}",
    )


@app.get(
    f"{settings.api_v1_prefix}/jobs/{{job_id}}",
    response_model=InferenceJobStatus,
    tags=["inference"],
)
async def get_job_status(job_id: str):
    """Poll for the status / result of an analysis job."""
    from celery.result import AsyncResult
    from worker.celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)

    if result.state == "PENDING":
        return InferenceJobStatus(job_id=job_id, status=JobStatus.QUEUED)

    if result.state == "RUNNING":
        meta = result.info or {}
        return InferenceJobStatus(
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress=meta.get("progress", 0.0),
        )

    if result.state == "SUCCESS":
        return InferenceJobStatus(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result=AnalysisResult(**result.result),
        )

    # FAILURE or REVOKED
    return InferenceJobStatus(
        job_id=job_id,
        status=JobStatus.FAILED,
        error=str(result.info) if result.info else "Unknown error",
    )


# ======================================================================
# Existing demo / mock endpoints (preserved for frontend compatibility)
# ======================================================================

DELAY = 1.5

CLUSTER_META = {
    1: {"name": "Metabolic Risk",      "center": (0.0,  0.0,  0.0),   "spread": 3.5, "count": 400},
    0: {"name": "General Population",  "center": (16.0, 0.0,  0.0),   "spread": 2.2, "count": 150},
    2: {"name": "Cardiac Risk",        "center": (-12.0, 0.0, 12.0),  "spread": 2.2, "count": 150},
    3: {"name": "Respiratory",         "center": (-12.0, 0.0, -12.0), "spread": 2.2, "count": 150},
    4: {"name": "Healthy / Active",    "center": (0.0,  0.0,  -16.0), "spread": 2.2, "count": 150},
}

FIRST_NAMES = [
    "James", "Maria", "Robert", "Sarah", "David", "Linda", "Michael", "Jennifer",
    "William", "Patricia", "Richard", "Elizabeth", "Joseph", "Barbara", "Thomas",
    "Susan", "Charles", "Jessica", "Daniel", "Karen", "Matthew", "Nancy", "Anthony",
    "Lisa", "Mark", "Betty", "Steven", "Margaret", "Paul", "Sandra", "Andrew", "Ashley",
    "Joshua", "Dorothy", "Kenneth", "Kimberly", "Kevin", "Emily", "Brian", "Donna",
]

USER_CLUSTER_ID = 1


def _generate_galaxy():
    rng = random.Random(42)
    points = []
    idx = 0
    for cluster_id, meta in CLUSTER_META.items():
        cx, cy, cz = meta["center"]
        spread = meta["spread"]
        for _ in range(meta["count"]):
            sex = rng.choice(["Male", "Female"])
            age = rng.randint(28, 72)
            name = rng.choice(FIRST_NAMES)
            outcome = "positive" if rng.random() < 0.55 else "negative"
            points.append({
                "id": f"pt-{idx:04d}",
                "label": f"{name}, {sex}, {age}",
                "outcome_type": outcome,
                "x": round(cx + rng.gauss(0, spread), 3),
                "y": round(cy + rng.gauss(0, spread * 0.4), 3),
                "z": round(cz + rng.gauss(0, spread), 3),
                "cluster_id": cluster_id,
            })
            idx += 1
    return points


GALAXY_DATA = _generate_galaxy()

DIGITAL_TWINS = [
    {
        "id": "twin-1",
        "label": "Twin 1 — Male, 44",
        "similarity": 0.96,
        "cluster_id": 1,
        "cluster_name": "Metabolic Risk",
        "outcome_type": "positive",
        "coordinate": {"x": 1.2, "y": 0.3, "z": -0.8},
        "outcome": "Reversed pre-diabetes in 14 months via structured Zone 2 cardio program and GLP-1 agonist. HbA1c dropped from 6.2 → 5.4.",
    },
    {
        "id": "twin-2",
        "label": "Twin 2 — Female, 40",
        "similarity": 0.93,
        "cluster_id": 1,
        "cluster_name": "Metabolic Risk",
        "outcome_type": "negative",
        "coordinate": {"x": -0.9, "y": -0.2, "z": 1.5},
        "outcome": "Progressed to Type 2 Diabetes within 18 months. No lifestyle intervention was adopted. HbA1c rose from 6.0 → 7.1.",
    },
    {
        "id": "twin-3",
        "label": "Twin 3 — Male, 46",
        "similarity": 0.91,
        "cluster_id": 1,
        "cluster_name": "Metabolic Risk",
        "outcome_type": "positive",
        "coordinate": {"x": 0.5, "y": 0.1, "z": 0.9},
        "outcome": "Stabilized HbA1c at 5.8 over 24 months through Mediterranean diet and 10k daily steps. BMI decreased from 29.1 → 25.8.",
    },
    {
        "id": "twin-4",
        "label": "Twin 4 — Female, 39",
        "similarity": 0.85,
        "cluster_id": 2,
        "cluster_name": "Cardiac Risk",
        "outcome_type": "negative",
        "coordinate": {"x": -10.5, "y": 0.4, "z": 10.8},
        "outcome": "Developed cardiovascular complications at 22 months. Metabolic syndrome progressed to include hypertension and elevated cardiac risk markers.",
    },
    {
        "id": "twin-5",
        "label": "Twin 5 — Male, 43",
        "similarity": 0.82,
        "cluster_id": 0,
        "cluster_name": "General Population",
        "outcome_type": "positive",
        "coordinate": {"x": 13.5, "y": -0.3, "z": 0.6},
        "outcome": "Complete reversal in 10 months. Combined metformin, CGM-guided nutrition, and resistance training moved them out of the risk cluster entirely. HbA1c 6.3 → 5.1.",
    },
]

INTERVENTIONS = [
    {
        "id": "int-1",
        "title": "Start GLP-1 Receptor Agonist",
        "description": "Initiate semaglutide 0.25mg weekly, titrating to 1.0mg. Projected HbA1c reduction: 1.2 points over 6 months.",
        "confidence": 0.92,
        "target_coordinate": {"x": 7.0, "y": 0.0, "z": 0.0},
    },
    {
        "id": "int-2",
        "title": "Increase Zone 2 Cardio to 180 min/week",
        "description": "Structured aerobic program targeting 60-70% max HR. Improves insulin sensitivity and mitochondrial density.",
        "confidence": 0.88,
        "target_coordinate": {"x": 0.0, "y": 0.0, "z": -7.0},
    },
    {
        "id": "int-3",
        "title": "Mediterranean Diet + CGM Monitoring",
        "description": "Adopt Mediterranean dietary pattern with continuous glucose monitoring feedback loops. Targets post-prandial glucose spikes.",
        "confidence": 0.85,
        "target_coordinate": {"x": 4.0, "y": 0.0, "z": -4.0},
    },
]

WEARABLE_DATA = {
    "source": "Apple Watch Series 9 + Oura Ring Gen 3",
    "sync_window": "Last 30 days",
    "metrics": {
        "resting_heart_rate": {"value": 72, "unit": "bpm", "trend": "stable", "series": [74, 73, 72, 73, 71, 72, 72, 71, 70, 72, 73, 72, 71, 72]},
        "hrv": {"value": 34, "unit": "ms (RMSSD)", "trend": "declining", "flag": "below_optimal", "series": [38, 37, 36, 35, 34, 33, 35, 34, 32, 34, 33, 34, 35, 34]},
        "daily_steps": {"value": 5842, "unit": "steps/day avg", "trend": "flat", "flag": "below_target", "series": [6200, 5100, 4800, 6500, 5900, 5200, 6100, 5400, 5800, 6300, 5700, 5500, 6000, 5900]},
        "active_zone_minutes": {"value": 68, "unit": "min/week", "trend": "flat", "flag": "below_target", "series": [72, 65, 70, 60, 75, 68, 64, 70, 66, 68, 72, 65, 70, 68]},
        "sleep_duration": {"value": 6.3, "unit": "hrs/night avg", "trend": "declining", "flag": "below_optimal", "series": [6.5, 6.2, 6.0, 6.8, 6.1, 6.4, 6.2, 5.9, 6.5, 6.3, 6.1, 6.4, 6.2, 6.3]},
        "deep_sleep": {"value": 48, "unit": "min/night avg", "trend": "declining", "flag": "below_optimal", "series": [52, 50, 48, 46, 49, 47, 45, 50, 48, 46, 49, 47, 48, 48]},
        "spo2": {"value": 96.2, "unit": "%", "trend": "stable", "series": [96.5, 96.3, 96.1, 96.4, 96.0, 96.2, 96.3, 96.1, 96.2, 96.4, 96.1, 96.3, 96.2, 96.2]},
        "body_temperature_deviation": {"value": 0.0, "unit": "°C from baseline", "trend": "stable", "series": [0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, -0.1, 0.1, 0.0, 0.0]},
        "respiratory_rate": {"value": 15.8, "unit": "breaths/min", "trend": "stable", "series": [15.5, 15.9, 16.0, 15.7, 15.8, 15.6, 16.1, 15.8, 15.7, 15.9, 15.8, 15.6, 15.9, 15.8]},
    },
    "insights": [
        "HRV trending 22% below age-adjusted median — consistent with chronic low-grade sympathetic activation.",
        "Daily step count averages 5,842 — well below the 8,000+ threshold associated with metabolic benefit.",
        "Active Zone Minutes at 68 min/week — AHA recommends 150 min/week moderate intensity.",
        "Deep sleep averaging 48 min/night — below the 60-90 min target for optimal glucose regulation.",
        "Sleep duration 6.3 hrs — below the 7-9 hr recommendation; correlated with insulin resistance risk.",
    ],
}


@app.post("/api/v1/wearables", tags=["demo"])
async def wearables():
    await asyncio.sleep(DELAY)
    return WEARABLE_DATA


@app.post("/api/v1/ingest", tags=["demo"])
async def ingest(file: UploadFile = File(None)):
    await asyncio.sleep(DELAY)
    return {
        "status": "processed",
        "patient": {
            "age": 42, "sex": "Male", "bmi": 28.4, "hba1c": 6.1,
            "fasting_glucose": 112, "blood_pressure": "134/86",
            "ldl": 142, "triglycerides": 198, "risk_label": "Pre-Diabetic",
            "summary": (
                "42-year-old male presenting with elevated BMI (28.4), HbA1c in "
                "pre-diabetic range (6.1%), and borderline hypertension. Lipid panel "
                "shows elevated LDL and triglycerides consistent with metabolic "
                "syndrome risk."
            ),
        },
    }


@app.get("/api/v1/galaxy", tags=["demo"])
async def galaxy():
    await asyncio.sleep(DELAY)
    return {
        "points": GALAXY_DATA,
        "clusters": {str(k): v["name"] for k, v in CLUSTER_META.items()},
    }


@app.post("/api/v1/embedding", tags=["demo"])
async def embedding():
    await asyncio.sleep(DELAY)
    return {
        "coordinate": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cluster_id": USER_CLUSTER_ID,
        "cluster_name": "Metabolic Risk",
        "embedding_dim": 3,
    }


@app.get("/api/v1/neighbors", tags=["demo"])
async def neighbors():
    await asyncio.sleep(DELAY)
    return {"twins": DIGITAL_TWINS}


@app.post("/api/v1/interventions", tags=["demo"])
async def interventions():
    await asyncio.sleep(DELAY)
    return {"interventions": INTERVENTIONS}
