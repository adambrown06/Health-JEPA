"use client";

import { useState } from "react";
import { useAppStore } from "@/lib/store";
import {
  ingestFile,
  fetchEmbedding,
  fetchGalaxy,
  fetchWearables,
} from "@/lib/api";

type UploadStage = "initial" | "uploading" | "ready" | "generating";

export default function IngestionView() {
  const setPhase = useAppStore((s) => s.setPhase);
  const setPatient = useAppStore((s) => s.setPatient);
  const setUserCoord = useAppStore((s) => s.setUserCoord);
  const setUserCluster = useAppStore((s) => s.setUserCluster);
  const setGalaxyPoints = useAppStore((s) => s.setGalaxyPoints);
  const setClusterNames = useAppStore((s) => s.setClusterNames);
  const setWearables = useAppStore((s) => s.setWearables);
  const wearablesSynced = useAppStore((s) => s.wearablesSynced);
  const setWearablesSynced = useAppStore((s) => s.setWearablesSynced);

  const [stage, setStage] = useState<UploadStage>("initial");
  const [chartUploaded, setChartUploaded] = useState(false);
  const [syncingWearables, setSyncingWearables] = useState(false);
  const [genStep, setGenStep] = useState(-1);

  const GEN_STEPS = [
    "Building patient feature vector…",
    "Computing JEPA embedding…",
    "Mapping into cohort galaxy…",
    "Locating digital twin neighborhood…",
    "Finalizing twin…",
  ];

  // --- Upload MyChart ---
  const handleUploadChart = async () => {
    setStage("uploading");
    try {
      const res = await ingestFile();
      setPatient(res.patient);
      setChartUploaded(true);
      setStage("ready");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Upload failed");
      setStage("initial");
    }
  };

  // --- Sync Wearables ---
  const handleSyncWearables = async () => {
    setSyncingWearables(true);
    try {
      const data = await fetchWearables();
      setWearables(data);
      setWearablesSynced(true);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Sync failed");
    } finally {
      setSyncingWearables(false);
    }
  };

  // --- Generate Digital Twin ---
  const handleGenerate = async () => {
    setStage("generating");
    try {
      setGenStep(0);
      const embeddingPromise = fetchEmbedding();
      await tick(700);
      setGenStep(1);
      const galaxyPromise = fetchGalaxy();
      await tick(600);
      setGenStep(2);

      const [embeddingRes, galaxyRes] = await Promise.all([
        embeddingPromise,
        galaxyPromise,
      ]);

      setGenStep(3);
      setUserCoord(embeddingRes.coordinate);
      setUserCluster(embeddingRes.cluster_name);
      setGalaxyPoints(galaxyRes.points);
      setClusterNames(galaxyRes.clusters);

      await tick(500);
      setGenStep(4);
      await tick(600);
      setPhase("dashboard");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Generation failed");
      setStage("ready");
      setGenStep(-1);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8 p-8 max-w-xl mx-auto">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight mb-2">
          Universal Patient Embedding
        </h1>
        <p className="text-neutral-400 text-sm">
          Upload your clinical records and sync wearables to generate your
          Health Digital Twin in the All of Us cohort.
        </p>
      </div>

      {/* Step 1: Upload MyChart */}
      <div className="w-full">
        <SectionLabel step={1} label="Clinical Records" />
        <div className="border border-neutral-800 rounded-lg p-4 bg-neutral-950">
          <div className="flex items-center gap-3 mb-3">
            <FileIcon />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-neutral-200 truncate">mychart_export.pdf</p>
              <p className="text-xs text-neutral-600">4.2 KB — MyChart clinical history</p>
            </div>
            {chartUploaded && <CheckBadge />}
          </div>
          <button
            onClick={handleUploadChart}
            disabled={stage !== "initial"}
            className={`w-full py-2.5 rounded text-sm font-medium transition ${
              chartUploaded
                ? "bg-green-900/30 text-green-400 border border-green-800 cursor-default"
                : stage === "uploading"
                ? "bg-neutral-800 text-neutral-500 cursor-wait"
                : "bg-white text-black hover:bg-neutral-200"
            }`}
          >
            {chartUploaded
              ? "Clinical records uploaded"
              : stage === "uploading"
              ? "Uploading…"
              : "Upload MyChart"}
          </button>
        </div>
      </div>

      {/* Step 2: Sync Wearables */}
      <div className="w-full">
        <SectionLabel step={2} label="Wearable Data" optional />
        <div className="border border-neutral-800 rounded-lg p-4 bg-neutral-950">
          <p className="text-xs text-neutral-500 mb-3 leading-relaxed">
            For a more accurate digital twin, sync your wearables to include
            day-to-day biometric data — heart rate variability, sleep quality,
            activity levels, and more.
          </p>
          <div className="flex items-center gap-2 mb-3">
            <WatchIcon />
            <span className="text-xs text-neutral-400">
              Apple Watch, Oura Ring, Fitbit, Garmin, Whoop
            </span>
          </div>
          <button
            onClick={handleSyncWearables}
            disabled={syncingWearables || wearablesSynced}
            className={`w-full py-2.5 rounded text-sm font-medium transition ${
              wearablesSynced
                ? "bg-green-900/30 text-green-400 border border-green-800 cursor-default"
                : syncingWearables
                ? "bg-neutral-800 text-neutral-500 cursor-wait"
                : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700 border border-neutral-700"
            }`}
          >
            {wearablesSynced
              ? "Wearables synced — Apple Watch + Oura Ring"
              : syncingWearables
              ? "Syncing wearables…"
              : "Sync Wearables"}
          </button>
        </div>
      </div>

      {/* Step 3: Generate Digital Twin (appears after chart is uploaded) */}
      {(chartUploaded || stage === "generating") && (
        <div className="w-full text-center">
          <div className="border-t border-neutral-800 pt-6 mt-2">
            {stage !== "generating" ? (
              <>
                <p className="text-neutral-500 text-xs mb-4">
                  {wearablesSynced
                    ? "Clinical records + wearables ready. Full-fidelity twin available."
                    : "Clinical records ready. Sync wearables above for higher accuracy."}
                </p>
                <button
                  onClick={handleGenerate}
                  className="w-full py-4 rounded-lg text-lg font-bold bg-white text-black hover:bg-neutral-200 transition"
                >
                  Generate Health Digital Twin
                </button>
              </>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <p className="text-lg font-semibold text-neutral-200">
                  Generating your Digital Twin…
                </p>
                <div className="w-full flex flex-col items-start gap-1.5 mt-2">
                  {GEN_STEPS.map((label, i) => (
                    <div
                      key={i}
                      className={`text-xs transition-all duration-300 ${
                        i < genStep
                          ? "text-green-500"
                          : i === genStep
                          ? "text-neutral-300 animate-pulse"
                          : "text-neutral-700"
                      }`}
                    >
                      {i < genStep ? "✓" : i === genStep ? "›" : "·"} {label}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// --- Small helper components ---

function SectionLabel({
  step,
  label,
  optional,
}: {
  step: number;
  label: string;
  optional?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="text-xs text-neutral-600 font-mono">{step}.</span>
      <span className="text-sm font-medium text-neutral-300">{label}</span>
      {optional && (
        <span className="text-[10px] text-neutral-600 uppercase tracking-wider">
          optional
        </span>
      )}
    </div>
  );
}

function FileIcon() {
  return (
    <svg
      className="w-8 h-8 text-neutral-600 shrink-0"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M7 21h10a2 2 0 002-2V9l-5-5H7a2 2 0 00-2 2v13a2 2 0 002 2z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M14 3v4a2 2 0 002 2h4"
      />
    </svg>
  );
}

function WatchIcon() {
  return (
    <svg
      className="w-4 h-4 text-neutral-600 shrink-0"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <rect
        x="6"
        y="4"
        width="12"
        height="16"
        rx="3"
        strokeWidth={1.5}
        strokeLinecap="round"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 1h6M9 23h6M12 8v4l2 2"
      />
    </svg>
  );
}

function CheckBadge() {
  return (
    <div className="w-6 h-6 rounded-full bg-green-900/50 border border-green-700 flex items-center justify-center shrink-0">
      <svg
        className="w-3.5 h-3.5 text-green-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2.5}
          d="M5 13l4 4L19 7"
        />
      </svg>
    </div>
  );
}

function tick(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
