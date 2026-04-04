"use client";

import { useState } from "react";
import Image from "next/image";
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
    <div className="flex flex-col items-center justify-center min-h-dvh h-dvh max-h-dvh overflow-y-auto overflow-x-hidden gap-3 px-4 py-3 max-w-xl mx-auto w-full bg-[#F4F6F8] [scrollbar-gutter:stable]">
      {/* Brand: blend removes white matte from PNG */}
      <div className="text-center w-full shrink-0">
        <div className="mb-0 flex justify-center">
          <div className="rounded-xl bg-[#F4F6F8] p-1 inline-flex">
            <Image
              src="/brand-compass-mark.png"
              alt="Cohort Compass — Pointing you towards a healthier tomorrow."
              width={640}
              height={320}
              className="logo-on-brand w-[min(100%,560px,96vw)] h-auto max-h-[min(340px,40vh)] object-contain"
              priority
            />
          </div>
        </div>
        <p className="-mt-1.5 text-xl sm:text-2xl font-bold text-[#0B3C8C] tracking-tight">
          Your Health Universe
        </p>
      </div>

      {/* Step 1 */}
      <div className="w-full min-h-0 shrink">
        <SectionLabel step={1} label="Clinical Records" />
        <div className="border border-[#D9DEE3] rounded-xl p-3 bg-white shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <FileIcon />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-[#0B3C8C] font-medium truncate">
                mychart_export.pdf
              </p>
              <p className="text-xs text-[#9AA3AD]">4.2 KB — MyChart clinical history</p>
            </div>
            {chartUploaded && <CheckBadge />}
          </div>
          <button
            onClick={handleUploadChart}
            disabled={stage !== "initial"}
            className={`w-full py-2 rounded-lg text-sm font-semibold transition ${
              chartUploaded
                ? "bg-[#E8F7F8] text-[#0E7C91] border border-[#2EC4C7]/40 cursor-default"
                : stage === "uploading"
                ? "bg-[#D9DEE3] text-[#9AA3AD] cursor-wait"
                : "bg-[#0B3C8C] text-white hover:bg-[#09306f] shadow-sm"
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

      {/* Step 2 */}
      <div className="w-full min-h-0 shrink">
        <SectionLabel step={2} label="Wearable Data" optional />
        <div className="border border-[#D9DEE3] rounded-xl p-3 bg-white shadow-sm">
          <p className="text-[11px] text-[#6B7280] mb-2 leading-snug">
            Optional: sync wearables for HRV, sleep, and activity in your twin.
          </p>
          <div className="flex items-center gap-2 mb-2">
            <WatchIcon />
            <span className="text-xs text-[#5C6773]">
              Apple Watch, Oura Ring, Fitbit, Garmin, Whoop
            </span>
          </div>
          <button
            onClick={handleSyncWearables}
            disabled={syncingWearables || wearablesSynced}
            className={`w-full py-2 rounded-lg text-sm font-semibold transition ${
              wearablesSynced
                ? "bg-[#E8F7F8] text-[#0E7C91] border border-[#2EC4C7]/40 cursor-default"
                : syncingWearables
                ? "bg-[#D9DEE3] text-[#9AA3AD] cursor-wait"
                : "border-2 border-[#9AA3AD] text-[#0B3C8C] bg-white hover:bg-[#F4F6F8]"
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

      {(chartUploaded || stage === "generating") && (
        <div className="w-full text-center shrink-0 min-h-0">
          <div className="border-t border-[#D9DEE3] pt-3 mt-1 w-full">
            {stage !== "generating" ? (
              <>
                <p className="text-[#6B7280] text-[11px] mb-2 leading-tight">
                  {wearablesSynced
                    ? "Records + wearables ready."
                    : "Records ready — optional wearables above."}
                </p>
                <button
                  onClick={handleGenerate}
                  className="w-full py-3 rounded-xl text-base font-bold text-white shadow-md transition hover:opacity-95 bg-gradient-to-r from-[#2EC4C7] via-[#1FA3B3] to-[#0E7C91]"
                >
                  Generate Health Digital Twin
                </button>
              </>
            ) : (
              <div className="flex flex-col items-center gap-1.5 max-h-[38vh] overflow-y-auto">
                <p className="text-base font-semibold text-[#0B3C8C] shrink-0">
                  Generating your Digital Twin…
                </p>
                <div className="w-full flex flex-col items-start gap-0.5 text-left max-w-sm mx-auto text-[11px]">
                  {GEN_STEPS.map((label, i) => (
                    <div
                      key={i}
                      className={`transition-all duration-300 ${
                        i < genStep
                          ? "text-[#0E7C91]"
                          : i === genStep
                          ? "text-[#5C6773] animate-pulse"
                          : "text-[#D9DEE3]"
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
    <div className="flex items-center gap-2 mb-1">
      <span className="text-xs text-[#9AA3AD] font-mono">{step}.</span>
      <span className="text-sm font-semibold text-[#0B3C8C]">{label}</span>
      {optional && (
        <span className="text-[10px] text-[#9AA3AD] uppercase tracking-wider">
          optional
        </span>
      )}
    </div>
  );
}

function FileIcon() {
  return (
    <svg
      className="w-8 h-8 text-[#9AA3AD] shrink-0"
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
      className="w-4 h-4 text-[#1FA3B3] shrink-0"
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
    <div className="w-6 h-6 rounded-full bg-[#E8F7F8] border border-[#2EC4C7]/50 flex items-center justify-center shrink-0">
      <svg
        className="w-3.5 h-3.5 text-[#0E7C91]"
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
