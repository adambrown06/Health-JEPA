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

const GEN_STEPS = [
  "Building patient feature vector…",
  "Computing JEPA embedding…",
  "Mapping into cohort galaxy…",
  "Locating digital twin neighborhood…",
  "Finalizing twin…",
];

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

  const currentStep = chartUploaded ? (stage === "generating" ? 3 : 2) : 1;

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
    <div className="landing-bg flex flex-col items-center min-h-dvh h-dvh max-h-dvh overflow-y-auto overflow-x-hidden px-5 py-6 [scrollbar-gutter:stable]">
      <div className="flex flex-col items-center w-full max-w-md mx-auto flex-1 justify-center gap-5">

        {/* ── Hero ── */}
        <div className="text-center w-full animate-fade-up">
          <div className="flex justify-center mb-4">
            <Image
              src="/compass-logo.png"
              alt=""
              width={320}
              height={320}
              className="w-36 h-36 sm:w-44 sm:h-44 object-contain animate-float"
              priority
            />
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight">
            <span className="text-[#0B3C8C]">Cohort </span>
            <span className="text-brand-compass">Compass</span>
          </h1>
          <p className="mt-1.5 text-sm sm:text-base text-[#6B7280] italic tracking-wide">
            Pointing you towards a healthier tomorrow.
          </p>
        </div>

        {/* ── Progress indicator ── */}
        <div className="flex items-center gap-0 w-full max-w-[220px] animate-fade-up" style={{ animationDelay: "0.1s" }}>
          <StepDot num={1} active={currentStep >= 1} current={currentStep === 1} />
          <StepLine filled={currentStep >= 2} />
          <StepDot num={2} active={currentStep >= 2} current={currentStep === 2} />
          <StepLine filled={currentStep >= 3} />
          <StepDot num={3} active={currentStep >= 3} current={currentStep === 3} />
        </div>

        {/* ── Cards ── */}
        <div className="w-full space-y-3">

          {/* Card 1: Clinical Records */}
          <div className="animate-fade-up" style={{ animationDelay: "0.15s" }}>
            <div className={`card-glass rounded-2xl p-4 shadow-sm transition-all duration-300 ${
              chartUploaded ? "ring-2 ring-[#2EC4C7]/30" : ""
            }`}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#0B3C8C] to-[#1FA3B3] flex items-center justify-center shrink-0">
                  <FileIcon />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-[#0B3C8C]">Clinical Records</p>
                  <p className="text-xs text-[#9AA3AD]">Upload your MyChart export</p>
                </div>
                {chartUploaded && <CheckBadge />}
              </div>

              <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-[#F4F6F8]/80 mb-3">
                <svg className="w-4 h-4 text-[#9AA3AD] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9l-5-5H7a2 2 0 00-2 2v13a2 2 0 002 2z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 3v4a2 2 0 002 2h4" />
                </svg>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-[#5C6773] truncate">mychart_export.pdf</p>
                  <p className="text-[10px] text-[#9AA3AD]">4.2 KB</p>
                </div>
              </div>

              <button
                onClick={handleUploadChart}
                disabled={stage !== "initial"}
                className={`w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  chartUploaded
                    ? "bg-[#E8F7F8] text-[#0E7C91] border border-[#2EC4C7]/40 cursor-default"
                    : stage === "uploading"
                    ? "bg-[#D9DEE3] text-[#9AA3AD] cursor-wait"
                    : "bg-[#0B3C8C] text-white hover:bg-[#09306f] shadow-md hover:shadow-lg hover:-translate-y-0.5"
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

          {/* Card 2: Wearables */}
          <div className="animate-fade-up" style={{ animationDelay: "0.25s" }}>
            <div className={`card-glass rounded-2xl p-4 shadow-sm transition-all duration-300 ${
              wearablesSynced ? "ring-2 ring-[#2EC4C7]/30" : ""
            }`}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#1FA3B3] to-[#2EC4C7] flex items-center justify-center shrink-0">
                  <WatchIcon />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-[#0B3C8C]">Wearable Data</p>
                    <span className="text-[9px] text-[#9AA3AD] uppercase tracking-widest font-medium px-1.5 py-0.5 rounded-full bg-[#F4F6F8] border border-[#D9DEE3]">
                      Optional
                    </span>
                  </div>
                  <p className="text-xs text-[#9AA3AD]">Sync HRV, sleep & activity data</p>
                </div>
                {wearablesSynced && <CheckBadge />}
              </div>

              <div className="flex items-center gap-2 flex-wrap px-3 py-2 rounded-lg bg-[#F4F6F8]/80 mb-3">
                {["Apple Watch", "Oura Ring", "Fitbit", "Garmin", "Whoop"].map((d) => (
                  <span
                    key={d}
                    className="text-[10px] text-[#5C6773] px-2 py-0.5 rounded-full bg-white border border-[#D9DEE3]"
                  >
                    {d}
                  </span>
                ))}
              </div>

              <button
                onClick={handleSyncWearables}
                disabled={syncingWearables || wearablesSynced}
                className={`w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  wearablesSynced
                    ? "bg-[#E8F7F8] text-[#0E7C91] border border-[#2EC4C7]/40 cursor-default"
                    : syncingWearables
                    ? "bg-[#D9DEE3] text-[#9AA3AD] cursor-wait"
                    : "border-2 border-[#D9DEE3] text-[#0B3C8C] bg-white hover:border-[#1FA3B3] hover:shadow-md hover:-translate-y-0.5"
                }`}
              >
                {wearablesSynced
                  ? "Wearables synced"
                  : syncingWearables
                  ? "Syncing wearables…"
                  : "Sync Wearables"}
              </button>
            </div>
          </div>
        </div>

        {/* ── Generate CTA ── */}
        {(chartUploaded || stage === "generating") && (
          <div className="w-full animate-fade-up" style={{ animationDelay: "0.1s" }}>
            {stage !== "generating" ? (
              <div className="text-center">
                <p className="text-[#6B7280] text-xs mb-3">
                  {wearablesSynced
                    ? "Clinical records & wearable data ready."
                    : "Clinical records ready — wearable sync is optional."}
                </p>
                <button
                  onClick={handleGenerate}
                  className="w-full py-3.5 rounded-2xl text-base font-bold text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5 btn-shimmer"
                >
                  Generate Health Digital Twin
                </button>
              </div>
            ) : (
              <div className="card-glass rounded-2xl p-5 text-center">
                <div className="relative w-16 h-16 mx-auto mb-4">
                  <div className="absolute inset-0 rounded-full border-2 border-[#D9DEE3]" />
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-[#2EC4C7] animate-spin-slow" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Image
                      src="/compass-logo.png"
                      alt=""
                      width={40}
                      height={40}
                      className="w-9 h-9 object-contain logo-on-brand"
                    />
                  </div>
                </div>
                <p className="text-base font-semibold text-[#0B3C8C] mb-3">
                  Generating your Digital Twin…
                </p>
                <div className="flex flex-col items-start gap-1.5 text-left max-w-xs mx-auto">
                  {GEN_STEPS.map((label, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-2 text-xs transition-all duration-300 ${
                        i < genStep
                          ? "text-[#0E7C91]"
                          : i === genStep
                          ? "text-[#0B3C8C] font-medium"
                          : "text-[#D9DEE3]"
                      }`}
                    >
                      {i < genStep ? (
                        <svg className="w-3.5 h-3.5 text-[#2EC4C7] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : i === genStep ? (
                        <span className="w-3.5 h-3.5 shrink-0 flex items-center justify-center">
                          <span className="w-2 h-2 rounded-full bg-[#2EC4C7] dot-active" />
                        </span>
                      ) : (
                        <span className="w-3.5 h-3.5 shrink-0 flex items-center justify-center">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#D9DEE3]" />
                        </span>
                      )}
                      {label}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Trust footer ── */}
        <div className="flex items-center justify-center gap-4 pt-1 animate-fade-up" style={{ animationDelay: "0.35s" }}>
          <TrustBadge icon={<LockIcon />} label="End-to-end encrypted" />
          <TrustBadge icon={<ShieldIcon />} label="HIPAA compliant" />
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ── */

function StepDot({ num, active, current }: { num: number; active: boolean; current: boolean }) {
  return (
    <div className="relative flex items-center justify-center">
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold transition-all duration-300 ${
          active
            ? "bg-gradient-to-br from-[#2EC4C7] to-[#0E7C91] text-white shadow-md"
            : "bg-white border-2 border-[#D9DEE3] text-[#9AA3AD]"
        } ${current ? "dot-active" : ""}`}
      >
        {num}
      </div>
    </div>
  );
}

function StepLine({ filled }: { filled: boolean }) {
  return (
    <div className="flex-1 h-0.5 mx-1 rounded-full transition-all duration-500">
      <div
        className={`h-full rounded-full transition-all duration-500 ${
          filled
            ? "bg-gradient-to-r from-[#2EC4C7] to-[#0E7C91]"
            : "bg-[#D9DEE3]"
        }`}
      />
    </div>
  );
}

function TrustBadge({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] text-[#9AA3AD]">
      {icon}
      <span>{label}</span>
    </div>
  );
}

function FileIcon() {
  return (
    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9l-5-5H7a2 2 0 00-2 2v13a2 2 0 002 2z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 3v4a2 2 0 002 2h4" />
    </svg>
  );
}

function WatchIcon() {
  return (
    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <rect x="6" y="4" width="12" height="16" rx="3" strokeWidth={1.5} strokeLinecap="round" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 1h6M9 23h6M12 8v4l2 2" />
    </svg>
  );
}

function CheckBadge() {
  return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[#2EC4C7] to-[#0E7C91] flex items-center justify-center shrink-0 shadow-sm">
      <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
      </svg>
    </div>
  );
}

function LockIcon() {
  return (
    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <rect x="5" y="11" width="14" height="10" rx="2" strokeWidth={1.5} />
      <path strokeWidth={1.5} d="M8 11V7a4 4 0 118 0v4" strokeLinecap="round" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4" />
    </svg>
  );
}

function tick(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
