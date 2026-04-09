"use client";

import { useMemo } from "react";
import { useAppStore } from "@/lib/store";
import type { DigitalTwin } from "@/lib/types";

const CLUSTER_COLORS: Record<number, string> = {
  0: "#ef4444",
  1: "#f97316",
  2: "#eab308",
  3: "#84cc16",
  4: "#22c55e",
};

const RISK_LABELS: Record<number, string> = {
  0: "Critical Risk",
  1: "High Risk",
  2: "Moderate Risk",
  3: "Low Risk",
  4: "Minimal Risk",
};

export default function ClusterPopup() {
  const focusedClusterId = useAppStore((s) => s.focusedClusterId);
  const clusterNames = useAppStore((s) => s.clusterNames);
  const twins = useAppStore((s) => s.twins);
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const setFocused = useAppStore((s) => s.setFocusedClusterId);

  const clusterTwins = useMemo(() => {
    if (focusedClusterId === null) return [];
    return twins
      .filter((t) => t.cluster_id === focusedClusterId)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3);
  }, [focusedClusterId, twins]);

  const clusterPatientCount = useMemo(() => {
    if (focusedClusterId === null) return 0;
    return galaxyPoints.filter((p) => p.cluster_id === focusedClusterId).length;
  }, [focusedClusterId, galaxyPoints]);

  const negativeRate = useMemo(() => {
    if (focusedClusterId === null) return 0;
    const pts = galaxyPoints.filter((p) => p.cluster_id === focusedClusterId);
    if (pts.length === 0) return 0;
    return Math.round((pts.filter((p) => p.outcome_type === "negative").length / pts.length) * 100);
  }, [focusedClusterId, galaxyPoints]);

  if (focusedClusterId === null) return null;

  const clusterName = clusterNames[String(focusedClusterId)] ?? "Unknown";
  const riskLabel = RISK_LABELS[focusedClusterId] ?? "Unknown";
  const color = CLUSTER_COLORS[focusedClusterId] ?? "#a5b4fc";
  const isPrimary = focusedClusterId === 0;

  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
      <div
        className="pointer-events-auto w-[420px] max-w-[95%] max-h-[85vh] overflow-y-auto rounded-2xl shadow-2xl border animate-fade-up"
        style={{
          background: "rgba(255,255,255,0.96)",
          backdropFilter: "blur(16px)",
          borderColor: color,
          borderWidth: 2,
        }}
      >
        {/* Header */}
        <div className="px-5 pt-5 pb-4 border-b border-[#D9DEE3]">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-xl font-extrabold" style={{ color }}>
                {clusterName}
              </h2>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full"
                  style={{ color, backgroundColor: `${color}18`, border: `1px solid ${color}40` }}
                >
                  {riskLabel}
                </span>
                {isPrimary && (
                  <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-[#0B3C8C]/10 text-[#0B3C8C] border border-[#0B3C8C]/20">
                    Your Cluster
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={() => setFocused(null)}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-[#9AA3AD] hover:text-[#5C6773] hover:bg-[#F4F6F8] transition shrink-0"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Stats row */}
          <div className="flex gap-4 mt-3">
            <MiniStat label="Patients" value={String(clusterPatientCount)} />
            <MiniStat label="Negative Outcomes" value={`${negativeRate}%`} warn={negativeRate > 40} />
            <MiniStat label="Top Match" value={clusterTwins[0] ? `${(clusterTwins[0].similarity * 100).toFixed(0)}%` : "—"} />
          </div>
        </div>

        {/* Twins */}
        <div className="px-5 py-4">
          <h3 className="text-sm font-bold text-[#0B3C8C] mb-3">
            Top Similar Patients
          </h3>

          {clusterTwins.length === 0 ? (
            <p className="text-xs text-[#6B7280] italic">No matched patients in this cluster.</p>
          ) : (
            <div className="space-y-3">
              {clusterTwins.map((twin, i) => (
                <TwinCard key={twin.id} twin={twin} rank={i + 1} color={color} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex-1 min-w-0">
      <div className="text-[10px] text-[#9AA3AD] uppercase tracking-wider font-medium">{label}</div>
      <div className={`text-lg font-bold ${warn ? "text-red-600" : "text-[#0B3C8C]"}`}>{value}</div>
    </div>
  );
}

function TwinCard({ twin, rank, color }: { twin: DigitalTwin; rank: number; color: string }) {
  const isPositive = twin.outcome_type === "positive";

  return (
    <div className="rounded-xl border border-[#D9DEE3] bg-[#F4F6F8]/60 p-4">
      {/* Top row: rank, name, similarity */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0"
            style={{ backgroundColor: color }}
          >
            {rank}
          </div>
          <div>
            <div className="text-sm font-bold text-[#0B3C8C]">{twin.label}</div>
            <div className="text-[10px] text-[#9AA3AD]">{twin.cluster_name}</div>
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-lg font-extrabold text-[#0B3C8C]">
            {(twin.similarity * 100).toFixed(0)}%
          </div>
          <div className="text-[9px] text-[#9AA3AD] uppercase tracking-wider">Match</div>
        </div>
      </div>

      {/* Similarity bar */}
      <div className="w-full h-1.5 rounded-full bg-[#D9DEE3] mb-3">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${twin.similarity * 100}%`, backgroundColor: color }}
        />
      </div>

      {/* Outcome badge + summary */}
      <div className="flex items-center gap-2 mb-1.5">
        <span
          className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
            isPositive
              ? "text-emerald-800 bg-emerald-100 border border-emerald-200"
              : "text-red-800 bg-red-100 border border-red-200"
          }`}
        >
          {isPositive ? "Positive Outcome" : "Negative Outcome"}
        </span>
      </div>
      <p className="text-xs text-[#5C6773] leading-relaxed mb-3">{twin.outcome}</p>

      {/* Lifestyle changes — highlighted section */}
      {twin.lifestyle_changes && (
        <div
          className={`rounded-lg p-3 mb-3 border ${
            isPositive
              ? "bg-emerald-50/80 border-emerald-200/60"
              : "bg-red-50/80 border-red-200/60"
          }`}
        >
          <div className="flex items-center gap-1.5 mb-1.5">
            <svg className={`w-3.5 h-3.5 shrink-0 ${isPositive ? "text-emerald-600" : "text-red-500"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span className={`text-[11px] font-bold uppercase tracking-wider ${isPositive ? "text-emerald-700" : "text-red-700"}`}>
              {isPositive ? "What They Changed" : "What Went Wrong"}
            </span>
          </div>
          <p className={`text-xs leading-relaxed ${isPositive ? "text-emerald-800" : "text-red-800"}`}>
            {twin.lifestyle_changes}
          </p>
        </div>
      )}

      {/* Outcome bullets */}
      {twin.outcome_bullets && twin.outcome_bullets.length > 0 && (
        <div>
          <div className="text-[10px] font-bold text-[#5C6773] uppercase tracking-wider mb-1.5">Key Results</div>
          <ul className="space-y-1">
            {twin.outcome_bullets.map((bullet, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-[#5C6773] leading-relaxed">
                <span
                  className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                    isPositive ? "bg-emerald-400" : "bg-red-400"
                  }`}
                />
                {bullet}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
