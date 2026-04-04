"use client";

import { useAppStore } from "@/lib/store";
import { fetchInterventions } from "@/lib/api";
import type { WearableMetric } from "@/lib/types";

const CLUSTER_COLORS: Record<number, string> = {
  0: "#ef4444", // red — highest risk
  1: "#f97316", // orange
  2: "#eab308", // yellow
  3: "#84cc16", // lime
  4: "#22c55e", // green — lowest risk
};

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-[#D9DEE3]">
      <span className="text-[#6B7280] text-sm">{label}</span>
      <span className="text-[#0B3C8C] text-sm font-medium">{value}</span>
    </div>
  );
}

function BaselineStats() {
  const patient = useAppStore((s) => s.patient);
  const userCluster = useAppStore((s) => s.userCluster);

  if (!patient) return null;

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Patient Baseline</h3>
      <div className="bg-white border border-[#D9DEE3] rounded-lg p-3 mb-3 shadow-sm">
        <span className="text-xs uppercase tracking-wide text-red-600 font-semibold">
          {patient.risk_label}
        </span>
        {userCluster && (
          <span className="text-xs uppercase tracking-wide text-[#6B7280] ml-2">
            — {userCluster} cluster
          </span>
        )}
      </div>
      <StatRow label="Age" value={patient.age} />
      <StatRow label="Sex" value={patient.sex} />
      <StatRow label="BMI" value={patient.bmi} />
      <StatRow label="HbA1c" value={`${patient.hba1c}%`} />
      <StatRow label="Fasting Glucose" value={`${patient.fasting_glucose} mg/dL`} />
      <StatRow label="Blood Pressure" value={patient.blood_pressure} />
      <StatRow label="LDL" value={`${patient.ldl} mg/dL`} />
      <StatRow label="Triglycerides" value={`${patient.triglycerides} mg/dL`} />
      <p className="text-xs text-[#6B7280] mt-3 leading-relaxed">{patient.summary}</p>
    </div>
  );
}

const METRIC_LABELS: Record<string, string> = {
  resting_heart_rate: "Resting Heart Rate",
  hrv: "Heart Rate Variability",
  daily_steps: "Daily Steps",
  active_zone_minutes: "Active Zone Minutes",
  sleep_duration: "Sleep Duration",
  deep_sleep: "Deep Sleep",
  spo2: "SpO2",
  body_temperature_deviation: "Body Temp Deviation",
  respiratory_rate: "Respiratory Rate",
};

function TrendBadge({ trend, flag }: { trend: string; flag?: string }) {
  const color =
    flag === "below_optimal" || flag === "below_target"
      ? "text-amber-700 bg-amber-50 border-amber-200"
      : trend === "declining"
      ? "text-red-700 bg-red-50 border-red-200"
      : trend === "stable"
      ? "text-[#5C6773] bg-[#F4F6F8] border-[#D9DEE3]"
      : "text-emerald-700 bg-emerald-50 border-emerald-200";

  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded border font-medium uppercase tracking-wide ${color}`}
    >
      {flag ? flag.replace(/_/g, " ") : trend}
    </span>
  );
}

function MiniSparkline({ series }: { series: number[] }) {
  if (series.length < 2) return null;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const range = max - min || 1;
  const h = 20;
  const w = 80;
  const step = w / (series.length - 1);
  const points = series
    .map((v, i) => `${i * step},${h - ((v - min) / range) * h}`)
    .join(" ");

  return (
    <svg width={w} height={h} className="shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className="text-[#1FA3B3]"
      />
    </svg>
  );
}

function WearableMetricCard({ name, metric }: { name: string; metric: WearableMetric }) {
  return (
    <div className="bg-white border border-[#D9DEE3] rounded-lg p-3 shadow-sm">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="text-xs text-[#5C6773]">{METRIC_LABELS[name] ?? name}</span>
        <TrendBadge trend={metric.trend} flag={metric.flag} />
      </div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <span className="text-lg font-semibold text-[#0B3C8C]">{metric.value}</span>
          <span className="text-xs text-[#9AA3AD] ml-1">{metric.unit}</span>
        </div>
        <MiniSparkline series={metric.series} />
      </div>
    </div>
  );
}

function WearablesView() {
  const wearables = useAppStore((s) => s.wearables);
  if (!wearables) {
    return (
      <div>
        <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Wearable Data</h3>
        <p className="text-xs text-[#6B7280]">No wearable data synced.</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-1 text-[#0B3C8C]">Wearable Data</h3>
      <p className="text-xs text-[#6B7280] mb-1">{wearables.source}</p>
      <p className="text-xs text-[#9AA3AD] mb-3">{wearables.sync_window}</p>
      <div className="space-y-2 mb-4">
        {Object.entries(wearables.metrics).map(([key, metric]) => (
          <WearableMetricCard key={key} name={key} metric={metric} />
        ))}
      </div>
      {wearables.insights.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-[#5C6773] uppercase tracking-wide mb-2">
            Insights
          </h4>
          <div className="space-y-2">
            {wearables.insights.map((insight, i) => (
              <div
                key={i}
                className="text-xs text-[#6B7280] leading-relaxed bg-white border border-[#D9DEE3] rounded-lg p-2.5 shadow-sm"
              >
                {insight}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PatientDetail() {
  const selectedPoint = useAppStore((s) => s.selectedPoint);
  const clusterNames = useAppStore((s) => s.clusterNames);

  if (!selectedPoint) {
    return (
      <div>
        <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Patient Detail</h3>
        <p className="text-xs text-[#6B7280]">Click a patient dot in the 3D map.</p>
      </div>
    );
  }

  const cName = clusterNames[String(selectedPoint.cluster_id)] ?? "Unknown";

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Patient Detail</h3>
      <div className="bg-white border border-[#D9DEE3] rounded-lg p-4 space-y-3 shadow-sm">
        <div>
          <div className="text-sm font-medium text-[#0B3C8C]">{selectedPoint.label}</div>
          <div className="text-xs text-[#9AA3AD] mt-0.5">{selectedPoint.id}</div>
        </div>
        <div className="flex items-center gap-2">
          <ClusterBadge clusterId={selectedPoint.cluster_id} name={cName} />
          <OutcomeBadge type={selectedPoint.outcome_type} />
        </div>
        <div className="text-xs text-[#6B7280] leading-relaxed">
          Coordinates: ({selectedPoint.x.toFixed(2)}, {selectedPoint.y.toFixed(2)}, {selectedPoint.z.toFixed(2)})
        </div>
      </div>
    </div>
  );
}

function ClusterBadge({ clusterId, name }: { clusterId: number; name: string }) {
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded border font-medium"
      style={{
        color: CLUSTER_COLORS[clusterId] ?? "#a5b4fc",
        borderColor: CLUSTER_COLORS[clusterId] ?? "#a5b4fc",
        backgroundColor: `${CLUSTER_COLORS[clusterId] ?? "#a5b4fc"}18`,
      }}
    >
      {name}
    </span>
  );
}

function OutcomeBadge({ type }: { type: "positive" | "negative" }) {
  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded font-medium border ${
        type === "positive"
          ? "text-emerald-800 bg-emerald-50 border-emerald-200"
          : "text-red-800 bg-red-50 border-red-200"
      }`}
    >
      {type === "positive" ? "Positive" : "Negative"}
    </span>
  );
}

function DigitalTwins() {
  const twins = useAppStore((s) => s.twins);
  const selectedTwin = useAppStore((s) => s.selectedTwin);
  const selectTwin = useAppStore((s) => s.selectTwin);

  if (twins.length === 0) {
    return (
      <div>
        <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Digital Twins</h3>
        <p className="text-xs text-[#6B7280]">Loading twins…</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Digital Twins</h3>
      <p className="text-xs text-[#6B7280] mb-3">
        Click a twin to locate them in the 3D map. Ranked by embedding similarity.
      </p>
      <div className="space-y-2">
        {twins.map((t) => {
          const isActive = selectedTwin?.id === t.id;
          return (
            <button
              key={t.id}
              onClick={() => selectTwin(isActive ? null : t)}
              className={`w-full text-left p-3 rounded-lg border transition shadow-sm ${
                isActive
                  ? "border-[#0B3C8C] bg-[#E8EEF8]"
                  : "border-[#D9DEE3] bg-white hover:border-[#9AA3AD]"
              }`}
            >
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm font-medium text-[#0B3C8C]">{t.label}</span>
                <span className="text-xs text-[#9AA3AD]">
                  {(t.similarity * 100).toFixed(0)}% match
                </span>
              </div>
              <div className="flex items-center gap-1.5 mb-2">
                <ClusterBadge clusterId={t.cluster_id} name={t.cluster_name} />
                <OutcomeBadge type={t.outcome_type} />
              </div>
              <p
                className={`text-xs leading-relaxed ${
                  isActive ? "text-[#5C6773]" : "text-[#6B7280]"
                }`}
              >
                {t.outcome}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function InterventionCards() {
  const interventions = useAppStore((s) => s.interventions);
  const active = useAppStore((s) => s.activeIntervention);
  const setActive = useAppStore((s) => s.setActiveIntervention);

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-[#0B3C8C]">Recommended Interventions</h3>
      <p className="text-xs text-[#6B7280] mb-3">
        Click a card to visualize the projected trajectory in the 3D map.
      </p>
      <div className="space-y-2">
        {interventions.map((iv) => (
          <button
            key={iv.id}
            onClick={() => setActive(active?.id === iv.id ? null : iv)}
            onMouseEnter={() => setActive(iv)}
            onMouseLeave={() => {
              if (active?.id === iv.id) setActive(null);
            }}
            className={`w-full text-left p-3 rounded-lg border transition shadow-sm ${
              active?.id === iv.id
                ? "border-[#1FA3B3] bg-[#E8F7F8]"
                : "border-[#D9DEE3] bg-white hover:border-[#9AA3AD]"
            }`}
          >
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm font-medium text-[#0B3C8C]">{iv.title}</span>
              <span className="text-xs text-[#0E7C91] font-semibold">
                {(iv.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-[#6B7280] leading-relaxed">{iv.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function SidePanel() {
  const sidePanelView = useAppStore((s) => s.sidePanelView);
  const setSidePanelView = useAppStore((s) => s.setSidePanelView);
  const interventions = useAppStore((s) => s.interventions);
  const interventionsLoading = useAppStore((s) => s.interventionsLoading);
  const wearables = useAppStore((s) => s.wearables);
  const setInterventions = useAppStore((s) => s.setInterventions);
  const setInterventionsLoading = useAppStore((s) => s.setInterventionsLoading);
  const setActive = useAppStore((s) => s.setActiveIntervention);
  const selectTwin = useAppStore((s) => s.selectTwin);

  const handleLoadInterventions = async () => {
    if (interventions.length > 0) {
      setSidePanelView("interventions");
      return;
    }
    setInterventionsLoading(true);
    const data = await fetchInterventions();
    setInterventions(data.interventions);
    setInterventionsLoading(false);
    setSidePanelView("interventions");
  };

  return (
    <div className="h-full overflow-y-auto p-4 flex flex-col gap-4">
      <div className="flex gap-1 flex-wrap">
        <TabButton
          active={sidePanelView === "stats"}
          onClick={() => {
            setSidePanelView("stats");
            setActive(null);
            selectTwin(null);
          }}
          label="Stats"
        />
        {wearables && (
          <TabButton
            active={sidePanelView === "wearables"}
            onClick={() => {
              setSidePanelView("wearables");
              setActive(null);
              selectTwin(null);
            }}
            label="Wearables"
          />
        )}
        <TabButton
          active={sidePanelView === "twins" || sidePanelView === "drill-down"}
          onClick={() => {
            setSidePanelView("twins");
            setActive(null);
          }}
          label="Twins"
        />
        <TabButton
          active={sidePanelView === "interventions"}
          onClick={handleLoadInterventions}
          label="Interventions"
          loading={interventionsLoading}
        />
      </div>

      {sidePanelView === "stats" && <BaselineStats />}
      {sidePanelView === "wearables" && <WearablesView />}
      {sidePanelView === "patient-detail" && <PatientDetail />}
      {(sidePanelView === "twins" || sidePanelView === "drill-down") && <DigitalTwins />}
      {sidePanelView === "interventions" && <InterventionCards />}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
  loading,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  loading?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border ${
        active
          ? "bg-[#0B3C8C] text-white border-[#0B3C8C]"
          : "bg-white text-[#5C6773] border-[#D9DEE3] hover:border-[#9AA3AD]"
      } ${loading ? "opacity-50 cursor-wait" : ""}`}
    >
      {loading ? "Loading…" : label}
    </button>
  );
}
