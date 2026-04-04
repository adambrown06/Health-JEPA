"use client";

import { useAppStore } from "@/lib/store";
import { fetchInterventions } from "@/lib/api";
import type { WearableMetric } from "@/lib/types";

const CLUSTER_COLORS: Record<number, string> = {
  0: "#6b7280",
  1: "#ef4444",
  2: "#f59e0b",
  3: "#3b82f6",
  4: "#22c55e",
};

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-neutral-800">
      <span className="text-neutral-400 text-sm">{label}</span>
      <span className="text-neutral-100 text-sm font-medium">{value}</span>
    </div>
  );
}

// --- Baseline Stats ---

function BaselineStats() {
  const patient = useAppStore((s) => s.patient);
  const userCluster = useAppStore((s) => s.userCluster);

  if (!patient) return null;

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3">Patient Baseline</h3>
      <div className="bg-neutral-900 rounded p-3 mb-3">
        <span className="text-xs uppercase tracking-wide text-red-400 font-semibold">
          {patient.risk_label}
        </span>
        {userCluster && (
          <span className="text-xs uppercase tracking-wide text-neutral-500 ml-2">
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
      <p className="text-xs text-neutral-500 mt-3 leading-relaxed">
        {patient.summary}
      </p>
    </div>
  );
}

// --- Wearables ---

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
      ? "text-amber-400 bg-amber-400/10 border-amber-800"
      : trend === "declining"
      ? "text-red-400 bg-red-400/10 border-red-800"
      : trend === "stable"
      ? "text-neutral-400 bg-neutral-800 border-neutral-700"
      : "text-green-400 bg-green-400/10 border-green-800";

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
        className="text-neutral-500"
      />
    </svg>
  );
}

function WearableMetricCard({ name, metric }: { name: string; metric: WearableMetric }) {
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded p-3">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="text-xs text-neutral-400">{METRIC_LABELS[name] ?? name}</span>
        <TrendBadge trend={metric.trend} flag={metric.flag} />
      </div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <span className="text-lg font-semibold text-neutral-100">{metric.value}</span>
          <span className="text-xs text-neutral-500 ml-1">{metric.unit}</span>
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
        <h3 className="text-lg font-semibold mb-3">Wearable Data</h3>
        <p className="text-xs text-neutral-500">No wearable data synced.</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-1">Wearable Data</h3>
      <p className="text-xs text-neutral-500 mb-1">{wearables.source}</p>
      <p className="text-xs text-neutral-600 mb-3">{wearables.sync_window}</p>
      <div className="space-y-2 mb-4">
        {Object.entries(wearables.metrics).map(([key, metric]) => (
          <WearableMetricCard key={key} name={key} metric={metric} />
        ))}
      </div>
      {wearables.insights.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wide mb-2">
            Insights
          </h4>
          <div className="space-y-2">
            {wearables.insights.map((insight, i) => (
              <div
                key={i}
                className="text-xs text-neutral-400 leading-relaxed bg-neutral-900 border border-neutral-800 rounded p-2.5"
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

// --- Patient Detail (any galaxy point clicked in 3D) ---

function PatientDetail() {
  const selectedPoint = useAppStore((s) => s.selectedPoint);
  const clusterNames = useAppStore((s) => s.clusterNames);

  if (!selectedPoint) {
    return (
      <div>
        <h3 className="text-lg font-semibold mb-3">Patient Detail</h3>
        <p className="text-xs text-neutral-500">Click a patient dot in the 3D map.</p>
      </div>
    );
  }

  const cName = clusterNames[String(selectedPoint.cluster_id)] ?? "Unknown";

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3">Patient Detail</h3>
      <div className="bg-neutral-900 border border-neutral-800 rounded p-4 space-y-3">
        <div>
          <div className="text-sm font-medium text-neutral-100">{selectedPoint.label}</div>
          <div className="text-xs text-neutral-500 mt-0.5">{selectedPoint.id}</div>
        </div>
        <div className="flex items-center gap-2">
          <ClusterBadge clusterId={selectedPoint.cluster_id} name={cName} />
          <OutcomeBadge type={selectedPoint.outcome_type} />
        </div>
        <div className="text-xs text-neutral-400 leading-relaxed">
          Coordinates: ({selectedPoint.x.toFixed(2)}, {selectedPoint.y.toFixed(2)}, {selectedPoint.z.toFixed(2)})
        </div>
      </div>
    </div>
  );
}

// --- Digital Twins (with cluster badges + bidirectional selection) ---

function ClusterBadge({ clusterId, name }: { clusterId: number; name: string }) {
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded border font-medium"
      style={{
        color: CLUSTER_COLORS[clusterId] ?? "#999",
        borderColor: CLUSTER_COLORS[clusterId] ?? "#999",
        backgroundColor: `${CLUSTER_COLORS[clusterId] ?? "#999"}15`,
      }}
    >
      {name}
    </span>
  );
}

function OutcomeBadge({ type }: { type: "positive" | "negative" }) {
  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
        type === "positive"
          ? "text-green-400 bg-green-400/10"
          : "text-red-400 bg-red-400/10"
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
        <h3 className="text-lg font-semibold mb-3">Digital Twins</h3>
        <p className="text-xs text-neutral-500">Loading twins…</p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3">Digital Twins</h3>
      <p className="text-xs text-neutral-500 mb-3">
        Click a twin to locate them in the 3D map. Ranked by embedding similarity.
      </p>
      <div className="space-y-2">
        {twins.map((t) => {
          const isActive = selectedTwin?.id === t.id;
          return (
            <button
              key={t.id}
              onClick={() => selectTwin(isActive ? null : t)}
              className={`w-full text-left p-3 rounded border transition ${
                isActive
                  ? "border-yellow-500 bg-yellow-500/10"
                  : "border-neutral-800 bg-neutral-900 hover:border-neutral-600"
              }`}
            >
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm font-medium">{t.label}</span>
                <span className="text-xs text-neutral-500">
                  {(t.similarity * 100).toFixed(0)}% match
                </span>
              </div>
              <div className="flex items-center gap-1.5 mb-2">
                <ClusterBadge clusterId={t.cluster_id} name={t.cluster_name} />
                <OutcomeBadge type={t.outcome_type} />
              </div>
              {/* Always show outcome summary */}
              <p
                className={`text-xs leading-relaxed ${
                  isActive ? "text-neutral-200" : "text-neutral-500"
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

// --- Intervention Cards ---

function InterventionCards() {
  const interventions = useAppStore((s) => s.interventions);
  const active = useAppStore((s) => s.activeIntervention);
  const setActive = useAppStore((s) => s.setActiveIntervention);

  return (
    <div>
      <h3 className="text-lg font-semibold mb-3">Recommended Interventions</h3>
      <p className="text-xs text-neutral-500 mb-3">
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
            className={`w-full text-left p-3 rounded border transition ${
              active?.id === iv.id
                ? "border-cyan-500 bg-cyan-500/10"
                : "border-neutral-800 bg-neutral-900 hover:border-neutral-600"
            }`}
          >
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm font-medium">{iv.title}</span>
              <span className="text-xs text-cyan-400">
                {(iv.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              {iv.description}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}

// --- Main SidePanel ---

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
      className={`px-3 py-1.5 rounded text-xs font-medium transition ${
        active
          ? "bg-neutral-100 text-neutral-900"
          : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
      } ${loading ? "opacity-50 cursor-wait" : ""}`}
    >
      {loading ? "Loading…" : label}
    </button>
  );
}
