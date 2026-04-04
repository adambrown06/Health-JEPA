"use client";

import { create } from "zustand";
import type {
  PatientBaseline,
  GalaxyPoint,
  Coordinate,
  DigitalTwin,
  Intervention,
  WearableData,
} from "./types";

export type AppPhase = "ingestion" | "loading" | "dashboard";

export type SidePanelView = "stats" | "wearables" | "twins" | "drill-down" | "interventions" | "patient-detail";

interface AppState {
  phase: AppPhase;
  setPhase: (p: AppPhase) => void;

  patient: PatientBaseline | null;
  setPatient: (p: PatientBaseline) => void;

  galaxyPoints: GalaxyPoint[];
  setGalaxyPoints: (pts: GalaxyPoint[]) => void;

  clusterNames: Record<string, string>;
  setClusterNames: (c: Record<string, string>) => void;

  userCoord: Coordinate | null;
  setUserCoord: (c: Coordinate) => void;

  userClusterId: number;

  userCluster: string | null;
  setUserCluster: (name: string) => void;

  twins: DigitalTwin[];
  setTwins: (t: DigitalTwin[]) => void;

  selectedTwin: DigitalTwin | null;
  selectTwin: (t: DigitalTwin | null) => void;

  focusedClusterId: number | null;
  setFocusedClusterId: (id: number | null) => void;

  selectedPatientId: string | null;
  selectedPoint: GalaxyPoint | null;
  selectPatientById: (id: string | null) => void;

  clearSelection: () => void;

  interventions: Intervention[];
  setInterventions: (i: Intervention[]) => void;

  activeIntervention: Intervention | null;
  setActiveIntervention: (i: Intervention | null) => void;

  sidePanelView: SidePanelView;
  setSidePanelView: (v: SidePanelView) => void;

  interventionsLoading: boolean;
  setInterventionsLoading: (b: boolean) => void;

  wearables: WearableData | null;
  setWearables: (w: WearableData) => void;

  wearablesSynced: boolean;
  setWearablesSynced: (b: boolean) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  phase: "ingestion",
  setPhase: (phase) => set({ phase }),

  patient: null,
  setPatient: (patient) => set({ patient }),

  galaxyPoints: [],
  setGalaxyPoints: (galaxyPoints) => set({ galaxyPoints }),

  clusterNames: {},
  setClusterNames: (clusterNames) => set({ clusterNames }),

  userCoord: null,
  setUserCoord: (userCoord) => set({ userCoord }),

  userClusterId: 1,

  userCluster: null,
  setUserCluster: (userCluster) => set({ userCluster }),

  twins: [],
  setTwins: (twins) => set({ twins }),

  selectedTwin: null,
  selectTwin: (twin) => {
    set({
      selectedTwin: twin,
      selectedPatientId: twin?.id ?? null,
      selectedPoint: null,
      focusedClusterId: twin?.cluster_id ?? null,
      sidePanelView: twin ? "twins" : get().sidePanelView,
    });
  },

  focusedClusterId: null,
  setFocusedClusterId: (focusedClusterId) =>
    set({ focusedClusterId, selectedTwin: null, selectedPatientId: null, selectedPoint: null }),

  selectedPatientId: null,
  selectedPoint: null,
  selectPatientById: (id) => {
    if (!id) {
      set({ selectedPatientId: null, selectedTwin: null, selectedPoint: null });
      return;
    }
    const twins = get().twins;
    const twinMatch = twins.find((t) => t.id === id);
    if (twinMatch) {
      set({
        selectedPatientId: id,
        selectedTwin: twinMatch,
        selectedPoint: null,
        focusedClusterId: twinMatch.cluster_id,
        sidePanelView: "twins",
      });
      return;
    }
    const pts = get().galaxyPoints;
    const ptMatch = pts.find((p) => p.id === id) ?? null;
    set({
      selectedPatientId: id,
      selectedTwin: null,
      selectedPoint: ptMatch,
      focusedClusterId: ptMatch?.cluster_id ?? get().focusedClusterId,
      sidePanelView: ptMatch ? "patient-detail" : get().sidePanelView,
    });
  },

  clearSelection: () =>
    set({
      focusedClusterId: null,
      selectedPatientId: null,
      selectedTwin: null,
      selectedPoint: null,
    }),

  interventions: [],
  setInterventions: (interventions) => set({ interventions }),

  activeIntervention: null,
  setActiveIntervention: (activeIntervention) => set({ activeIntervention }),

  sidePanelView: "stats",
  setSidePanelView: (sidePanelView) =>
    set({ sidePanelView, activeIntervention: null }),

  interventionsLoading: false,
  setInterventionsLoading: (interventionsLoading) =>
    set({ interventionsLoading }),

  wearables: null,
  setWearables: (wearables) => set({ wearables }),

  wearablesSynced: false,
  setWearablesSynced: (wearablesSynced) => set({ wearablesSynced }),
}));
