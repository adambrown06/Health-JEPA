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

export type SidePanelView = "stats" | "wearables" | "twins" | "drill-down" | "interventions";

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

  userCluster: string | null;
  setUserCluster: (name: string) => void;

  twins: DigitalTwin[];
  setTwins: (t: DigitalTwin[]) => void;

  selectedTwin: DigitalTwin | null;
  setSelectedTwin: (t: DigitalTwin | null) => void;

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

export const useAppStore = create<AppState>((set) => ({
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

  userCluster: null,
  setUserCluster: (userCluster) => set({ userCluster }),

  twins: [],
  setTwins: (twins) => set({ twins }),

  selectedTwin: null,
  setSelectedTwin: (selectedTwin) => set({ selectedTwin }),

  interventions: [],
  setInterventions: (interventions) => set({ interventions }),

  activeIntervention: null,
  setActiveIntervention: (activeIntervention) => set({ activeIntervention }),

  sidePanelView: "stats",
  setSidePanelView: (sidePanelView) =>
    set({ sidePanelView, selectedTwin: null, activeIntervention: null }),

  interventionsLoading: false,
  setInterventionsLoading: (interventionsLoading) =>
    set({ interventionsLoading }),

  wearables: null,
  setWearables: (wearables) => set({ wearables }),

  wearablesSynced: false,
  setWearablesSynced: (wearablesSynced) => set({ wearablesSynced }),
}));
