export interface PatientBaseline {
  age: number;
  sex: string;
  bmi: number;
  hba1c: number;
  fasting_glucose: number;
  blood_pressure: string;
  ldl: number;
  triglycerides: number;
  risk_label: string;
  summary: string;
}

export interface IngestResponse {
  status: string;
  patient: PatientBaseline;
}

export interface GalaxyPoint {
  id: string;
  label: string;
  outcome_type: "positive" | "negative";
  x: number;
  y: number;
  z: number;
  cluster_id: number;
}

export interface GalaxyResponse {
  points: GalaxyPoint[];
  clusters: Record<string, string>;
}

export interface Coordinate {
  x: number;
  y: number;
  z: number;
}

export interface EmbeddingResponse {
  coordinate: Coordinate;
  cluster_id: number;
  cluster_name: string;
  embedding_dim: number;
}

export interface DigitalTwin {
  id: string;
  label: string;
  similarity: number;
  cluster_id: number;
  cluster_name: string;
  outcome_type: "positive" | "negative";
  coordinate: Coordinate;
  outcome: string;
}

export interface NeighborsResponse {
  twins: DigitalTwin[];
}

export interface Intervention {
  id: string;
  title: string;
  description: string;
  confidence: number;
  target_coordinate: Coordinate;
}

export interface InterventionsResponse {
  interventions: Intervention[];
}

export interface WearableMetric {
  value: number;
  unit: string;
  trend: string;
  flag?: string;
  series: number[];
}

export interface WearableData {
  source: string;
  sync_window: string;
  metrics: Record<string, WearableMetric>;
  insights: string[];
}
