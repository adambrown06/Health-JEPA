import type {
  IngestResponse,
  GalaxyResponse,
  EmbeddingResponse,
  NeighborsResponse,
  InterventionsResponse,
  WearableData,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function ingestFile(): Promise<IngestResponse> {
  return request<IngestResponse>("/api/v1/ingest", { method: "POST" });
}

export function fetchGalaxy(): Promise<GalaxyResponse> {
  return request<GalaxyResponse>("/api/v1/galaxy");
}

export function fetchEmbedding(): Promise<EmbeddingResponse> {
  return request<EmbeddingResponse>("/api/v1/embedding", { method: "POST" });
}

export function fetchNeighbors(): Promise<NeighborsResponse> {
  return request<NeighborsResponse>("/api/v1/neighbors");
}

export function fetchInterventions(): Promise<InterventionsResponse> {
  return request<InterventionsResponse>("/api/v1/interventions", {
    method: "POST",
  });
}

export function fetchWearables(): Promise<WearableData> {
  return request<WearableData>("/api/v1/wearables", { method: "POST" });
}
