import type {
  IngestResponse,
  GalaxyResponse,
  EmbeddingResponse,
  NeighborsResponse,
  InterventionsResponse,
  WearableData,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function ingestFile(): Promise<IngestResponse> {
  const pdfRes = await fetch("/mock_mychart.pdf");
  const blob = await pdfRes.blob();

  const form = new FormData();
  form.append("file", blob, "mychart_export.pdf");

  const res = await fetch(`${BASE}/api/v1/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json() as Promise<IngestResponse>;
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
