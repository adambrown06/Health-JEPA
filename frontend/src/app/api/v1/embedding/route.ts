import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json({
    coordinate: { x: 2.1, y: -1.4, z: 0.8 },
    cluster_id: 0,
    cluster_name: "Metabolic Syndrome",
    embedding_dim: 128,
  });
}
