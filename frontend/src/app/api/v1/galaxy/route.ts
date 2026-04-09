import { NextResponse } from "next/server";

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

export async function GET() {
  const rand = seededRandom(42);

  // Ordered by risk: 0 = highest risk (most patients), 4 = lowest risk (fewest)
  const clusterCenters: Record<number, { x: number; y: number; z: number }> = {
    0: { x: 2, y: -1, z: 1 },
    1: { x: -12, y: 3, z: -8 },
    2: { x: 14, y: 5, z: -4 },
    3: { x: -6, y: -8, z: 10 },
    4: { x: 10, y: -6, z: 8 },
  };

  // Higher risk = more patients, spread wider
  const clusterSizes = [120, 85, 60, 40, 25];
  const spreads = [5, 4, 3.5, 3, 2.5];
  // Higher risk clusters have more negative outcomes
  const negativeRates = [0.65, 0.50, 0.35, 0.20, 0.10];

  const labels = [
    "Metabolic Syndrome",
    "Cardiovascular Risk",
    "Renal & Endocrine",
    "Respiratory Health",
    "Active & Preventive",
  ];
  const sampleNames = [
    "Patient", "Subject", "Individual", "Case", "Participant",
  ];

  const points = [];
  let id = 0;

  for (let cid = 0; cid < 5; cid++) {
    const center = clusterCenters[cid];
    const size = clusterSizes[cid];
    const spread = spreads[cid];
    const negRate = negativeRates[cid];

    for (let i = 0; i < size; i++) {
      points.push({
        id: `pt_${String(id).padStart(4, "0")}`,
        label: `${sampleNames[cid]} ${id + 1}`,
        outcome_type: rand() < negRate ? "negative" : "positive",
        x: center.x + (rand() - 0.5) * spread * 2,
        y: center.y + (rand() - 0.5) * spread * 2,
        z: center.z + (rand() - 0.5) * spread * 2,
        cluster_id: cid,
      });
      id++;
    }
  }

  const clusters: Record<string, string> = {};
  labels.forEach((name, i) => {
    clusters[String(i)] = name;
  });

  return NextResponse.json({ points, clusters });
}
