import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json({
    interventions: [
      {
        id: "iv_001",
        title: "Structured Exercise Program",
        description:
          "150 minutes/week of moderate-intensity aerobic activity combined with " +
          "resistance training 2x/week. Based on twin outcomes, this intervention " +
          "projects a 1.2-point HbA1c reduction and 8–12% weight loss over 12 months.",
        confidence: 0.91,
        target_coordinate: { x: 3.4, y: 0.8, z: 2.5 },
      },
      {
        id: "iv_002",
        title: "GLP-1 Receptor Agonist Therapy",
        description:
          "Initiation of semaglutide or liraglutide alongside current metformin. " +
          "Digital twin analysis shows 87% of similar patients achieved target " +
          "HbA1c < 6.5% and significant weight reduction within 10 months.",
        confidence: 0.87,
        target_coordinate: { x: 4.1, y: 1.2, z: 3.0 },
      },
      {
        id: "iv_003",
        title: "Mediterranean Dietary Pattern",
        description:
          "Transition to Mediterranean-style diet rich in olive oil, nuts, legumes, " +
          "and fish. Cohort data indicates a 22% reduction in cardiovascular events " +
          "and improved lipid profiles within 6 months for similar risk profiles.",
        confidence: 0.84,
        target_coordinate: { x: 2.9, y: 0.4, z: 1.9 },
      },
      {
        id: "iv_004",
        title: "SGLT2 Inhibitor Addition",
        description:
          "Add empagliflozin or dapagliflozin for dual glycemic and cardiovascular " +
          "benefit. Neighborhood analysis shows cardio-renal protection with " +
          "average 3 kg weight loss as secondary benefit.",
        confidence: 0.79,
        target_coordinate: { x: 3.8, y: 0.6, z: 2.8 },
      },
    ],
  });
}
