import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json({
    status: "ok",
    patient: {
      age: 52,
      sex: "Female",
      bmi: 31.4,
      hba1c: 7.2,
      fasting_glucose: 142,
      blood_pressure: "138/88",
      ldl: 145,
      triglycerides: 198,
      risk_label: "High Risk — Metabolic Syndrome",
      summary:
        "52-year-old female presenting with early-stage type 2 diabetes, " +
        "elevated triglycerides, and stage 1 hypertension. BMI of 31.4 " +
        "indicates class I obesity. HbA1c of 7.2% suggests suboptimal " +
        "glycemic control over the past 3 months. Lipid panel shows " +
        "elevated LDL and triglycerides, increasing cardiovascular risk. " +
        "Recommend lifestyle intervention alongside pharmacotherapy review.",
    },
  });
}
