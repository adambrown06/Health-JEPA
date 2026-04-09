import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    twins: [
      // Cluster 0 — Metabolic Syndrome (Critical Risk)
      {
        id: "tw_001",
        label: "Twin A — F/54",
        similarity: 0.94,
        cluster_id: 0,
        cluster_name: "Metabolic Syndrome",
        outcome_type: "positive",
        coordinate: { x: 2.8, y: -0.9, z: 1.3 },
        outcome: "Successfully reversed early T2D and metabolic syndrome markers within 14 months.",
        lifestyle_changes:
          "Adopted a whole-food Mediterranean diet, eliminated processed sugars, and committed to 150 min/week of brisk walking + twice-weekly resistance training. Joined a community cooking class for accountability.",
        outcome_bullets: [
          "HbA1c dropped from 7.3% to 6.2% in 8 months",
          "BMI reduced from 30.8 to 27.2 over 14 months",
          "Triglycerides normalized from 210 to 148 mg/dL",
          "Metformin dosage reduced after month 10",
        ],
      },
      {
        id: "tw_002",
        label: "Twin B — F/50",
        similarity: 0.91,
        cluster_id: 0,
        cluster_name: "Metabolic Syndrome",
        outcome_type: "negative",
        coordinate: { x: 1.5, y: -2.1, z: 0.4 },
        outcome: "Progressed to insulin-dependent T2D with peripheral neuropathy within 18 months.",
        lifestyle_changes:
          "No sustained dietary changes — reported difficulty with meal planning and time constraints. Exercise attempts were sporadic. Missed multiple follow-up appointments and stopped taking metformin after 4 months.",
        outcome_bullets: [
          "HbA1c rose from 7.1% to 9.4% over 18 months",
          "Required insulin initiation at month 16",
          "Developed early peripheral neuropathy (numbness in feet)",
          "Medication non-adherence was a primary contributing factor",
        ],
      },
      {
        id: "tw_003",
        label: "Twin C — F/49",
        similarity: 0.89,
        cluster_id: 0,
        cluster_name: "Metabolic Syndrome",
        outcome_type: "positive",
        coordinate: { x: 3.2, y: -0.5, z: 2.1 },
        outcome: "Achieved glycemic control and normalized blood pressure through exercise-first approach.",
        lifestyle_changes:
          "Enrolled in a structured cardiac fitness program (150 min/week aerobic + yoga for stress). Reduced sodium intake to <2g/day. Started meal prepping on weekends to control portions and macros.",
        outcome_bullets: [
          "HbA1c improved from 7.4% to 6.2% in 10 months",
          "Blood pressure normalized to 122/78 without new medication",
          "Lost 8 kg through exercise alone in first 6 months",
          "Reported improved sleep quality and reduced stress levels",
        ],
      },
      // Cluster 1 — Cardiovascular Risk (High Risk)
      {
        id: "tw_004",
        label: "Twin D — M/61",
        similarity: 0.82,
        cluster_id: 1,
        cluster_name: "Cardiovascular Risk",
        outcome_type: "negative",
        coordinate: { x: -11.5, y: 3.4, z: -7.2 },
        outcome: "Experienced non-fatal MI at 14 months requiring percutaneous coronary intervention.",
        lifestyle_changes:
          "Struggled with statin adherence due to muscle pain side effects. Continued high-sodium diet and sedentary work schedule. Skipped follow-up lipid panels for 8+ months.",
        outcome_bullets: [
          "LDL remained above 160 mg/dL due to statin non-adherence",
          "Non-fatal MI at month 14 requiring PCI with stent placement",
          "Post-event: initiated dual antiplatelet therapy",
          "Delayed specialist referral contributed to late intervention",
        ],
      },
      {
        id: "tw_005",
        label: "Twin E — F/58",
        similarity: 0.79,
        cluster_id: 1,
        cluster_name: "Cardiovascular Risk",
        outcome_type: "positive",
        coordinate: { x: -12.8, y: 2.1, z: -8.8 },
        outcome: "Dramatically reduced cardiovascular risk through medication switch and lifestyle overhaul.",
        lifestyle_changes:
          "After statin intolerance, switched to PCSK9 inhibitor. Began daily 45-minute morning walks and adopted a DASH diet. Reduced alcohol to ≤2 drinks/week and quit smoking after 20+ years with nicotine patch support.",
        outcome_bullets: [
          "LDL reduced from 172 to 68 mg/dL on PCSK9 inhibitor",
          "Framingham risk score dropped from 22% to 11% in 18 months",
          "Smoking cessation sustained for 12+ months",
          "Resting heart rate improved from 82 to 68 bpm",
        ],
      },
      {
        id: "tw_006",
        label: "Twin F — M/55",
        similarity: 0.76,
        cluster_id: 1,
        cluster_name: "Cardiovascular Risk",
        outcome_type: "positive",
        coordinate: { x: -11.2, y: 4.1, z: -7.8 },
        outcome: "Completed cardiac rehab and built sustainable daily exercise habit over 2+ years.",
        lifestyle_changes:
          "Completed 12-week cardiac rehabilitation program. Transitioned to daily 8,000-step walking habit using fitness tracker for accountability. Switched to plant-forward meals 5 days/week and attended monthly support group.",
        outcome_bullets: [
          "Resting BP normalized to 124/78 from 148/92",
          "Daily walking habit (8k steps) sustained for 2+ years",
          "Weight loss of 9 kg maintained at 24-month follow-up",
          "No cardiac events since rehab completion",
        ],
      },
      // Cluster 2 — Renal & Endocrine (Moderate Risk)
      {
        id: "tw_007",
        label: "Twin G — F/56",
        similarity: 0.74,
        cluster_id: 2,
        cluster_name: "Renal & Endocrine",
        outcome_type: "negative",
        coordinate: { x: 13.5, y: 4.8, z: -3.5 },
        outcome: "Progressed to stage 3 CKD within 24 months due to uncontrolled hypertension.",
        lifestyle_changes:
          "Inconsistent blood pressure monitoring at home. High-sodium dietary habits persisted despite counseling. Delayed nephrology referral by 8 months due to missed appointments.",
        outcome_bullets: [
          "eGFR declined from 68 to 42 mL/min over 24 months",
          "Persistent hypertension (avg 152/96) despite medication",
          "Stage 3 CKD diagnosis at month 20",
          "Late specialist referral was a key contributing factor",
        ],
      },
      {
        id: "tw_008",
        label: "Twin H — M/52",
        similarity: 0.71,
        cluster_id: 2,
        cluster_name: "Renal & Endocrine",
        outcome_type: "positive",
        coordinate: { x: 14.2, y: 5.5, z: -4.8 },
        outcome: "Stabilized kidney function through medication optimization and dietary protein management.",
        lifestyle_changes:
          "Worked with a renal dietitian to reduce protein intake to 0.8 g/kg/day. Increased water intake to 2.5L/day. Adopted stress management through daily 15-minute meditation practice and consistent sleep schedule.",
        outcome_bullets: [
          "eGFR stabilized at 62 mL/min (was declining at 5.2/year)",
          "Annual eGFR decline slowed to just 1.1 mL/min/year",
          "ACE inhibitor titrated to optimal dose",
          "Blood pressure consistently below 130/80 for 12+ months",
        ],
      },
      {
        id: "tw_009",
        label: "Twin I — F/60",
        similarity: 0.68,
        cluster_id: 2,
        cluster_name: "Renal & Endocrine",
        outcome_type: "positive",
        coordinate: { x: 13.8, y: 4.2, z: -3.1 },
        outcome: "Thyroid function normalized with cascading improvements across metabolic markers.",
        lifestyle_changes:
          "Levothyroxine dose optimized with quarterly TSH monitoring. Started morning strength training (3x/week) to combat fatigue. Eliminated gluten after discovering sensitivity. Prioritized 7.5+ hours of sleep nightly.",
        outcome_bullets: [
          "TSH normalized from 8.2 to 2.1 mIU/L within 4 months",
          "Energy levels dramatically improved per patient report",
          "Weight loss of 6 kg as a secondary benefit",
          "HbA1c dropped from 7.1% to 6.4% without diabetes medication changes",
        ],
      },
      // Cluster 3 — Respiratory Health (Low Risk)
      {
        id: "tw_010",
        label: "Twin J — M/48",
        similarity: 0.65,
        cluster_id: 3,
        cluster_name: "Respiratory Health",
        outcome_type: "positive",
        coordinate: { x: -5.5, y: -7.2, z: 10.5 },
        outcome: "Achieved complete asthma control with zero exacerbations over 24 months.",
        lifestyle_changes:
          "Switched from reactive to preventive inhaler use. Removed carpet from bedroom and invested in HEPA air purifier. Started swimming 3x/week (low-trigger exercise). Tracked peak flow daily using a phone app.",
        outcome_bullets: [
          "Zero asthma exacerbations in 24 months",
          "Peak flow consistently above 85% predicted",
          "Reduced rescue inhaler use from 4x/week to <1x/month",
          "Improved exercise tolerance — completed first 5K run",
        ],
      },
      {
        id: "tw_011",
        label: "Twin K — F/45",
        similarity: 0.62,
        cluster_id: 3,
        cluster_name: "Respiratory Health",
        outcome_type: "positive",
        coordinate: { x: -6.8, y: -8.5, z: 9.8 },
        outcome: "Fully resolved sleep apnea through weight loss and consistent CPAP adherence.",
        lifestyle_changes:
          "Committed to calorie tracking and lost 15 kg over 10 months. Used CPAP every night with >6 hours compliance. Established strict sleep hygiene — no screens after 9pm, consistent 10:30pm bedtime. Started cycling to work (25 min each way).",
        outcome_bullets: [
          "AHI dropped from 28 to 3 events/hour",
          "Weight loss of 15 kg sustained for 12+ months",
          "Daytime fatigue and brain fog fully resolved",
          "CPAP adherence >90% of nights, >6 hours avg",
        ],
      },
      {
        id: "tw_012",
        label: "Twin L — M/50",
        similarity: 0.59,
        cluster_id: 3,
        cluster_name: "Respiratory Health",
        outcome_type: "negative",
        coordinate: { x: -5.8, y: -8.1, z: 10.8 },
        outcome: "COPD progressed from GOLD stage 1 to stage 2 due to continued tobacco use.",
        lifestyle_changes:
          "Multiple smoking cessation attempts failed — tried patches and varenicline without success. Declined pulmonary rehab referral. Continued occupational dust exposure without recommended PPE.",
        outcome_bullets: [
          "FEV1 declined from 72% to 58% predicted over 18 months",
          "Progression from GOLD stage 1 to stage 2",
          "Required initiation of long-acting bronchodilator (LABA)",
          "Continued tobacco use (0.5 pack/day) despite counseling",
        ],
      },
      // Cluster 4 — Active & Preventive (Minimal Risk)
      {
        id: "tw_013",
        label: "Twin M — F/44",
        similarity: 0.55,
        cluster_id: 4,
        cluster_name: "Active & Preventive",
        outcome_type: "positive",
        coordinate: { x: 10.5, y: -5.5, z: 8.5 },
        outcome: "Maintained optimal health markers for 3 consecutive years through proactive lifestyle.",
        lifestyle_changes:
          "Trains 200+ min/week (mix of running, strength, and yoga). Follows a flexitarian diet rich in vegetables, legumes, and omega-3s. Annual comprehensive screening with proactive specialist consults. Meditates 10 min daily.",
        outcome_bullets: [
          "All biomarkers in optimal range for 3 consecutive years",
          "BMI stable at 22.8, resting HR 58 bpm",
          "No new risk factors identified at annual screening",
          "VO2max estimated at 42 (excellent for age)",
        ],
      },
      {
        id: "tw_014",
        label: "Twin N — M/47",
        similarity: 0.52,
        cluster_id: 4,
        cluster_name: "Active & Preventive",
        outcome_type: "positive",
        coordinate: { x: 9.8, y: -6.2, z: 7.5 },
        outcome: "Early polyp detection through preventive screening — all markers remain low-risk.",
        lifestyle_changes:
          "Proactive about preventive screenings (colonoscopy, cardiac calcium score). Runs 3x/week and does functional fitness 2x/week. Home-cooks 90% of meals, limits alcohol to weekends only. Uses continuous glucose monitor out of curiosity, not necessity.",
        outcome_bullets: [
          "Preventive colonoscopy found early polyps — removed successfully",
          "All cardiovascular and metabolic markers remain low-risk",
          "BMI stable at 23.1, blood pressure 118/74",
          "Cardiac calcium score of 0 (no coronary calcification)",
        ],
      },
      {
        id: "tw_015",
        label: "Twin O — F/51",
        similarity: 0.50,
        cluster_id: 4,
        cluster_name: "Active & Preventive",
        outcome_type: "positive",
        coordinate: { x: 10.2, y: -6.8, z: 8.2 },
        outcome: "Transitioned from moderate to high fitness level with significant cardio improvement.",
        lifestyle_changes:
          "Hired a personal trainer and progressed from 3x to 5x/week training. Completed first half-marathon. Switched to whole-food plant-based diet. Practices cold exposure (cold showers) and prioritizes 8 hours of sleep.",
        outcome_bullets: [
          "VO2max improved from 32 to 38 mL/kg/min",
          "10-year ASCVD risk score dropped below 5%",
          "Resting heart rate decreased from 74 to 60 bpm",
          "Completed first half-marathon at age 50",
        ],
      },
    ],
  });
}
