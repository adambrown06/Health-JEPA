import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json({
    source: "Apple Watch Series 9 + Oura Ring Gen 3",
    sync_window: "Last 30 days — synced just now",
    metrics: {
      resting_heart_rate: {
        value: 72,
        unit: "bpm",
        trend: "stable",
        series: [74, 73, 72, 73, 71, 72, 72, 73, 71, 72, 71, 70, 72, 71],
      },
      hrv: {
        value: 34,
        unit: "ms",
        trend: "declining",
        flag: "below_optimal",
        series: [42, 40, 38, 36, 37, 35, 34, 33, 35, 34, 33, 34, 34, 34],
      },
      daily_steps: {
        value: 5820,
        unit: "steps",
        trend: "declining",
        flag: "below_target",
        series: [7200, 6800, 6100, 5900, 5500, 6200, 5800, 5400, 5900, 5600, 5800, 5700, 5900, 5820],
      },
      active_zone_minutes: {
        value: 18,
        unit: "min/day",
        trend: "declining",
        flag: "below_target",
        series: [28, 25, 22, 20, 19, 21, 18, 17, 19, 18, 16, 18, 17, 18],
      },
      sleep_duration: {
        value: 6.3,
        unit: "hrs",
        trend: "stable",
        series: [6.5, 6.2, 6.4, 6.1, 6.3, 6.5, 6.2, 6.3, 6.1, 6.4, 6.2, 6.3, 6.3, 6.3],
      },
      deep_sleep: {
        value: 48,
        unit: "min",
        trend: "declining",
        flag: "below_optimal",
        series: [62, 58, 55, 52, 50, 53, 48, 47, 50, 49, 48, 47, 48, 48],
      },
      spo2: {
        value: 96.8,
        unit: "%",
        trend: "stable",
        series: [97.1, 96.9, 97.0, 96.8, 96.9, 96.7, 96.8, 96.9, 96.8, 96.7, 96.8, 96.8, 96.9, 96.8],
      },
      body_temperature_deviation: {
        value: 0.1,
        unit: "°C",
        trend: "stable",
        series: [0.0, 0.1, 0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1],
      },
      respiratory_rate: {
        value: 15.2,
        unit: "br/min",
        trend: "stable",
        series: [15.0, 15.1, 15.3, 15.2, 15.1, 15.2, 15.3, 15.2, 15.1, 15.2, 15.2, 15.3, 15.2, 15.2],
      },
    },
    insights: [
      "HRV has declined 19% over the past 30 days, suggesting elevated sympathetic nervous system activity. " +
        "This pattern is commonly associated with metabolic stress and suboptimal recovery.",
      "Daily step count averages 5,820 — below the 7,500-step threshold linked to improved glycemic control " +
        "in similar metabolic profiles. Increasing to 7,500+ steps could support HbA1c reduction.",
      "Deep sleep has dropped from 62 to 48 minutes. Research indicates deep sleep below 55 minutes " +
        "correlates with impaired glucose metabolism and increased insulin resistance.",
      "Active zone minutes (18 min/day) fall short of the 22+ min/day target for cardiovascular benefit. " +
        "Even a 10-minute brisk walk after meals could meaningfully improve post-prandial glucose.",
    ],
  });
}
