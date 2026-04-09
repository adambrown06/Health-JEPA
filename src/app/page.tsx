"use client";

import { useAppStore } from "@/lib/store";
import IngestionView from "@/components/IngestionView";
import Dashboard from "@/components/Dashboard";

export default function Home() {
  const phase = useAppStore((s) => s.phase);

  if (phase === "ingestion" || phase === "loading") {
    return <IngestionView />;
  }

  return <Dashboard />;
}
