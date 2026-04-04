"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import SidePanel from "./SidePanel";
import { useAppStore } from "@/lib/store";
import { fetchNeighbors } from "@/lib/api";

const GalaxyScene = dynamic(() => import("./GalaxyScene"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-neutral-950 text-neutral-500 text-sm">
      Initializing 3D engine…
    </div>
  ),
});

export default function Dashboard() {
  const twins = useAppStore((s) => s.twins);
  const setTwins = useAppStore((s) => s.setTwins);

  useEffect(() => {
    if (twins.length === 0) {
      fetchNeighbors().then((data) => setTwins(data.twins));
    }
  }, [twins.length, setTwins]);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <div className="w-[70%] h-full relative">
        <GalaxyScene />
      </div>
      <div className="w-[30%] h-full border-l border-neutral-800 bg-black">
        <SidePanel />
      </div>
    </div>
  );
}
