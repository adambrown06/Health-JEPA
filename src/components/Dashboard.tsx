"use client";

import dynamic from "next/dynamic";
import SidePanel from "./SidePanel";

const GalaxyScene = dynamic(() => import("./GalaxyScene"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-neutral-950 text-neutral-500 text-sm">
      Initializing 3D engine…
    </div>
  ),
});

export default function Dashboard() {
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* 3D Visualization — 70% */}
      <div className="w-[70%] h-full relative">
        <GalaxyScene />
      </div>

      {/* Side Panel — 30% */}
      <div className="w-[30%] h-full border-l border-neutral-800 bg-black">
        <SidePanel />
      </div>
    </div>
  );
}
