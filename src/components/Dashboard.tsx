"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import SidePanel from "./SidePanel";
import { useAppStore } from "@/lib/store";
import { fetchNeighbors } from "@/lib/api";

const GalaxyScene = dynamic(() => import("./GalaxyScene"), {
  ssr: false,
  loading: () => (
    <div className="flex flex-col items-center justify-center h-full bg-[#F4F6F8] text-[#6B7280] text-sm gap-3">
      <div className="rounded-lg bg-[#F4F6F8] p-1.5 inline-flex">
        <Image
          src="/brand-compass-mark.png"
          alt=""
          width={240}
          height={112}
          className="logo-on-brand w-48 sm:w-52 h-auto object-contain"
          aria-hidden
        />
      </div>
      <span>Initializing 3D engine…</span>
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
    <div className="flex h-screen w-screen overflow-hidden bg-[#F4F6F8]">
      <div className="w-[70%] h-full relative border-r border-[#D9DEE3] bg-[#F4F6F8]">
        <GalaxyScene />
      </div>
      <div className="w-[30%] h-full bg-[#F4F6F8] overflow-hidden flex flex-col">
        <div className="shrink-0 px-4 pt-3 pb-3 border-b border-[#D9DEE3] bg-[#F4F6F8]">
          <div className="flex items-center gap-3 min-w-0">
            <div className="rounded-lg bg-[#F4F6F8] p-1 shrink-0">
              <Image
                src="/brand-compass-mark.png"
                alt="Cohort Compass"
                width={280}
                height={140}
                className="logo-on-brand h-16 sm:h-[4.5rem] w-auto max-w-[min(200px,42vw)] object-contain object-left"
              />
            </div>
            <p className="text-xs sm:text-sm font-bold text-[#0B3C8C] leading-snug min-w-0 tracking-tight">
              Your Health Universe
            </p>
          </div>
        </div>
        <div className="flex-1 min-h-0">
          <SidePanel />
        </div>
      </div>
    </div>
  );
}
