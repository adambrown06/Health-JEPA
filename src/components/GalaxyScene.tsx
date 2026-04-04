"use client";

import {
  useRef,
  useMemo,
  useEffect,
  useCallback,
  Component,
  type ReactNode,
} from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { useAppStore } from "@/lib/store";
import type { GalaxyPoint, DigitalTwin } from "@/lib/types";

const CLUSTER_COLORS: Record<number, string> = {
  0: "#6b7280",
  1: "#ef4444",
  2: "#f59e0b",
  3: "#3b82f6",
  4: "#22c55e",
};

const PRIMARY_CLUSTER = 1;
const DOT_RADIUS = 0.18;
const DOT_SEGMENTS = 8;

class SceneErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: string }
> {
  state = { hasError: false, error: "" };
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full bg-neutral-950 text-neutral-400 p-8 text-center">
          <div>
            <p className="text-sm font-medium mb-2">3D scene encountered an error</p>
            <p className="text-xs text-neutral-600">{this.state.error}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- All galaxy points rendered as a single InstancedMesh ---

function PatientDots() {
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const selectedId = useAppStore((s) => s.selectedPatientId);
  const selectById = useAppStore((s) => s.selectPatientById);
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current || galaxyPoints.length === 0) return;

    const color = new THREE.Color();
    const colors = new Float32Array(galaxyPoints.length * 3);

    for (let i = 0; i < galaxyPoints.length; i++) {
      const pt = galaxyPoints[i];

      dummy.position.set(pt.x, pt.y, pt.z);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      color.set(CLUSTER_COLORS[pt.cluster_id] ?? "#6b7280");
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  }, [galaxyPoints, dummy]);

  // Pulse the selected instance
  useFrame((state) => {
    if (!meshRef.current || galaxyPoints.length === 0) return;
    const selIdx = selectedId
      ? galaxyPoints.findIndex((p) => p.id === selectedId)
      : -1;

    if (selIdx >= 0) {
      const pt = galaxyPoints[selIdx];
      const s = 1.8 + Math.sin(state.clock.elapsedTime * 3) * 0.4;
      dummy.position.set(pt.x, pt.y, pt.z);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(selIdx, dummy.matrix);
      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const idx = e.instanceId;
      if (idx !== undefined && galaxyPoints[idx]) {
        const clicked = galaxyPoints[idx];
        selectById(selectedId === clicked.id ? null : clicked.id);
      }
    },
    [galaxyPoints, selectedId, selectById]
  );

  if (galaxyPoints.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, galaxyPoints.length]}
      onClick={handleClick}
    >
      <sphereGeometry args={[DOT_RADIUS, DOT_SEGMENTS, DOT_SEGMENTS]} />
      <meshStandardMaterial toneMapped={false} />
    </instancedMesh>
  );
}

// --- Twin dots: larger, brighter, rendered on top of galaxy dots ---

function TwinDots() {
  const twins = useAppStore((s) => s.twins);
  const selectedId = useAppStore((s) => s.selectedPatientId);
  const selectById = useAppStore((s) => s.selectPatientById);
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current || twins.length === 0) return;

    const color = new THREE.Color();
    const colors = new Float32Array(twins.length * 3);

    for (let i = 0; i < twins.length; i++) {
      const t = twins[i];
      dummy.position.set(t.coordinate.x, t.coordinate.y, t.coordinate.z);
      dummy.scale.setScalar(1.6);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      color.set(t.outcome_type === "positive" ? "#22c55e" : "#ef4444");
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  }, [twins, dummy]);

  useFrame((state) => {
    if (!meshRef.current || twins.length === 0) return;
    const selIdx = selectedId
      ? twins.findIndex((t) => t.id === selectedId)
      : -1;

    if (selIdx >= 0) {
      const t = twins[selIdx];
      const s = 2.0 + Math.sin(state.clock.elapsedTime * 3) * 0.5;
      dummy.position.set(t.coordinate.x, t.coordinate.y, t.coordinate.z);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(selIdx, dummy.matrix);
      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const idx = e.instanceId;
      if (idx !== undefined && twins[idx]) {
        selectById(selectedId === twins[idx].id ? null : twins[idx].id);
      }
    },
    [twins, selectedId, selectById]
  );

  if (twins.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, twins.length]}
      onClick={handleClick}
    >
      <sphereGeometry args={[DOT_RADIUS, 12, 12]} />
      <meshStandardMaterial emissiveIntensity={0.5} toneMapped={false} />
    </instancedMesh>
  );
}

// --- Tooltip for selected patient ---

function SelectionTooltip() {
  const selectedId = useAppStore((s) => s.selectedPatientId);
  const selectedTwin = useAppStore((s) => s.selectedTwin);
  const selectedPoint = useAppStore((s) => s.selectedPoint);
  const clusterNames = useAppStore((s) => s.clusterNames);

  if (!selectedId) return null;

  if (selectedTwin) {
    const c = selectedTwin.coordinate;
    return (
      <Html position={[c.x, c.y + 0.7, c.z]} center>
        <div className="bg-black/90 text-white text-xs px-2.5 py-1.5 rounded pointer-events-none select-none border border-neutral-600 max-w-[220px]">
          <div className="font-semibold">{selectedTwin.label}</div>
          <div className="text-[10px] text-neutral-400 mt-0.5">
            {(selectedTwin.similarity * 100).toFixed(0)}% match · {selectedTwin.cluster_name}
          </div>
          <div className={`text-[10px] mt-0.5 ${selectedTwin.outcome_type === "positive" ? "text-green-400" : "text-red-400"}`}>
            {selectedTwin.outcome_type === "positive" ? "Positive" : "Negative"} outcome
          </div>
        </div>
      </Html>
    );
  }

  if (selectedPoint) {
    const cName = clusterNames[String(selectedPoint.cluster_id)] ?? "Unknown";
    return (
      <Html position={[selectedPoint.x, selectedPoint.y + 0.6, selectedPoint.z]} center>
        <div className="bg-black/90 text-white text-xs px-2.5 py-1.5 rounded pointer-events-none select-none border border-neutral-600 max-w-[200px]">
          <div className="font-semibold">{selectedPoint.label}</div>
          <div className="text-[10px] text-neutral-400 mt-0.5">{cName}</div>
          <div className={`text-[10px] mt-0.5 ${selectedPoint.outcome_type === "positive" ? "text-green-400" : "text-red-400"}`}>
            {selectedPoint.outcome_type === "positive" ? "Positive" : "Negative"} outcome
          </div>
        </div>
      </Html>
    );
  }

  return null;
}

// --- Cluster labels ---

function IslandLabel({
  clusterId,
  center,
  name,
  isPrimary,
  count,
}: {
  clusterId: number;
  center: [number, number, number];
  name: string;
  isPrimary: boolean;
  count: number;
}) {
  const focusedId = useAppStore((s) => s.focusedClusterId);
  const setFocused = useAppStore((s) => s.setFocusedClusterId);
  const isFocused = focusedId === clusterId;

  return (
    <Html position={[center[0], center[1] + (isPrimary ? 6 : 4.5), center[2]]} center>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setFocused(isFocused ? null : clusterId);
        }}
        className={`whitespace-nowrap text-center transition-opacity cursor-pointer ${
          isFocused ? "opacity-100" : "opacity-50 hover:opacity-80"
        }`}
      >
        <div
          className={`font-semibold uppercase tracking-widest ${
            isPrimary ? "text-sm" : "text-[10px]"
          }`}
          style={{ color: CLUSTER_COLORS[clusterId] ?? "#999" }}
        >
          {name}
        </div>
        <div className="text-[9px] text-neutral-500 mt-0.5">
          {isPrimary ? "Your primary cluster" : `${count} patients`}
        </div>
      </button>
    </Html>
  );
}

// --- Camera controller ---

function CameraController() {
  const focusedClusterId = useAppStore((s) => s.focusedClusterId);
  const selectedPatientId = useAppStore((s) => s.selectedPatientId);
  const selectedTwin = useAppStore((s) => s.selectedTwin);
  const selectedPoint = useAppStore((s) => s.selectedPoint);
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const { camera } = useThree();
  const target = useRef(new THREE.Vector3(0, 2, 0));
  const desiredPos = useRef(new THREE.Vector3(0, 28, 35));

  useEffect(() => {
    if (selectedPatientId) {
      let px = 0, py = 0, pz = 0;
      if (selectedTwin) {
        px = selectedTwin.coordinate.x;
        py = selectedTwin.coordinate.y;
        pz = selectedTwin.coordinate.z;
      } else if (selectedPoint) {
        px = selectedPoint.x;
        py = selectedPoint.y;
        pz = selectedPoint.z;
      }
      target.current.set(px, py, pz);
      desiredPos.current.set(px + 3, py + 3, pz + 5);
      return;
    }

    if (focusedClusterId !== null) {
      const pts = galaxyPoints.filter((p) => p.cluster_id === focusedClusterId);
      if (pts.length > 0) {
        let sx = 0, sy = 0, sz = 0;
        for (const p of pts) { sx += p.x; sy += p.y; sz += p.z; }
        const n = pts.length;
        const cx = sx / n, cy = sy / n, cz = sz / n;
        const isPrimary = focusedClusterId === PRIMARY_CLUSTER;
        const dist = isPrimary ? 12 : 8;
        target.current.set(cx, cy, cz);
        desiredPos.current.set(cx + dist * 0.4, cy + dist * 0.6, cz + dist);
        return;
      }
    }

    target.current.set(0, 0, 0);
    desiredPos.current.set(0, 28, 35);
  }, [focusedClusterId, selectedPatientId, selectedTwin, selectedPoint, galaxyPoints]);

  useFrame(() => {
    camera.position.lerp(desiredPos.current, 0.03);
    camera.lookAt(target.current);
  });

  return null;
}

// --- Trajectory line for interventions ---

function TrajectoryLine() {
  const intervention = useAppStore((s) => s.activeIntervention);

  const geometry = useMemo(() => {
    if (!intervention) return null;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array([
      0, 0, 0,
      intervention.target_coordinate.x,
      intervention.target_coordinate.y,
      intervention.target_coordinate.z,
    ]);
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [intervention]);

  if (!intervention || !geometry) return null;
  const end = intervention.target_coordinate;

  return (
    <group>
      <line geometry={geometry}>
        <lineDashedMaterial color="#22d3ee" dashSize={0.4} gapSize={0.2} linewidth={1} />
      </line>
      <mesh position={[end.x, end.y, end.z]}>
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.8} transparent opacity={0.9} />
      </mesh>
      <Html position={[end.x, end.y + 0.6, end.z]} center>
        <div className="bg-cyan-950/90 text-cyan-300 text-xs px-2 py-1 rounded whitespace-nowrap pointer-events-none select-none border border-cyan-800">
          Projected Position
        </div>
      </Html>
    </group>
  );
}

function DashedLineUpdater() {
  const intervention = useAppStore((s) => s.activeIntervention);
  const { scene } = useThree();
  useEffect(() => {
    scene.traverse((obj) => {
      if ((obj as THREE.Line).isLine && (obj as THREE.Line).computeLineDistances) {
        (obj as THREE.Line).computeLineDistances();
      }
    });
  }, [intervention, scene]);
  return null;
}

// --- Main scene ---

function SceneContents() {
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const clusterNames = useAppStore((s) => s.clusterNames);
  const clearSelection = useAppStore((s) => s.clearSelection);

  const clusterInfo = useMemo(() => {
    const acc: Record<number, { sx: number; sy: number; sz: number; n: number }> = {};
    for (const p of galaxyPoints) {
      if (!acc[p.cluster_id]) acc[p.cluster_id] = { sx: 0, sy: 0, sz: 0, n: 0 };
      acc[p.cluster_id].sx += p.x;
      acc[p.cluster_id].sy += p.y;
      acc[p.cluster_id].sz += p.z;
      acc[p.cluster_id].n += 1;
    }
    return Object.entries(acc).map(([cid, { sx, sy, sz, n }]) => ({
      clusterId: Number(cid),
      center: [sx / n, sy / n, sz / n] as [number, number, number],
      count: n,
    }));
  }, [galaxyPoints]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[15, 20, 15]} intensity={0.5} />
      <pointLight position={[-15, 15, -15]} intensity={0.3} />

      {/* Click on empty space (canvas miss) to deselect */}
      <mesh visible={false} onClick={() => clearSelection()}>
        <sphereGeometry args={[150, 8, 8]} />
        <meshBasicMaterial side={THREE.BackSide} />
      </mesh>

      <PatientDots />
      <TwinDots />
      <SelectionTooltip />

      {clusterInfo.map(({ clusterId, center, count }) => (
        <IslandLabel
          key={clusterId}
          clusterId={clusterId}
          center={center}
          name={clusterNames[String(clusterId)] ?? `Cluster ${clusterId}`}
          isPrimary={clusterId === PRIMARY_CLUSTER}
          count={count}
        />
      ))}

      <TrajectoryLine />
      <DashedLineUpdater />
      <CameraController />
      <OrbitControls enableDamping dampingFactor={0.1} />
    </>
  );
}

export default function GalaxyScene() {
  return (
    <SceneErrorBoundary>
      <Canvas
        camera={{ position: [0, 28, 35], fov: 50, near: 0.1, far: 400 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => gl.setClearColor("#0a0a0a")}
      >
        <SceneContents />
      </Canvas>
    </SceneErrorBoundary>
  );
}
