"use client";

import {
  useRef,
  useMemo,
  useEffect,
  useLayoutEffect,
  Component,
  type ReactNode,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import * as THREE from "three";
import { useAppStore } from "@/lib/store";

/** Risk gradient: 0 = highest risk (red) → 4 = lowest risk (green) */
const CLUSTER_COLORS: Record<number, string> = {
  0: "#ef4444",
  1: "#f97316",
  2: "#eab308",
  3: "#84cc16",
  4: "#22c55e",
};

const RISK_LABELS: Record<number, string> = {
  0: "Critical Risk",
  1: "High Risk",
  2: "Moderate Risk",
  3: "Low Risk",
  4: "Minimal Risk",
};

const PRIMARY_CLUSTER = 0;
const BRAND_BG = "#f4f6f8";
const DOT_RADIUS = 0.2;
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
        <div className="flex items-center justify-center h-full bg-[#F4F6F8] text-[#5C6773] p-8 text-center">
          <div>
            <p className="text-sm font-medium mb-2 text-[#0B3C8C]">3D scene encountered an error</p>
            <p className="text-xs text-[#6B7280]">{this.state.error}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/** Per-vertex white so instanceColor works correctly. */
function sphereGeometryWithWhiteColors(
  radius: number,
  widthSegments: number,
  heightSegments: number
) {
  const g = new THREE.SphereGeometry(radius, widthSegments, heightSegments);
  const n = g.attributes.position.count;
  const white = new Float32Array(n * 3);
  white.fill(1);
  g.setAttribute("color", new THREE.BufferAttribute(white, 3));
  return g;
}

// --- All galaxy points as visual (non-interactive) dots ---

function PatientDots() {
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const geometry = useMemo(
    () => sphereGeometryWithWhiteColors(DOT_RADIUS, DOT_SEGMENTS, DOT_SEGMENTS),
    []
  );

  useLayoutEffect(() => {
    if (!meshRef.current || galaxyPoints.length === 0) return;

    const color = new THREE.Color();
    const colors = new Float32Array(galaxyPoints.length * 3);

    for (let i = 0; i < galaxyPoints.length; i++) {
      const pt = galaxyPoints[i];
      const cid = Number(pt.cluster_id);

      dummy.position.set(pt.x, pt.y, pt.z);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      color.set(CLUSTER_COLORS[cid] ?? "#a5b4fc");
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    const attr = new THREE.InstancedBufferAttribute(colors, 3);
    attr.needsUpdate = true;
    meshRef.current.instanceColor = attr;
  }, [galaxyPoints, dummy]);

  if (galaxyPoints.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, undefined, galaxyPoints.length]}
      raycast={() => null}
    >
      <meshStandardMaterial
        vertexColors
        roughness={0.22}
        metalness={0.06}
        toneMapped
      />
    </instancedMesh>
  );
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
  const anyFocused = focusedId !== null;
  const riskLabel = RISK_LABELS[clusterId] ?? "Unknown";
  const clusterColor = CLUSTER_COLORS[clusterId] ?? "#a5b4fc";

  if (anyFocused && isFocused) return null;

  return (
    <Html position={[center[0], center[1] + (isPrimary ? 7 : 5.5), center[2]]} center>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setFocused(isFocused ? null : clusterId);
        }}
        className={`whitespace-nowrap text-center cursor-pointer transition-all duration-200 ${
          anyFocused ? "opacity-40" : "hover:scale-105"
        }`}
      >
        <div
          className="rounded-xl px-4 py-2.5 shadow-lg border backdrop-blur-sm"
          style={{
            background: "rgba(255,255,255,0.92)",
            borderColor: clusterColor,
            borderWidth: isPrimary ? 2 : 1,
          }}
        >
          <div
            className={`font-extrabold tracking-wide leading-tight ${
              isPrimary ? "text-base" : "text-sm"
            }`}
            style={{ color: clusterColor }}
          >
            {name}
          </div>
          <div
            className="text-[11px] font-bold mt-0.5 uppercase tracking-widest"
            style={{ color: clusterColor, opacity: 0.75 }}
          >
            {riskLabel}
          </div>
          <div className="text-[11px] text-[#5C6773] mt-0.5 font-medium">
            {isPrimary ? `Your cluster · ${count} patients` : `${count} patients`}
          </div>
        </div>
      </button>
    </Html>
  );
}

// --- Camera controller ---

function CameraController() {
  const focusedClusterId = useAppStore((s) => s.focusedClusterId);
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const { camera } = useThree();
  const target = useRef(new THREE.Vector3(0, 2, 0));
  const desiredPos = useRef(new THREE.Vector3(0, 28, 35));

  useEffect(() => {
    if (focusedClusterId !== null) {
      const pts = galaxyPoints.filter((p) => p.cluster_id === focusedClusterId);
      if (pts.length > 0) {
        let sx = 0, sy = 0, sz = 0;
        for (const p of pts) { sx += p.x; sy += p.y; sz += p.z; }
        const n = pts.length;
        const cx = sx / n, cy = sy / n, cz = sz / n;
        const dist = focusedClusterId === PRIMARY_CLUSTER ? 14 : 10;
        target.current.set(cx, cy, cz);
        desiredPos.current.set(cx + dist * 0.4, cy + dist * 0.6, cz + dist);
        return;
      }
    }

    target.current.set(0, 0, 0);
    desiredPos.current.set(0, 28, 35);
  }, [focusedClusterId, galaxyPoints]);

  useFrame(() => {
    camera.position.lerp(desiredPos.current, 0.03);
    camera.lookAt(target.current);
  });

  return null;
}

// --- Trajectory line for interventions ---

function TrajectoryLine() {
  const intervention = useAppStore((s) => s.activeIntervention);
  if (!intervention) return null;
  const end = intervention.target_coordinate;
  const start: [number, number, number] = [0, 0, 0];
  const endPt: [number, number, number] = [end.x, end.y, end.z];

  return (
    <group>
      <Line
        points={[start, endPt]}
        color="#2EC4C7"
        lineWidth={2}
        dashed
        dashSize={0.4}
        gapSize={0.2}
      />
      <mesh position={[end.x, end.y, end.z]}>
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshStandardMaterial color="#1FA3B3" emissive="#0E7C91" emissiveIntensity={0.5} transparent opacity={0.95} />
      </mesh>
      <Html position={[end.x, end.y + 0.6, end.z]} center>
        <div className="bg-[#0E7C91]/95 text-white text-xs px-2 py-1 rounded whitespace-nowrap pointer-events-none select-none border border-[#2EC4C7]/50">
          Projected Position
        </div>
      </Html>
    </group>
  );
}

// --- Main scene ---

function SceneContents() {
  const galaxyPoints = useAppStore((s) => s.galaxyPoints);
  const clusterNames = useAppStore((s) => s.clusterNames);
  const setFocused = useAppStore((s) => s.setFocusedClusterId);

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
      <hemisphereLight args={["#ffffff", "#c8d0e0", 0.6]} />
      <ambientLight intensity={0.68} />
      <directionalLight position={[12, 18, 10]} intensity={1.05} />
      <pointLight position={[15, 20, 15]} intensity={0.55} />
      <pointLight position={[-15, 15, -15]} intensity={0.38} />

      {/* Click empty space to deselect */}
      <mesh visible={false} onClick={() => setFocused(null)}>
        <sphereGeometry args={[150, 8, 8]} />
        <meshBasicMaterial side={THREE.BackSide} />
      </mesh>

      <PatientDots />

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
        onCreated={({ gl, scene }) => {
          const col = new THREE.Color(BRAND_BG);
          gl.setClearColor(col);
          scene.background = col;
        }}
      >
        <SceneContents />
      </Canvas>
    </SceneErrorBoundary>
  );
}
