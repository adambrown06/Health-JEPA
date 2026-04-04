"use client";

import { useRef, useMemo, useEffect, Component, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { useAppStore } from "@/lib/store";

const CLUSTER_COLORS: Record<number, string> = {
  0: "#6b7280",
  1: "#ef4444",
  2: "#f59e0b",
  3: "#3b82f6",
  4: "#22c55e",
};

// --- Error Boundary to prevent white-screen crashes ---

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

// --- Background galaxy points (imperative geometry) ---

function BackgroundPoints() {
  const points = useAppStore((s) => s.galaxyPoints);
  const ref = useRef<THREE.Points>(null!);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    if (points.length === 0) return geo;

    const pos = new Float32Array(points.length * 3);
    const col = new Float32Array(points.length * 3);
    const tmp = new THREE.Color();

    for (let i = 0; i < points.length; i++) {
      pos[i * 3] = points[i].x;
      pos[i * 3 + 1] = points[i].y;
      pos[i * 3 + 2] = points[i].z;

      tmp.set(CLUSTER_COLORS[points[i].cluster_id] ?? "#6b7280");
      col[i * 3] = tmp.r;
      col[i * 3 + 1] = tmp.g;
      col[i * 3 + 2] = tmp.b;
    }

    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return geo;
  }, [points]);

  if (points.length === 0) return null;

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial
        size={0.12}
        vertexColors
        transparent
        opacity={0.45}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

// --- User's position (pulsing sphere) ---

function UserPoint() {
  const coord = useAppStore((s) => s.userCoord);
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const s = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.15;
      meshRef.current.scale.setScalar(s);
    }
  });

  if (!coord) return null;

  return (
    <group>
      <mesh ref={meshRef} position={[coord.x, coord.y, coord.z]}>
        <sphereGeometry args={[0.22, 16, 16]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive="#ffffff"
          emissiveIntensity={0.6}
        />
      </mesh>
      <Html position={[coord.x, coord.y + 0.55, coord.z]} center>
        <div className="bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap pointer-events-none select-none border border-neutral-700">
          You — Metabolic Risk
        </div>
      </Html>
    </group>
  );
}

// --- Neighbor bounding sphere ---

function NeighborSphere() {
  const coord = useAppStore((s) => s.userCoord);
  const twins = useAppStore((s) => s.twins);

  const radius = useMemo(() => {
    if (!coord || twins.length === 0) return 0;
    let maxDist = 0;
    for (const t of twins) {
      const dx = t.coordinate.x - coord.x;
      const dy = t.coordinate.y - coord.y;
      const dz = t.coordinate.z - coord.z;
      maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maxDist + 0.3;
  }, [coord, twins]);

  if (!coord || radius === 0) return null;

  return (
    <mesh position={[coord.x, coord.y, coord.z]}>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshBasicMaterial
        color="#facc15"
        transparent
        opacity={0.04}
        wireframe
      />
    </mesh>
  );
}

// --- Digital twin points ---

function TwinPoints() {
  const twins = useAppStore((s) => s.twins);
  if (twins.length === 0) return null;

  return (
    <>
      {twins.map((t) => (
        <mesh
          key={t.id}
          position={[t.coordinate.x, t.coordinate.y, t.coordinate.z]}
        >
          <sphereGeometry args={[0.1, 12, 12]} />
          <meshStandardMaterial
            color="#facc15"
            emissive="#facc15"
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}
    </>
  );
}

// --- Trajectory line (imperative geometry, no drei Line) ---

function TrajectoryLine() {
  const coord = useAppStore((s) => s.userCoord);
  const intervention = useAppStore((s) => s.activeIntervention);
  const lineRef = useRef<THREE.Line>(null!);

  const geometry = useMemo(() => {
    if (!coord || !intervention) return null;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array([
      coord.x, coord.y, coord.z,
      intervention.target_coordinate.x,
      intervention.target_coordinate.y,
      intervention.target_coordinate.z,
    ]);
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [coord, intervention]);

  if (!coord || !intervention || !geometry) return null;

  const end = intervention.target_coordinate;

  return (
    <group>
      {/* dashed trajectory line */}
      <line ref={lineRef} geometry={geometry}>
        <lineDashedMaterial
          color="#22d3ee"
          dashSize={0.3}
          gapSize={0.15}
          linewidth={1}
        />
      </line>

      {/* target endpoint sphere */}
      <mesh position={[end.x, end.y, end.z]}>
        <sphereGeometry args={[0.18, 16, 16]} />
        <meshStandardMaterial
          color="#22d3ee"
          emissive="#22d3ee"
          emissiveIntensity={0.8}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* label */}
      <Html position={[end.x, end.y + 0.5, end.z]} center>
        <div className="bg-cyan-950/90 text-cyan-300 text-xs px-2 py-1 rounded whitespace-nowrap pointer-events-none select-none border border-cyan-800">
          Projected Position
        </div>
      </Html>
    </group>
  );
}

// --- Cluster labels using Html (avoids drei Text font-loading crashes) ---

function ClusterLabels() {
  const clusterNames = useAppStore((s) => s.clusterNames);
  const points = useAppStore((s) => s.galaxyPoints);

  const centroids = useMemo(() => {
    if (points.length === 0) return [];
    const acc: Record<number, { sx: number; sy: number; sz: number; n: number }> = {};
    for (const p of points) {
      if (!acc[p.cluster_id])
        acc[p.cluster_id] = { sx: 0, sy: 0, sz: 0, n: 0 };
      acc[p.cluster_id].sx += p.x;
      acc[p.cluster_id].sy += p.y;
      acc[p.cluster_id].sz += p.z;
      acc[p.cluster_id].n += 1;
    }
    return Object.entries(acc).map(([cid, { sx, sy, sz, n }]) => ({
      cluster_id: Number(cid),
      x: sx / n,
      y: sy / n + 2.5,
      z: sz / n,
    }));
  }, [points]);

  if (centroids.length === 0) return null;

  return (
    <>
      {centroids.map((c) => (
        <Html key={c.cluster_id} position={[c.x, c.y, c.z]} center>
          <div
            className="text-[10px] font-semibold uppercase tracking-widest whitespace-nowrap pointer-events-none select-none opacity-60"
            style={{ color: CLUSTER_COLORS[c.cluster_id] ?? "#999" }}
          >
            {clusterNames[String(c.cluster_id)] ?? `Cluster ${c.cluster_id}`}
          </div>
        </Html>
      ))}
    </>
  );
}

// --- Smooth camera controller ---

function CameraController() {
  const coord = useAppStore((s) => s.userCoord);
  const intervention = useAppStore((s) => s.activeIntervention);
  const { camera } = useThree();
  const lookTarget = useRef(new THREE.Vector3(0, 0, 0));
  const desiredPos = useRef(new THREE.Vector3(10, 8, 10));

  useEffect(() => {
    if (intervention && coord) {
      const tc = intervention.target_coordinate;
      const mid = new THREE.Vector3(
        (coord.x + tc.x) / 2,
        (coord.y + tc.y) / 2,
        (coord.z + tc.z) / 2
      );
      lookTarget.current.copy(mid);
      const dist = new THREE.Vector3(
        coord.x - tc.x,
        coord.y - tc.y,
        coord.z - tc.z
      ).length();
      const offset = Math.max(dist * 1.5, 6);
      desiredPos.current.set(
        mid.x + offset * 0.6,
        mid.y + offset * 0.5,
        mid.z + offset
      );
    } else if (coord) {
      lookTarget.current.set(coord.x, coord.y, coord.z);
      desiredPos.current.set(coord.x + 6, coord.y + 4, coord.z + 8);
    } else {
      lookTarget.current.set(0, 0, 0);
      desiredPos.current.set(10, 8, 10);
    }
  }, [coord, intervention]);

  useFrame(() => {
    camera.position.lerp(desiredPos.current, 0.02);
    camera.lookAt(lookTarget.current);
  });

  return null;
}

// --- Dashed line needs computeLineDistances call ---

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

// --- Main export ---

export default function GalaxyScene() {
  return (
    <SceneErrorBoundary>
      <Canvas
        camera={{ position: [10, 8, 10], fov: 50, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.setClearColor("#0a0a0a");
        }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.6} />

        <BackgroundPoints />
        <UserPoint />
        <NeighborSphere />
        <TwinPoints />
        <TrajectoryLine />
        <ClusterLabels />
        <CameraController />
        <DashedLineUpdater />

        <OrbitControls enableDamping dampingFactor={0.1} />
      </Canvas>
    </SceneErrorBoundary>
  );
}
