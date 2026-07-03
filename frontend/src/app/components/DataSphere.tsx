"use client";

import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

// Central glowing network globe
function Globe() {
  const globeRef = useRef<THREE.Group>(null);

  // Slow, cinematic rotation on both axes
  useFrame(({ clock }) => {
    const elapsedTime = clock.getElapsedTime();
    if (globeRef.current) {
      globeRef.current.rotation.y = elapsedTime * 0.12;
      globeRef.current.rotation.x = elapsedTime * 0.04;
    }
  });

  // Calculate points on a sphere using a golden spiral distribution
  const [positions, lineIndices] = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const idxs: THREE.Vector3[] = [];
    const count = 100; // Optimal density of nodes

    for (let i = 0; i < count; i++) {
      const phi = Math.acos(-1 + (2 * i) / count);
      const theta = Math.sqrt(count * Math.PI) * phi;

      const radius = 2.0;
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);

      pts.push(new THREE.Vector3(x, y, z));
    }

    // Connect close neighbors with segments to form the network grid
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const dist = pts[i].distanceTo(pts[j]);
        // Connect points that are within a specific threshold
        if (dist > 0.5 && dist < 1.3) {
          idxs.push(pts[i], pts[j]);
        }
      }
    }

    const flatPositions = new Float32Array(pts.flatMap((p) => [p.x, p.y, p.z]));
    const flatLinePositions = new Float32Array(idxs.flatMap((p) => [p.x, p.y, p.z]));

    return [flatPositions, flatLinePositions];
  }, []);

  return (
    <group ref={globeRef}>
      {/* 1. Network connections (glowing purple lines) */}
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[lineIndices, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#8a63f7" transparent opacity={0.3} />
      </lineSegments>

      {/* 2. Network nodes (glowing turquoise dots) */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color="#2dd4bf"
          size={0.065}
          sizeAttenuation
          transparent
          opacity={0.8}
        />
      </points>

      {/* 3. Outer delicate latitude/longitude wireframe structure */}
      <mesh>
        <sphereGeometry args={[2.08, 18, 18]} />
        <meshBasicMaterial
          color="#8a63f7"
          wireframe
          transparent
          opacity={0.06}
        />
      </mesh>

      {/* 4. Deep inner core (blocking background light) */}
      <mesh>
        <sphereGeometry args={[1.9, 16, 16]} />
        <meshBasicMaterial
          color="#0b0c10"
          transparent
          opacity={0.5}
        />
      </mesh>
    </group>
  );
}

// Surrounding floating particle cloud
function ParticleCloud() {
  const count = 350;
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      
      // Radii range between 2.6 and 4.2
      const r = 2.6 + Math.random() * 1.6;
      
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, []);

  const pointsRef = useRef<THREE.Points>(null);

  // Slow reverse rotation for depth
  useFrame(({ clock }) => {
    const elapsedTime = clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = -elapsedTime * 0.03;
      pointsRef.current.rotation.x = -elapsedTime * 0.01;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#8a63f7"
        size={0.035}
        sizeAttenuation
        transparent
        opacity={0.55}
      />
    </points>
  );
}

export default function DataSphere() {
  return (
    <div style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0, zIndex: 1, pointerEvents: "none" }}>
      <Canvas camera={{ position: [0, 0, 5.5], fof: 60 } as any} gl={{ antialias: true }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.2} />
        <Globe />
        <ParticleCloud />
      </Canvas>
    </div>
  );
}
