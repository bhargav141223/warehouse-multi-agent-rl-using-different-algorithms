'use client'

import { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'

/* ─── helpers ─────────────────────────────────────────────────────── */
function cellToWorld(col: number, row: number, gridSize: number): THREE.Vector3 {
  return new THREE.Vector3(col - gridSize / 2 + 0.5, 0, row - gridSize / 2 + 0.5)
}

/* ─── PATH DASHED LINE  ───────────────────────────────────────────── */
interface PathLineProps {
  fromCol: number; fromRow: number
  toCol: number;   toRow: number
  gridSize: number; color: string
}
function PathLine({ fromCol, fromRow, toCol, toRow, gridSize, color }: PathLineProps) {
  const lineRef = useRef<THREE.Line>(null)

  // Build a segmented dashed path on the grid
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = []
    const steps = Math.max(Math.abs(toCol - fromCol), Math.abs(toRow - fromRow), 1)
    for (let s = 0; s <= steps; s++) {
      const col = Math.round(fromCol + ((toCol - fromCol) * s) / steps)
      const row = Math.round(fromRow + ((toRow - fromRow) * s) / steps)
      const w = cellToWorld(col, row, gridSize)
      pts.push(new THREE.Vector3(w.x, -0.42, w.z))
    }
    return pts
  }, [fromCol, fromRow, toCol, toRow, gridSize])

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry().setFromPoints(points)
    return geo
  }, [points])

  useFrame((state) => {
    if (lineRef.current) {
      // Pulse opacity
      const mat = lineRef.current.material as THREE.LineBasicMaterial
      mat.opacity = 0.35 + Math.sin(state.clock.elapsedTime * 3) * 0.25
    }
  })

  return (
    <line ref={lineRef as any} geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={0.5} linewidth={2} />
    </line>
  )
}

/* ─── AGENT ROBOT ─────────────────────────────────────────────────── */
interface AgentProps {
  col: number; row: number          // current grid position (changes each step)
  goalCol: number; goalRow: number  // fixed goal
  gridSize: number
  color: string; id: number
  isMoving: boolean
}

function AgentRobot({ col, row, goalCol, goalRow, gridSize, color, id, isMoving }: AgentProps) {
  const bodyRef   = useRef<THREE.Group>(null)
  const glowRef   = useRef<THREE.Mesh>(null)
  const shadowRef = useRef<THREE.Mesh>(null)
  const ringRef   = useRef<THREE.Mesh>(null)

  // Keep world-space target in a ref – updated via useEffect
  const worldTarget = useRef<THREE.Vector3>(cellToWorld(col, row, gridSize).clone())
  const worldCurrent= useRef<THREE.Vector3>(cellToWorld(col, row, gridSize).clone())

  // ★ KEY FIX: use primitive values (col, row) as deps so the effect fires on every cell change
  useEffect(() => {
    worldTarget.current.copy(cellToWorld(col, row, gridSize))
  }, [col, row, gridSize])

  const emissCol = useMemo(() => new THREE.Color(color), [color])

  useFrame((state, delta) => {
    const speed = 5
    worldCurrent.current.lerp(worldTarget.current, Math.min(1, delta * speed))
    const wx = worldCurrent.current.x
    const wz = worldCurrent.current.z

    if (bodyRef.current) {
      bodyRef.current.position.set(wx, 0, wz)
      const dist = worldCurrent.current.distanceTo(worldTarget.current)
      if (dist > 0.05) {
        const dir   = worldTarget.current.clone().sub(worldCurrent.current)
        const angle = Math.atan2(dir.x, dir.z)
        bodyRef.current.rotation.y = THREE.MathUtils.lerp(bodyRef.current.rotation.y, angle, delta * 10)
      }
    }
    if (glowRef.current) {
      glowRef.current.position.set(wx, 0, wz)
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 5 + id * 1.5) * 0.14
      glowRef.current.scale.setScalar(pulse)
    }
    if (shadowRef.current) shadowRef.current.position.set(wx, -0.48, wz)
    if (ringRef.current)   ringRef.current.rotation.z = state.clock.elapsedTime * 2.5
  })

  const initW = cellToWorld(col, row, gridSize)

  return (
    <group>
      {/* Dashed path line from start to goal */}
      <PathLine
        fromCol={col} fromRow={row}
        toCol={goalCol} toRow={goalRow}
        gridSize={gridSize} color={color}
      />

      {/* Drop shadow */}
      <mesh ref={shadowRef} position={[initW.x, -0.48, initW.z]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.3, 24]} />
        <meshBasicMaterial color="#000" transparent opacity={0.3} depthWrite={false} />
      </mesh>

      {/* Glow halo */}
      <mesh ref={glowRef} position={[initW.x, 0, initW.z]}>
        <sphereGeometry args={[0.36, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.13} side={THREE.BackSide} />
      </mesh>

      {/* Body group – moved by useFrame */}
      <group ref={bodyRef} position={[initW.x, 0, initW.z]}>
        {/* Main sphere */}
        <mesh castShadow>
          <sphereGeometry args={[0.26, 32, 32]} />
          <meshStandardMaterial color={color} metalness={0.65} roughness={0.18}
            emissive={emissCol} emissiveIntensity={0.25} />
        </mesh>
        {/* Sensor dome */}
        <mesh position={[0, 0.22, 0]} castShadow>
          <sphereGeometry args={[0.11, 16, 16]} />
          <meshStandardMaterial color="#e2e8f0" metalness={0.9} roughness={0.05} />
        </mesh>
        {/* Spinner ring */}
        <mesh ref={ringRef} position={[0, -0.24, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.23, 0.31, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.65} />
        </mesh>
        {/* Forward dot */}
        <mesh position={[0, 0, 0.27]}>
          <circleGeometry args={[0.045, 10]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <pointLight color={color} intensity={0.55} distance={2.3} />
      </group>
    </group>
  )
}

/* ─── GOAL MARKER ─────────────────────────────────────────────────── */
interface GoalProps {
  col: number; row: number
  gridSize: number
  reached: boolean
  agentColor: string
}
function GoalMarker({ col, row, gridSize, reached, agentColor }: GoalProps) {
  const gemRef  = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const w = cellToWorld(col, row, gridSize)
  const baseY = 0.28
  const activeColor = reached ? '#22c55e' : agentColor
  const emissCol = useMemo(() => new THREE.Color(activeColor), [activeColor])

  useFrame((state) => {
    if (reached) { if (gemRef.current) gemRef.current.rotation.y += 0.04; return }
    if (gemRef.current) {
      gemRef.current.rotation.y  = state.clock.elapsedTime * 1.3
      gemRef.current.position.y  = baseY + Math.sin(state.clock.elapsedTime * 2.2) * 0.1
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 1.8
      ringRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1)
    }
  })

  return (
    <group position={[w.x, 0, w.z]}>
      {/* Floor circle */}
      <mesh position={[0, -0.48, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.48, 32]} />
        <meshBasicMaterial color={activeColor} transparent opacity={reached ? 0.55 : 0.28} depthWrite={false} />
      </mesh>
      {/* Ring border */}
      <mesh ref={ringRef} position={[0, -0.47, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.44, 0.5, 32]} />
        <meshBasicMaterial color={activeColor} transparent opacity={0.75} depthWrite={false} />
      </mesh>
      {/* Crystal gem */}
      <mesh ref={gemRef} position={[0, baseY, 0]} castShadow>
        <octahedronGeometry args={[0.2, 0]} />
        <meshStandardMaterial color={activeColor} metalness={0.95} roughness={0.04}
          emissive={emissCol} emissiveIntensity={reached ? 1.1 : 0.65} />
      </mesh>
      {/* Glow shell */}
      <mesh position={[0, baseY, 0]}>
        <octahedronGeometry args={[0.28, 0]} />
        <meshBasicMaterial color={activeColor} transparent opacity={0.14} side={THREE.BackSide} />
      </mesh>
      <pointLight color={activeColor} intensity={reached ? 1.1 : 0.6} distance={2.5} position={[0, 0.9, 0]} />
    </group>
  )
}

/* ─── SHELF / OBSTACLE ────────────────────────────────────────────── */
function WarehouseShelf({ position }: { position: [number, number, number] }) {
  const boxColors = ['#c2410c', '#0369a1', '#15803d']
  return (
    <group position={position}>
      {[[-0.38, -0.38], [0.38, -0.38], [-0.38, 0.38], [0.38, 0.38]].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.45, z]} castShadow>
          <boxGeometry args={[0.07, 1.4, 0.07]} />
          <meshStandardMaterial color="#b45309" metalness={0.7} roughness={0.3} />
        </mesh>
      ))}
      {[-0.28, 0.1, 0.48].map((y, i) => (
        <mesh key={i} position={[0, y, 0]} castShadow receiveShadow>
          <boxGeometry args={[0.88, 0.06, 0.88]} />
          <meshStandardMaterial color="#92400e" metalness={0.3} roughness={0.7} />
        </mesh>
      ))}
      {[[-0.18, 0.2, 0.08], [0.18, 0.2, 0.08], [0, 0.2, -0.08]].map(([x, y, z], i) => (
        <mesh key={`box-${i}`} position={[x, y, z]} castShadow>
          <boxGeometry args={[0.22, 0.2, 0.22]} />
          <meshStandardMaterial color={boxColors[i % 3]} metalness={0.1} roughness={0.8} />
        </mesh>
      ))}
    </group>
  )
}

/* ─── WAREHOUSE BUILDING ──────────────────────────────────────────── */
function WarehouseBuilding({ gridSize }: { gridSize: number }) {
  const half  = gridSize / 2
  const wallH = 3.8
  const wallY = wallH / 2 - 0.5

  const lights = useMemo(() =>
    Array.from({ length: 3 }, (_, r) =>
      Array.from({ length: 2 }, (_, c) => ({
        key: `${r}-${c}`,
        lx: (c - 0.5) * (gridSize / 2),
        lz: (r - 1) * (gridSize / 3),
      }))
    ).flat(), [gridSize])

  return (
    <group>
      {/* Concrete floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]} receiveShadow>
        <planeGeometry args={[gridSize, gridSize]} />
        <meshStandardMaterial color="#9ca3af" roughness={0.92} metalness={0.04} />
      </mesh>

      {/* Grid lines */}
      <Grid args={[gridSize, gridSize]}
        cellSize={1} cellThickness={0.5} cellColor="#6b7280"
        sectionSize={gridSize} sectionThickness={1.5} sectionColor="#374151"
        fadeDistance={40} fadeStrength={1.2}
        infiniteGrid={false} followCamera={false} />

      {/* Yellow safety lanes – horizontal */}
      {[-gridSize / 4, 0, gridSize / 4].map((z, i) => (
        <mesh key={`hl-${i}`} position={[0, -0.48, z]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[gridSize, 0.09]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.65} depthWrite={false} />
        </mesh>
      ))}
      {/* Yellow safety lanes – vertical */}
      {[-gridSize / 4, 0, gridSize / 4].map((x, i) => (
        <mesh key={`vl-${i}`} position={[x, -0.48, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[0.09, gridSize]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.65} depthWrite={false} />
        </mesh>
      ))}

      {/* Walls */}
      {[
        { pos: [0, wallY, -half] as [number,number,number], size: [gridSize + 0.24, wallH, 0.18] as [number,number,number] },
        { pos: [-half, wallY, 0] as [number,number,number], size: [0.18, wallH, gridSize + 0.24] as [number,number,number] },
        { pos: [half, wallY, 0]  as [number,number,number], size: [0.18, wallH, gridSize + 0.24] as [number,number,number] },
        { pos: [0, 0.4, half]    as [number,number,number], size: [gridSize + 0.24, 1.5, 0.18] as [number,number,number] },
      ].map(({ pos, size }, i) => (
        <mesh key={i} position={pos} receiveShadow castShadow>
          <boxGeometry args={size} />
          <meshStandardMaterial color="#d1d5db" roughness={0.88} metalness={0.06} />
        </mesh>
      ))}

      {/* Roof trusses */}
      {Array.from({ length: Math.ceil(gridSize / 3) }, (_, i) => i * 3 - half + 1.5).map((x, i) => (
        <group key={i} position={[x, wallH - 0.45, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.13, 0.22, gridSize]} />
            <meshStandardMaterial color="#6b7280" metalness={0.7} roughness={0.3} />
          </mesh>
          <mesh rotation={[Math.PI / 4, 0, 0]} castShadow>
            <boxGeometry args={[0.07, 0.1, gridSize * 1.1]} />
            <meshStandardMaterial color="#9ca3af" metalness={0.6} roughness={0.4} />
          </mesh>
        </group>
      ))}

      {/* Ceiling fluorescent lights */}
      {lights.map(({ key, lx, lz }) => (
        <group key={key} position={[lx, wallH - 0.5, lz]}>
          <mesh>
            <boxGeometry args={[1.6, 0.1, 0.3]} />
            <meshStandardMaterial color="#f3f4f6" emissive="#ffffff" emissiveIntensity={0.75} />
          </mesh>
          <pointLight intensity={2.4} distance={gridSize * 0.9} color="#fff8e7"
            castShadow shadow-mapSize={[512, 512]} />
        </group>
      ))}

      {/* Corner columns */}
      {[[-half, -half], [half, -half], [-half, half], [half, half]].map(([cx, cz], i) => (
        <group key={i}>
          <mesh position={[cx, wallY, cz]} castShadow>
            <boxGeometry args={[0.28, wallH, 0.28]} />
            <meshStandardMaterial color="#9ca3af" metalness={0.5} roughness={0.5} />
          </mesh>
          {[0.2, 0.8, 1.4].map((dy, j) => (
            <mesh key={j} position={[cx, dy - 0.3, cz]}>
              <boxGeometry args={[0.3, 0.08, 0.3]} />
              <meshBasicMaterial color={j % 2 === 0 ? '#f97316' : '#1f2937'} />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  )
}

/* ─── DEMO ANIMATION HOOK ─────────────────────────────────────────── */
/**
 * Produces smoothly-stepping positions that walk each agent from its
 * start cell to its goal cell and back, one cell at a time.
 * Returns live backend positions while training is active.
 */
function useDemoPositions(
  starts: [number, number][],
  goals:  [number, number][],
  isTraining: boolean,
  livePositions: [number, number][]
): [number, number][] {
  const [demoPos, setDemoPos] = useState<[number, number][]>(starts)
  const stepRef   = useRef(0)
  const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (isTraining) return   // live positions handled outside

    // Build back-and-forth grid path for each agent
    function buildPath(from: [number, number], to: [number, number]): [number, number][] {
      const path: [number, number][] = []
      const dx = to[0] - from[0]
      const dy = to[1] - from[1]
      const steps = Math.max(Math.abs(dx), Math.abs(dy), 1)

      for (let s = 0; s <= steps; s++) {
        path.push([
          Math.round(from[0] + (dx * s) / steps),
          Math.round(from[1] + (dy * s) / steps),
        ])
      }
      // Reverse back to start (skip duplicate endpoints)
      for (let s = steps - 1; s >= 1; s--) {
        path.push([
          Math.round(from[0] + (dx * s) / steps),
          Math.round(from[1] + (dy * s) / steps),
        ])
      }
      return path
    }

    const paths = starts.map((s, i) => buildPath(s, goals[i] ?? s))
    const maxLen = Math.max(...paths.map(p => p.length))
    stepRef.current = 0

    timerRef.current = setInterval(() => {
      stepRef.current = (stepRef.current + 1) % maxLen
      setDemoPos(paths.map(p => p[stepRef.current % p.length]))
    }, 300)

    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, JSON.stringify(starts), JSON.stringify(goals)])

  return isTraining ? livePositions : demoPos
}

/* ─── SCENE ───────────────────────────────────────────────────────── */
export interface SceneConfig {
  gridSize: number
  agentPositions: [number, number][]
  goalPositions:  [number, number][]
  obstaclePositions: [number, number][]
  goalsReached: boolean[]
  agentColors?: string[]
  isTraining?: boolean
}

const DEFAULT_COLORS = ['#3b82f6', '#ef4444', '#fbbf24', '#a855f7']

function Scene({
  gridSize, agentPositions, goalPositions, obstaclePositions,
  goalsReached, agentColors = DEFAULT_COLORS, isTraining = false,
}: SceneConfig) {
  const displayPos = useDemoPositions(
    agentPositions, goalPositions, isTraining, agentPositions
  )

  return (
    <>
      <ambientLight intensity={0.42} color="#e8f4f8" />
      <WarehouseBuilding gridSize={gridSize} />

      {obstaclePositions.map((pos, i) => {
        const w = cellToWorld(pos[0], pos[1], gridSize)
        return <WarehouseShelf key={`sh-${i}`} position={[w.x, -0.05, w.z]} />
      })}

      {goalPositions.map((pos, i) => (
        <GoalMarker key={`g-${i}`} col={pos[0]} row={pos[1]} gridSize={gridSize}
          reached={goalsReached?.[i] || false}
          agentColor={agentColors[i % agentColors.length]} />
      ))}

      {displayPos.map((pos, i) => (
        <AgentRobot key={`a-${i}`}
          col={pos[0]}     row={pos[1]}
          goalCol={goalPositions[i]?.[0] ?? pos[0]}
          goalRow={goalPositions[i]?.[1] ?? pos[1]}
          gridSize={gridSize}
          color={agentColors[i % agentColors.length]}
          id={i} isMoving={isTraining} />
      ))}

      <OrbitControls target={[0, 0, 0]} enablePan enableZoom enableRotate
        minDistance={5} maxDistance={gridSize * 2.5}
        maxPolarAngle={Math.PI / 2 - 0.04}
        autoRotate={!isTraining} autoRotateSpeed={0.5} />
    </>
  )
}

/* ─── PUBLIC EXPORT ───────────────────────────────────────────────── */
export interface Warehouse3DProps extends SceneConfig {
  className?: string
}

export default function Warehouse3D({ className = '', ...props }: Warehouse3DProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  if (!mounted) return (
    <div className={`w-full h-full flex items-center justify-center bg-gray-800 ${className}`}>
      <div className="flex flex-col items-center gap-3">
        <div className="w-10 h-10 border-2 border-orange-500/40 border-t-orange-400 rounded-full animate-spin" />
        <p className="text-gray-400 text-sm">Loading Warehouse…</p>
      </div>
    </div>
  )

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas shadows
        camera={{ position: [props.gridSize * 0.95, props.gridSize * 0.72, props.gridSize * 0.95], fov: 46, near: 0.1, far: 300 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color('#1c2333'))
          gl.shadowMap.enabled = true
          gl.shadowMap.type = THREE.PCFSoftShadowMap
        }}
      >
        <Scene {...props} />
      </Canvas>
    </div>
  )
}
