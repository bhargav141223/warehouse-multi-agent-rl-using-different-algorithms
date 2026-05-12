'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar
} from 'recharts'
import {
  Play, Pause, Square, ArrowLeft, Download, RotateCcw,
  Zap, Database, Brain, Activity, CheckCircle, XCircle,
  ChevronRight, Settings, TrendingUp, AlertTriangle
} from 'lucide-react'
import toast from 'react-hot-toast'
import dynamic from 'next/dynamic'

const Warehouse3D = dynamic(() => import('../../components/Warehouse3D'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900">
      <div className="w-10 h-10 border-2 border-blue-500/40 border-t-blue-400 rounded-full animate-spin mb-3" />
      <p className="text-gray-400 text-sm">Loading 3D Engine…</p>
    </div>
  ),
})

interface TrainingMetrics {
  episode: number
  reward: number
  successRate: number
  collisions: number
  episodeLength: number
}

interface AgentState {
  episode: number
  step: number
  agentPositions: [number, number][]
  goalsReached: boolean[]
  dynamicObstacles: [number, number][]
  currentReward: number
  collisions: number
}

export default function TrainingPage() {
  const { sessionId } = useParams()
  const router = useRouter()
  
  const [isTraining, setIsTraining] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([])
  const [currentState, setCurrentState] = useState<AgentState | null>(null)
  const [currentEpisode, setCurrentEpisode] = useState(0)
  const [totalEpisodes, setTotalEpisodes] = useState(1000)
  const [isComplete, setIsComplete] = useState(false)
  const [ragStats, setRagStats] = useState({ totalQueries: 0, successfulRetrievals: 0 })
  
  const wsRef = useRef<WebSocket | null>(null)

  // Initialize WebSocket
  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/training/${sessionId}`)
    
    ws.onopen = () => {
      setIsConnected(true)
      toast.success('Connected to training server')
      
      // Send start command
      fetch('/api/training/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, action: 'start' })
      })
      setIsTraining(true)
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'step_update') {
        setCurrentState({
          episode: data.episode,
          step: data.step,
          agentPositions: data.agent_positions,
          goalsReached: data.goals_reached,
          dynamicObstacles: data.dynamic_obstacles || [],
          currentReward: data.current_reward,
          collisions: data.collisions
        })
        setCurrentEpisode(data.episode)
      } else if (data.type === 'episode_complete') {
        setMetrics(prev => {
          const newMetrics = [...prev, {
            episode: data.episode,
            reward: data.data.total_reward,
            successRate: data.data.success_rate,
            collisions: data.data.collisions,
            episodeLength: data.data.steps
          }]
          return newMetrics
        })
      } else if (data.type === 'training_complete') {
        setIsComplete(true)
        setIsTraining(false)
        toast.success('Training completed!')
      } else if (data.type === 'error') {
        toast.error(data.message)
      }
    }
    
    ws.onclose = () => {
      setIsConnected(false)
      setIsTraining(false)
    }
    
    ws.onerror = (error) => {
      toast.error('WebSocket error occurred')
      setIsConnected(false)
    }
    
    wsRef.current = ws
  }, [sessionId])

  // Fetch training stats periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!isTraining || !sessionId) return
      
      try {
        const response = await fetch(`/api/training/stats/${sessionId}`)
        if (response.ok) {
          const data = await response.json()
          if (data.rag_stats) {
            setRagStats(data.rag_stats)
          }
        }
      } catch (e) {
        // Silently fail
      }
    }, 5000)
    
    return () => clearInterval(interval)
  }, [isTraining, sessionId])

  const handleStart = () => {
    if (!isConnected) {
      connectWebSocket()
    }
  }

  const handlePause = async () => {
    await fetch('/api/training/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, action: 'pause' })
    })
    setIsTraining(false)
    toast.success('Training paused')
  }

  const handleStop = async () => {
    await fetch('/api/training/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, action: 'stop' })
    })
    wsRef.current?.close()
    setIsTraining(false)
    setIsConnected(false)
    toast.success('Training stopped')
  }

  const handleExportModel = () => {
    window.open(`/api/export/model/${sessionId}`, '_blank')
  }

  const handleExportReport = () => {
    window.open(`/api/export/report/${sessionId}`, '_blank')
  }

  const handleGoToResults = () => {
    router.push(`/results/${sessionId}`)
  }

  // Calculate moving averages for charts
  const chartData = metrics.map((m, idx) => {
    // Calculate 10-episode rolling averages for visualization
    const startIdx = Math.max(0, idx - 9)
    const recentMetrics = metrics.slice(startIdx, idx + 1)
    const avgReward10 = recentMetrics.reduce((a, b) => a + b.reward, 0) / recentMetrics.length
    const avgSuccessRate10 = recentMetrics.reduce((a, b) => a + b.successRate, 0) / recentMetrics.length
    const avgCollisions10 = recentMetrics.reduce((a, b) => a + b.collisions, 0) / recentMetrics.length
    
    return {
      ...m,
      episode: m.episode + 1,
      avgReward10,
      avgSuccessRate10,
      avgCollisions10
    }
  })

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 flex items-center justify-between"
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push('/environments')}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-2xl font-bold">Training Dashboard</h1>
              <p className="text-gray-400 text-sm">Session: {sessionId}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 rounded-lg">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            
            <div className="flex gap-2">
              {!isTraining && !isComplete ? (
                <button
                  onClick={handleStart}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                  Start Training
                </button>
              ) : isTraining ? (
                <button
                  onClick={handlePause}
                  className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>
              ) : null}
              
              {isTraining && (
                <button
                  onClick={handleStop}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  <Square className="w-4 h-4" />
                  Stop
                </button>
              )}
            </div>
          </div>
        </motion.div>

        {/* Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          {[
            { label: 'Episode', value: metrics.length, icon: <Activity className="w-4 h-4" />, color: 'blue' },
            { label: 'Avg Reward', value: metrics.length > 0 ? (metrics.reduce((a, b) => a + b.reward, 0) / metrics.length).toFixed(2) : '0.00', icon: <TrendingUp className="w-4 h-4" />, color: 'green' },
            { label: 'Success Rate', value: metrics.length > 0 ? `${((metrics.reduce((a, b) => a + b.successRate, 0) / metrics.length) * 100).toFixed(1)}%` : '0%', icon: <CheckCircle className="w-4 h-4" />, color: 'cyan' },
            { label: 'Collisions', value: metrics.length > 0 ? (metrics.reduce((a, b) => a + b.collisions, 0) / metrics.length).toFixed(1) : '0.0', icon: <AlertTriangle className="w-4 h-4" />, color: 'red' },
            { label: 'RAG Queries', value: ragStats.totalQueries, icon: <Database className="w-4 h-4" />, color: 'purple' },
            { label: 'RAG Success', value: ragStats.successfulRetrievals, icon: <Brain className="w-4 h-4" />, color: 'pink' }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className={`p-4 bg-slate-800/50 rounded-xl border border-${stat.color}-500/20`}
            >
              <div className={`flex items-center gap-2 text-${stat.color}-400 mb-2`}>
                {stat.icon}
                <span className="text-xs font-medium">{stat.label}</span>
              </div>
              <div className="text-2xl font-bold">{stat.value}</div>
            </motion.div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 3D Warehouse Visualization */}
          <div className="lg:col-span-1 bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                3D Warehouse View
              </h2>
              <span className="text-xs text-gray-400 flex items-center gap-1">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Live 3D
              </span>
            </div>
            <div className="h-[420px] rounded-lg overflow-hidden border border-slate-600">
              {currentState ? (
                <Warehouse3D
                  gridSize={8}
                  agentPositions={currentState.agentPositions}
                  goalPositions={([[7, 7], [0, 7], [7, 0], [0, 0]] as [number, number][]).slice(0, currentState.agentPositions.length)}
                  obstaclePositions={currentState.dynamicObstacles}
                  goalsReached={currentState.goalsReached}
                  isTraining={isTraining}
                />
              ) : (
                <Warehouse3D
                  gridSize={8}
                  agentPositions={[[1,1],[6,1],[1,6],[6,6]]}
                  goalPositions={[[7,7],[0,7],[7,0],[0,0]]}
                  obstaclePositions={[[3,2],[4,2],[3,5],[4,5],[2,4],[5,3]]}
                  goalsReached={[false,false,false,false]}
                  isTraining={false}
                />
              )}
            </div>
            {/* ── Agent Legend Panel ── */}
            <div className="mt-4 space-y-2">
              {[
                { id: 0, color: '#3b82f6', hex: 'bg-blue-500',   ring: 'border-blue-500',   name: 'Agent 0', start: [1,1] as [number,number], goal: [7,7] as [number,number] },
                { id: 1, color: '#ef4444', hex: 'bg-red-500',    ring: 'border-red-500',    name: 'Agent 1', start: [6,1] as [number,number], goal: [0,7] as [number,number] },
                { id: 2, color: '#fbbf24', hex: 'bg-yellow-500', ring: 'border-yellow-500', name: 'Agent 2', start: [1,6] as [number,number], goal: [7,0] as [number,number] },
                { id: 3, color: '#a855f7', hex: 'bg-purple-500', ring: 'border-purple-500', name: 'Agent 3', start: [6,6] as [number,number], goal: [0,0] as [number,number] },
              ].map((agent) => {
                const livePosArr = currentState?.agentPositions
                const livePos    = livePosArr ? livePosArr[agent.id] : null
                const reached    = currentState?.goalsReached?.[agent.id] || false
                const displayPos = livePos ?? agent.start
                return (
                  <div key={agent.id}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-900/70 border ${
                      reached ? 'border-green-500/50' : 'border-slate-700'
                    }`}
                  >
                    {/* Color dot */}
                    <div className={`w-3.5 h-3.5 rounded-full flex-shrink-0 ${agent.hex}`}
                      style={{ boxShadow: `0 0 8px ${agent.color}99` }} />

                    {/* Name */}
                    <span className="text-xs font-semibold text-white w-14 flex-shrink-0">{agent.name}</span>

                    {/* Route */}
                    <div className="flex items-center gap-1 text-[11px] text-gray-400 flex-1 min-w-0">
                      <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded text-gray-300">({agent.start[0]},{agent.start[1]})</span>
                      <span className="text-gray-600">→</span>
                      <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded" style={{ color: agent.color }}>({agent.goal[0]},{agent.goal[1]})</span>
                    </div>

                    {/* Current pos or reached badge */}
                    {reached ? (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 font-medium flex-shrink-0">✓ Done</span>
                    ) : (
                      <span className="text-[10px] font-mono text-gray-500 flex-shrink-0">@({displayPos[0]},{displayPos[1]})</span>
                    )}
                  </div>
                )
              })}

              {/* Global legend */}
              <div className="flex items-center justify-between text-[10px] text-gray-600 pt-1 px-1">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-400/60 inline-block" style={{display:'inline-block'}} />— path line</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-green-400/70 inline-block" />Goal reached</span>
                <span>Drag · Zoom · Pan</span>
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Reward Chart */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Reward Curve
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="episode" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="reward" stroke="#ef4444" strokeWidth={2} dot={false} name="Episode Reward" />
                  <Line type="monotone" dataKey="avgReward10" stroke="#22c55e" strokeWidth={2} dot={false} name="Avg Reward (10)" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Success Rate & Collisions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-cyan-400" />
                  Success Rate
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="episode" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[0, 1]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                    />
                    <Area type="monotone" dataKey="successRate" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} name="Success Rate (Real-time)" />
                    <Area type="monotone" dataKey="avgSuccessRate10" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.2} name="Success Rate (10-ep Avg)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  Collision Rate
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="episode" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    />
                    <Bar dataKey="collisions" fill="#ef4444" name="Collisions" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Episode Length */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                Episode Length
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="episode" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="episodeLength" stroke="#3b82f6" strokeWidth={2} dot={false} name="Steps" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {isComplete && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 flex justify-center gap-4"
          >
            <button
              onClick={handleGoToResults}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-xl transition-colors"
            >
              <ChevronRight className="w-5 h-5" />
              View Results
            </button>
          </motion.div>
        )}
      </div>
    </div>
  )
}
