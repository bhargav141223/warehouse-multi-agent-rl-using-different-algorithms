'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts'
import {
  Trophy, Download, RotateCcw, ArrowLeft, TrendingUp, AlertTriangle,
  Target, Clock, Users, CheckCircle, Brain, Database, Activity
} from 'lucide-react'
import toast from 'react-hot-toast'

interface TrainingResult {
  totalEpisodes: number
  avgReward: number
  maxReward: number
  minReward: number
  avgSuccessRate: number
  avgEpisodeLength: number
  avgCollisions: number
  totalCollisions: number
  rewardsPerEpisode: number[]
  successRates: number[]
  collisionCounts: number[]
  episodeLengths: number[]
  ragStats: {
    totalQueries: number
    successfulRetrievals: number
    retrievalSuccessRate: number
  }
  finalAgentPerformance: {
    agentId: number
    avgReward: number
    successRate: number
    collisions: number
  }[]
}

export default function ResultsPage() {
  const { sessionId } = useParams()
  const router = useRouter()
  const [results, setResults] = useState<TrainingResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchResults()
  }, [sessionId])

  const fetchResults = async () => {
    try {
      // Fetch training stats from API
      const response = await fetch(`/api/training/stats/${sessionId}`)
      if (!response.ok) throw new Error('Failed to fetch results')
      
      const data = await response.json()
      
      // Calculate aggregated metrics
      const rewards = data.rewards || []
      const successRates = data.success_rates || []
      const collisionCounts = data.collision_counts || []
      const episodeLengths = data.episode_lengths || []
      
      const resultData = {
        totalEpisodes: rewards.length,
        avgReward: rewards.length > 0 ? rewards.reduce((a: number, b: number) => a + b, 0) / rewards.length : 0,
        maxReward: rewards.length > 0 ? Math.max(...rewards) : 0,
        minReward: rewards.length > 0 ? Math.min(...rewards) : 0,
        avgSuccessRate: successRates.length > 0 ? successRates.reduce((a: number, b: number) => a + b, 0) / successRates.length : 0,
        avgEpisodeLength: episodeLengths.length > 0 ? episodeLengths.reduce((a: number, b: number) => a + b, 0) / episodeLengths.length : 0,
        avgCollisions: collisionCounts.length > 0 ? collisionCounts.reduce((a: number, b: number) => a + b, 0) / collisionCounts.length : 0,
        totalCollisions: collisionCounts.reduce((a: number, b: number) => a + b, 0),
        rewardsPerEpisode: rewards,
        successRates: successRates,
        collisionCounts: collisionCounts,
        episodeLengths: episodeLengths,
        ragStats: data.rag_stats || { totalQueries: 0, successfulRetrievals: 0, retrievalSuccessRate: 0 },
        finalAgentPerformance: data.agent_performance || []
      }
      
      setResults(resultData)
      
      // Save to localStorage for comparison
      saveToLocalStorage(resultData, data)
      
      setLoading(false)
    } catch (error) {
      toast.error('Failed to load results')
      setLoading(false)
    }
  }

  const saveToLocalStorage = (resultData: any, apiData: any) => {
    try {
      const now = new Date()
      const sessionInfo = JSON.parse(localStorage.getItem('currentSession') || '{}')
      
      // Use the API data config directly, fallback to sessionInfo if not available
      const config = apiData.config || sessionInfo.config || {}
      
      const storedResult = {
        sessionId,
        timestamp: now.toISOString(),
        date: now.toLocaleDateString(),
        time: now.toLocaleTimeString(),
        environmentType: config.environment_type || sessionInfo.environmentType || 'unknown',
        config: {
          numEpisodes: resultData.totalEpisodes,
          learningRate: config.learning_rate || 0.0003,
          gamma: config.gamma || 0.99,
          useRag: config.use_rag ?? false,
          useLlm: config.use_llm_shaping ?? false,
          gridSize: config.grid_size || 8,
          numAgents: config.num_agents || 2,
          numObstacles: config.num_obstacles || 0,
          hasDynamicObstacles: config.has_dynamic_obstacles || false
        },
        metrics: {
          totalEpisodes: resultData.totalEpisodes,
          avgReward: resultData.avgReward,
          maxReward: resultData.maxReward,
          minReward: resultData.minReward,
          avgSuccessRate: resultData.avgSuccessRate,
          avgEpisodeLength: resultData.avgEpisodeLength,
          avgCollisions: resultData.avgCollisions,
          totalCollisions: resultData.totalCollisions,
          rewardPerStep: resultData.avgEpisodeLength > 0 ? resultData.avgReward / resultData.avgEpisodeLength : 0,
          bestEpisode: resultData.rewardsPerEpisode.indexOf(resultData.maxReward) + 1,
          worstEpisode: resultData.rewardsPerEpisode.indexOf(resultData.minReward) + 1,
          episodesWithGoals: resultData.successRates.filter((s: number) => s >= 1.0).length
        },
        ragStats: resultData.ragStats,
        agentPerformance: resultData.finalAgentPerformance
      }
      
      const existing = JSON.parse(localStorage.getItem('training_results') || '[]')
      const updated = [...existing.filter((r: any) => r.sessionId !== sessionId), storedResult]
      localStorage.setItem('training_results', JSON.stringify(updated))
    } catch (error) {
      console.error('Error saving to localStorage:', error)
    }
  }

  const handleExportModel = () => {
    window.open(`/api/export/model/${sessionId}`, '_blank')
  }

  const handleExportReport = () => {
    window.open(`/api/export/report/${sessionId}`, '_blank')
  }

  const handleNewTraining = () => {
    router.push('/environments')
  }

  const handleInference = async () => {
    try {
      const response = await fetch(`/api/inference/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_episodes: 5 })
      })
      
      if (!response.ok) throw new Error('Inference failed')
      
      const data = await response.json()
      toast.success(`Inference complete! Avg Reward: ${data.avg_reward.toFixed(2)}`)
    } catch (error) {
      toast.error('Inference failed')
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!results) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">No results found for this session</p>
          <button
            onClick={() => router.push('/environments')}
            className="px-6 py-3 bg-blue-600 rounded-lg hover:bg-blue-700"
          >
            Start New Training
          </button>
        </div>
      </div>
    )
  }

  // Prepare chart data
  const rewardData = results.rewardsPerEpisode.map((r, i) => ({ episode: i + 1, reward: r }))
  const successData = results.successRates.map((s, i) => ({ episode: i + 1, success: s * 100 }))
  const collisionData = results.collisionCounts.map((c, i) => ({ episode: i + 1, collisions: c }))

  // Calculate best performing agent
  const bestAgent = results.finalAgentPerformance.reduce((best, current) => 
    current.avgReward > best.avgReward ? current : best, 
    results.finalAgentPerformance[0] || { agentId: 0, avgReward: 0 }
  )

  // Environment comparison data (placeholder for when multiple environments are trained)
  const comparisonData = [
    { name: 'Simple', success: 85, reward: 45.2, collisions: 2.1 },
    { name: 'Medium', success: 72, reward: 38.5, collisions: 4.3 },
    { name: 'Complex', success: 58, reward: 31.2, collisions: 6.8 },
    { name: 'Dynamic', success: 45, reward: 25.8, collisions: 8.2 }
  ]

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <button
            onClick={() => router.push('/environments')}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Environments
          </button>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-yellow-500/20 rounded-xl">
                <Trophy className="w-8 h-8 text-yellow-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">Training Results</h1>
                <p className="text-gray-400">Session: {sessionId}</p>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={handleNewTraining}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                New Training
              </button>
            </div>
          </div>
        </motion.div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { 
              label: 'Total Episodes', 
              value: results.totalEpisodes, 
              icon: <Target className="w-5 h-5 text-blue-400" />,
              color: 'blue'
            },
            { 
              label: 'Average Reward', 
              value: results.avgReward.toFixed(2), 
              icon: <TrendingUp className="w-5 h-5 text-green-400" />,
              color: 'green'
            },
            { 
              label: 'Success Rate', 
              value: `${(results.avgSuccessRate * 100).toFixed(1)}%`, 
              icon: <CheckCircle className="w-5 h-5 text-cyan-400" />,
              color: 'cyan'
            },
            { 
              label: 'Avg Collisions', 
              value: results.avgCollisions.toFixed(1), 
              icon: <AlertTriangle className="w-5 h-5 text-red-400" />,
              color: 'red'
            }
          ].map((metric, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`p-6 bg-slate-800/50 rounded-2xl border border-${metric.color}-500/20`}
            >
              <div className="flex items-center gap-2 mb-2">
                {metric.icon}
                <span className="text-sm text-gray-400">{metric.label}</span>
              </div>
              <div className="text-3xl font-bold">{metric.value}</div>
            </motion.div>
          ))}
        </div>

        {/* RAG & LLM Stats */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-purple-500/20"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Brain className="w-5 h-5 text-purple-400" />
              </div>
              <h2 className="text-lg font-semibold">RAG Memory Statistics</h2>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-2xl font-bold text-purple-400">{results.ragStats.totalQueries}</div>
                <div className="text-sm text-gray-400">Total Queries</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-400">{results.ragStats.successfulRetrievals}</div>
                <div className="text-sm text-gray-400">Successful</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-cyan-400">{(results.ragStats.retrievalSuccessRate * 100).toFixed(1)}%</div>
                <div className="text-sm text-gray-400">Success Rate</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-blue-500/20"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Database className="w-5 h-5 text-blue-400" />
              </div>
              <h2 className="text-lg font-semibold">Training Efficiency</h2>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-2xl font-bold text-blue-400">{results.avgEpisodeLength.toFixed(1)}</div>
                <div className="text-sm text-gray-400">Avg Steps</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-400">{results.maxReward.toFixed(2)}</div>
                <div className="text-sm text-gray-400">Max Reward</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-400">{results.minReward.toFixed(2)}</div>
                <div className="text-sm text-gray-400">Min Reward</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Charts Grid */}
        <div className="grid lg:grid-cols-2 gap-6 mb-8">
          {/* Reward Curve */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700"
          >
            <h3 className="text-lg font-semibold mb-4">Reward Curve</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rewardData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                />
                <Line type="monotone" dataKey="reward" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Success Rate */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700"
          >
            <h3 className="text-lg font-semibold mb-4">Success Rate Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={successData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <Area type="monotone" dataKey="success" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Collisions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700"
          >
            <h3 className="text-lg font-semibold mb-4">Collision Count per Episode</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={collisionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                />
                <Bar dataKey="collisions" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Episode Length Chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700"
          >
            <h3 className="text-lg font-semibold mb-4">Episode Length (Steps to Goal)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={results.episodeLengths.map((l, i) => ({ episode: i + 1, steps: l }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value: number) => `${value} steps`}
                />
                <Area type="monotone" dataKey="steps" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Evaluation Metrics & Graphs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-blue-400" />
            RL Evaluation Metrics
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Per-Agent Success Rate Comparison */}
            <div className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4">Per-Agent Success Rate</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={results.finalAgentPerformance.map(a => ({ 
                  agent: `Agent ${a.agentId}`, 
                  success: a.successRate * 100 
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="agent" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    formatter={(value: number) => `${value.toFixed(1)}%`}
                  />
                  <Bar dataKey="success" fill="#06b6d4" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Per-Agent Average Reward Comparison */}
            <div className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4">Per-Agent Average Reward</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={results.finalAgentPerformance.map(a => ({ 
                  agent: `Agent ${a.agentId}`, 
                  reward: a.avgReward 
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="agent" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    formatter={(value: number) => value.toFixed(2)}
                  />
                  <Bar dataKey="reward" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Reward Per Step Efficiency */}
            <div className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4">Reward Efficiency (Reward per Step)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={results.rewardsPerEpisode.map((r, i) => ({ 
                  episode: i + 1, 
                  efficiency: r / (results.episodeLengths[i] || 1)
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="episode" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    formatter={(value: number) => value.toFixed(3)}
                  />
                  <Line type="monotone" dataKey="efficiency" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Learning Progress (Moving Average) */}
            <div className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4">Learning Progress (20-Ep Moving Avg)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={(() => {
                  const window = 20;
                  const data = [];
                  for (let i = window - 1; i < results.rewardsPerEpisode.length; i++) {
                    const avg = results.rewardsPerEpisode.slice(i - window + 1, i + 1)
                      .reduce((a, b) => a + b, 0) / window;
                    data.push({ episode: i + 1, movingAvg: avg });
                  }
                  return data;
                })()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="episode" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    formatter={(value: number) => value.toFixed(2)}
                  />
                  <Line type="monotone" dataKey="movingAvg" stroke="#ec4899" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Agent Performance Cards */}
        {results.finalAgentPerformance.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 p-6 bg-slate-800/50 rounded-2xl border border-slate-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users className="w-5 h-5" />
              Per-Agent Summary
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {results.finalAgentPerformance.map((agent) => (
                <div key={agent.agentId} className="p-4 bg-slate-900/50 rounded-xl">
                  <div className="text-lg font-semibold mb-2">Agent {agent.agentId}</div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Avg Reward:</span>
                      <span className="text-green-400">{agent.avgReward.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Success Rate:</span>
                      <span className="text-cyan-400">{(agent.successRate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Collisions:</span>
                      <span className="text-red-400">{agent.collisions}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Final Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-8 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-2xl border border-blue-500/20 text-center"
        >
          <h2 className="text-2xl font-bold mb-4">Training Complete!</h2>
          <p className="text-gray-300 max-w-2xl mx-auto mb-6">
            Your multi-agent system has been trained using MAPPO with LLM reward shaping and RAG memory. 
            The agents achieved an average success rate of {(results.avgSuccessRate * 100).toFixed(1)}% 
            over {results.totalEpisodes} episodes.
          </p>
          <div className="flex justify-center gap-4">
            <button
              onClick={() => router.push('/environments')}
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-xl font-semibold"
            >
              Train New Environment
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
