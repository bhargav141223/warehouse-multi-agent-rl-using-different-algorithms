'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ArrowLeft, Download, Trash2, BarChart3, Calendar,
  Target, TrendingUp, CheckCircle, AlertTriangle, Activity,
  Clock, Users, Brain, Database, FileSpreadsheet
} from 'lucide-react'
import toast from 'react-hot-toast'

interface TrainingResult {
  sessionId: string
  timestamp: string
  date: string
  time: string
  environmentType: string
  config: {
    numEpisodes: number
    learningRate: number
    gamma: number
    useRag: boolean
    useLlm: boolean
    gridSize: number
    numAgents: number
    numObstacles: number
    hasDynamicObstacles: boolean
  }
  metrics: {
    totalEpisodes: number
    avgReward: number
    maxReward: number
    minReward: number
    avgSuccessRate: number
    avgEpisodeLength: number
    avgCollisions: number
    totalCollisions: number
    rewardPerStep: number
    bestEpisode: number
    worstEpisode: number
    episodesWithGoals: number
  }
  ragStats: {
    totalQueries: number
    successfulRetrievals: number
    retrievalSuccessRate: number
  }
  agentPerformance: {
    agentId: number
    avgReward: number
    successRate: number
    collisions: number
  }[]
}

export default function CompareResultsPage() {
  const router = useRouter()
  const [results, setResults] = useState<TrainingResult[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResults, setSelectedResults] = useState<string[]>([])

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = () => {
    try {
      // Load from localStorage
      const stored = localStorage.getItem('training_results')
      if (stored) {
        const parsed = JSON.parse(stored)
        // Filter out old results that don't have proper RAG/LLM flags
        const filtered = parsed.filter((r: any) => r.config && 'useRag' in r.config && 'useLlm' in r.config)
        setResults(filtered)
        // Update localStorage with filtered results
        localStorage.setItem('training_results', JSON.stringify(filtered))
      }
      setLoading(false)
    } catch (error) {
      console.error('Error loading results:', error)
      setLoading(false)
    }
  }

  const saveCurrentResult = (sessionId: string, data: any, config: any) => {
    const now = new Date()
    const date = now.toLocaleDateString()
    const time = now.toLocaleTimeString()
    
    const rewards = data.rewards || []
    const successRates = data.success_rates || []
    const collisionCounts = data.collision_counts || []
    const episodeLengths = data.episode_lengths || []
    
    const totalReward = rewards.reduce((a: number, b: number) => a + b, 0)
    const avgReward = rewards.length > 0 ? totalReward / rewards.length : 0
    const maxReward = rewards.length > 0 ? Math.max(...rewards) : 0
    const minReward = rewards.length > 0 ? Math.min(...rewards) : 0
    const avgSuccessRate = successRates.length > 0 ? successRates.reduce((a: number, b: number) => a + b, 0) / successRates.length : 0
    const avgEpisodeLength = episodeLengths.length > 0 ? episodeLengths.reduce((a: number, b: number) => a + b, 0) / episodeLengths.length : 0
    const totalCollisions = collisionCounts.reduce((a: number, b: number) => a + b, 0)
    const avgCollisions = collisionCounts.length > 0 ? totalCollisions / collisionCounts.length : 0
    const rewardPerStep = avgEpisodeLength > 0 ? avgReward / avgEpisodeLength : 0
    
    const bestEpisode = rewards.indexOf(maxReward) + 1
    const worstEpisode = rewards.indexOf(minReward) + 1
    const episodesWithGoals = successRates.filter((s: number) => s >= 1.0).length
    
    const newResult: TrainingResult = {
      sessionId,
      timestamp: now.toISOString(),
      date,
      time,
      environmentType: config.environment_type || 'unknown',
      config: {
        numEpisodes: config.num_episodes || 100,
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
        totalEpisodes: rewards.length,
        avgReward,
        maxReward,
        minReward,
        avgSuccessRate,
        avgEpisodeLength,
        avgCollisions,
        totalCollisions,
        rewardPerStep,
        bestEpisode,
        worstEpisode,
        episodesWithGoals
      },
      ragStats: data.rag_stats || {
        totalQueries: 0,
        successfulRetrievals: 0,
        retrievalSuccessRate: 0
      },
      agentPerformance: data.agent_performance || []
    }
    
    const existing = JSON.parse(localStorage.getItem('training_results') || '[]')
    const updated = [...existing.filter((r: TrainingResult) => r.sessionId !== sessionId), newResult]
    localStorage.setItem('training_results', JSON.stringify(updated))
    setResults(updated)
  }

  const exportToCSV = () => {
    if (results.length === 0) {
      toast.error('No results to export')
      return
    }
    
    const headers = [
      'Session ID',
      'Date',
      'Time',
      'Environment Type',
      'Grid Size',
      'Num Agents',
      'Num Obstacles',
      'Dynamic Obstacles',
      'Episodes',
      'Learning Rate',
      'Gamma',
      'Use RAG',
      'Use LLM',
      'Avg Reward',
      'Max Reward',
      'Min Reward',
      'Avg Success Rate (%)',
      'Avg Episode Length',
      'Avg Collisions',
      'Total Collisions',
      'Reward Per Step',
      'Best Episode',
      'Worst Episode',
      'Episodes with Goals',
      'RAG Queries',
      'RAG Success',
      'RAG Success Rate (%)'
    ].join(',')
    
    const rows = results.map(r => [
      r.sessionId,
      r.date,
      r.time,
      r.environmentType,
      r.config.gridSize,
      r.config.numAgents,
      r.config.numObstacles,
      r.config.hasDynamicObstacles ? 'Yes' : 'No',
      r.metrics.totalEpisodes,
      r.config.learningRate,
      r.config.gamma,
      r.config.useRag ? 'Yes' : 'No',
      r.config.useLlm ? 'Yes' : 'No',
      r.metrics.avgReward.toFixed(2),
      r.metrics.maxReward.toFixed(2),
      r.metrics.minReward.toFixed(2),
      (r.metrics.avgSuccessRate * 100).toFixed(1),
      r.metrics.avgEpisodeLength.toFixed(1),
      r.metrics.avgCollisions.toFixed(1),
      r.metrics.totalCollisions,
      r.metrics.rewardPerStep.toFixed(3),
      r.metrics.bestEpisode,
      r.metrics.worstEpisode,
      r.metrics.episodesWithGoals,
      r.ragStats.totalQueries,
      r.ragStats.successfulRetrievals,
      (r.ragStats.retrievalSuccessRate * 100).toFixed(1)
    ].join(','))
    
    const csv = [headers, ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training_results_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success('CSV exported successfully!')
  }

  const deleteResult = (sessionId: string) => {
    const updated = results.filter(r => r.sessionId !== sessionId)
    localStorage.setItem('training_results', JSON.stringify(updated))
    setResults(updated)
    toast.success('Result deleted')
  }

  const clearAll = () => {
    if (confirm('Are you sure you want to delete all results?')) {
      localStorage.removeItem('training_results')
      setResults([])
      toast.success('All results cleared')
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-[1800px] mx-auto">
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
              <div className="p-3 bg-blue-500/20 rounded-xl">
                <BarChart3 className="w-8 h-8 text-blue-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">Compare All Training Results</h1>
                <p className="text-gray-400">View and compare all training sessions</p>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={exportToCSV}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2"
                disabled={results.length === 0}
              >
                <FileSpreadsheet className="w-4 h-4" />
                Export CSV
              </button>
              {results.length > 0 && (
                <button
                  onClick={clearAll}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All
                </button>
              )}
            </div>
          </div>
        </motion.div>

        {results.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <BarChart3 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No training results found</p>
            <p className="text-gray-500 mt-2">Train some agents to see results here</p>
            <button
              onClick={() => router.push('/environments')}
              className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-xl"
            >
              Start Training
            </button>
          </motion.div>
        ) : (
          <>
            {/* Summary Cards */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6"
            >
              {[
                { label: 'Total Sessions', value: results.length, icon: <Target className="w-4 h-4" />, color: 'blue' },
                { label: 'Avg Success Rate', value: `${(results.reduce((a, r) => a + r.metrics.avgSuccessRate, 0) / results.length * 100).toFixed(1)}%`, icon: <CheckCircle className="w-4 h-4" />, color: 'green' },
                { label: 'Avg Reward', value: (results.reduce((a, r) => a + r.metrics.avgReward, 0) / results.length).toFixed(2), icon: <TrendingUp className="w-4 h-4" />, color: 'cyan' },
                { label: 'Avg Collisions', value: (results.reduce((a, r) => a + r.metrics.avgCollisions, 0) / results.length).toFixed(1), icon: <AlertTriangle className="w-4 h-4" />, color: 'red' },
                { label: 'Total RAG Queries', value: results.reduce((a, r) => a + r.ragStats.totalQueries, 0), icon: <Database className="w-4 h-4" />, color: 'purple' },
                { label: 'With LLM/RAG', value: results.filter(r => r.config.useRag && r.config.useLlm).length, icon: <Brain className="w-4 h-4" />, color: 'pink' }
              ].map((stat, idx) => (
                <div key={idx} className={`p-4 bg-slate-800/50 rounded-xl border border-${stat.color}-500/20`}>
                  <div className={`flex items-center gap-2 text-${stat.color}-400 mb-2`}>
                    {stat.icon}
                    <span className="text-xs font-medium">{stat.label}</span>
                  </div>
                  <div className="text-2xl font-bold">{stat.value}</div>
                </div>
              ))}
            </motion.div>

            {/* Comparison Table */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-slate-800/50 rounded-2xl border border-slate-700 overflow-hidden"
            >
              <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Training Results Comparison Table
                </h2>
                <span className="text-sm text-gray-400">{results.length} sessions</span>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900/50 text-gray-400">
                    <tr>
                      <th className="px-4 py-3 text-left">Date/Time</th>
                      <th className="px-4 py-3 text-left">Environment</th>
                      <th className="px-4 py-3 text-center">Episodes</th>
                      <th className="px-4 py-3 text-center">Grid</th>
                      <th className="px-4 py-3 text-center">Agents</th>
                      <th className="px-4 py-3 text-center">RAG</th>
                      <th className="px-4 py-3 text-center">LLM</th>
                      <th className="px-4 py-3 text-right">Avg Reward</th>
                      <th className="px-4 py-3 text-right">Success Rate</th>
                      <th className="px-4 py-3 text-right">Avg Collisions</th>
                      <th className="px-4 py-3 text-right">Episode Length</th>
                      <th className="px-4 py-3 text-right">Reward/Step</th>
                      <th className="px-4 py-3 text-right">RAG Queries</th>
                      <th className="px-4 py-3 text-center">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700">
                    {results.map((result, idx) => (
                      <tr key={result.sessionId} className="hover:bg-slate-800/30">
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-gray-500" />
                            <div>
                              <div className="text-white">{result.date}</div>
                              <div className="text-xs text-gray-500">{result.time}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            result.environmentType === 'simple' ? 'bg-green-500/20 text-green-400' :
                            result.environmentType === 'medium' ? 'bg-blue-500/20 text-blue-400' :
                            result.environmentType === 'complex' ? 'bg-orange-500/20 text-orange-400' :
                            'bg-purple-500/20 text-purple-400'
                          }`}>
                            {result.environmentType.charAt(0).toUpperCase() + result.environmentType.slice(1)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center">{result.metrics.totalEpisodes}</td>
                        <td className="px-4 py-3 text-center">{result.config.gridSize}x{result.config.gridSize}</td>
                        <td className="px-4 py-3 text-center">{result.config.numAgents}</td>
                        <td className="px-4 py-3 text-center">
                          {result.config.useRag ? <span className="text-green-400">Yes</span> : <span className="text-red-400">No</span>}
                        </td>
                        <td className="px-4 py-3 text-center">
                          {result.config.useLlm ? <span className="text-green-400">Yes</span> : <span className="text-red-400">No</span>}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">{result.metrics.avgReward.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right">
                          <span className={`font-mono ${result.metrics.avgSuccessRate >= 0.6 ? 'text-green-400' : result.metrics.avgSuccessRate >= 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
                            {(result.metrics.avgSuccessRate * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono">{result.metrics.avgCollisions.toFixed(1)}</td>
                        <td className="px-4 py-3 text-right font-mono">{result.metrics.avgEpisodeLength.toFixed(1)}</td>
                        <td className="px-4 py-3 text-right font-mono">{result.metrics.rewardPerStep.toFixed(3)}</td>
                        <td className="px-4 py-3 text-right font-mono">{result.ragStats.totalQueries}</td>
                        <td className="px-4 py-3 text-center">
                          <div className="flex items-center gap-2 justify-center">
                            <button
                              onClick={() => router.push(`/results/${result.sessionId}`)}
                              className="p-1.5 bg-blue-600 hover:bg-blue-700 rounded text-xs"
                            >
                              View
                            </button>
                            <button
                              onClick={() => deleteResult(result.sessionId)}
                              className="p-1.5 bg-red-600 hover:bg-red-700 rounded"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>

            {/* Environment Comparison */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-8 bg-slate-800/50 rounded-2xl border border-slate-700 p-6"
            >
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Users className="w-5 h-5 text-cyan-400" />
                Performance by Environment Type
              </h2>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['simple', 'medium', 'complex', 'dynamic'].map(envType => {
                  const envResults = results.filter(r => r.environmentType === envType)
                  if (envResults.length === 0) return null
                  
                  const avgReward = envResults.reduce((a, r) => a + r.metrics.avgReward, 0) / envResults.length
                  const avgSuccess = envResults.reduce((a, r) => a + r.metrics.avgSuccessRate, 0) / envResults.length
                  const avgCollisions = envResults.reduce((a, r) => a + r.metrics.avgCollisions, 0) / envResults.length
                  
                  return (
                    <div key={envType} className="bg-slate-900/50 rounded-xl p-4">
                      <h3 className="font-medium capitalize mb-3 text-gray-300">{envType}</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-500">Sessions:</span>
                          <span className="text-white">{envResults.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Avg Reward:</span>
                          <span className="text-green-400">{avgReward.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Success Rate:</span>
                          <span className="text-cyan-400">{(avgSuccess * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Collisions:</span>
                          <span className="text-red-400">{avgCollisions.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </motion.div>

            {/* RL Evaluation Metrics Reference */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-8 bg-slate-800/50 rounded-2xl border border-slate-700 p-6"
            >
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                RL Evaluation Parameters Reference
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-green-400 mb-1">Episode Reward</div>
                  <div className="text-gray-400 text-xs">Sum of all rewards (env + LLM) received per episode. Range: -50 to +30. Higher is better.</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-cyan-400 mb-1">Success Rate</div>
                  <div className="text-gray-400 text-xs">Percentage of episodes where agents reach goals. 0.8-1.0 = Excellent.</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-red-400 mb-1">Collision Rate</div>
                  <div className="text-gray-400 text-xs">Average collisions per episode. Lower is better. 0.0-0.1 = Excellent.</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-blue-400 mb-1">Episode Length</div>
                  <div className="text-gray-400 text-xs">Steps to completion. 20-40 steps = Excellent (Simple env).</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-yellow-400 mb-1">Reward Per Step</div>
                  <div className="text-gray-400 text-xs">Efficiency metric. Higher values mean agents accumulate rewards faster.</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="font-medium text-purple-400 mb-1">RAG Memory Stats</div>
                  <div className="text-gray-400 text-xs">Retrieval-Augmented Generation queries and successful retrievals from memory.</div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </div>
    </div>
  )
}
