'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  Grid3X3, 
  Blocks, 
  AlertTriangle, 
  Zap, 
  ArrowRight, 
  ArrowLeft,
  Settings,
  Users,
  Box,
  BarChart3
} from 'lucide-react'
import toast from 'react-hot-toast'

interface Environment {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  gridSize: number
  numAgents: number
  numObstacles: number
  hasDynamicObstacles: boolean
  difficulty: string
  color: string
}

const environments: Environment[] = [
  {
    id: 'simple',
    name: 'Simple Environment',
    description: 'Perfect for initial training and validation. 5x5 grid with 2 agents and no obstacles.',
    icon: <Grid3X3 className="w-8 h-8" />,
    gridSize: 5,
    numAgents: 2,
    numObstacles: 0,
    hasDynamicObstacles: false,
    difficulty: 'Easy',
    color: 'from-green-500 to-emerald-600'
  },
  {
    id: 'medium',
    name: 'Medium Environment',
    description: 'Introduces static obstacles. 8x8 grid with 3 agents and 8 obstacles.',
    icon: <Blocks className="w-8 h-8" />,
    gridSize: 8,
    numAgents: 3,
    numObstacles: 8,
    hasDynamicObstacles: false,
    difficulty: 'Medium',
    color: 'from-blue-500 to-indigo-600'
  },
  {
    id: 'complex',
    name: 'Complex Environment',
    description: 'Narrow corridors challenge. 10x10 grid with 4 agents and narrow pathways.',
    icon: <AlertTriangle className="w-8 h-8" />,
    gridSize: 10,
    numAgents: 4,
    numObstacles: 15,
    hasDynamicObstacles: false,
    difficulty: 'Hard',
    color: 'from-orange-500 to-red-600'
  },
  {
    id: 'dynamic',
    name: 'Dynamic Environment',
    description: 'Moving obstacles add unpredictability. 8x8 grid with 4 agents.',
    icon: <Zap className="w-8 h-8" />,
    gridSize: 8,
    numAgents: 4,
    numObstacles: 6,
    hasDynamicObstacles: true,
    difficulty: 'Hard',
    color: 'from-purple-500 to-pink-600'
  }
]

export default function EnvironmentSelection() {
  const router = useRouter()
  const [selectedEnv, setSelectedEnv] = useState<string | null>(null)
  const [isInitializing, setIsInitializing] = useState(false)
  const [config, setConfig] = useState({
    numEpisodes: 1000,
    learningRate: 3e-4,
    gamma: 0.99,
    useRAG: true,
    useLLMShaping: true
  })

  const handleInitialize = async () => {
    if (!selectedEnv) {
      toast.error('Please select an environment')
      return
    }

    setIsInitializing(true)

    try {
      const response = await fetch('/api/environment/initialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          environment_type: selectedEnv,
          num_episodes: config.numEpisodes,
          learning_rate: config.learningRate,
          gamma: config.gamma,
          use_rag: config.useRAG,
          use_llm_shaping: config.useLLMShaping
        })
      })

      if (!response.ok) {
        throw new Error('Failed to initialize environment')
      }

      const data = await response.json()
      toast.success('Environment initialized!')
      
      // Store session info
      localStorage.setItem('currentSession', JSON.stringify({
        sessionId: data.session_id,
        environmentType: selectedEnv,
        config: data.environment
      }))

      // Navigate to training page
      router.push(`/training/${data.session_id}`)
    } catch (error) {
      toast.error('Failed to initialize environment. Make sure the backend is running.')
      setIsInitializing(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </button>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">Select Environment</h1>
              <p className="text-gray-400">Choose the warehouse configuration for training your agents</p>
            </div>
            <button
              onClick={() => router.push('/results/compare')}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-xl flex items-center gap-2 transition-colors"
            >
              <BarChart3 className="w-5 h-5" />
              Compare All Results
            </button>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Environment Cards */}
          <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
            {environments.map((env, index) => (
              <motion.div
                key={env.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setSelectedEnv(env.id)}
                className={`p-6 rounded-2xl border-2 cursor-pointer transition-all duration-300 ${
                  selectedEnv === env.id
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                }`}
              >
                <div className={`p-3 rounded-xl bg-gradient-to-r ${env.color} w-fit mb-4`}>
                  {env.icon}
                </div>
                
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xl font-semibold">{env.name}</h3>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    env.difficulty === 'Easy' ? 'bg-green-500/20 text-green-400' :
                    env.difficulty === 'Medium' ? 'bg-blue-500/20 text-blue-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {env.difficulty}
                  </span>
                </div>
                
                <p className="text-gray-400 text-sm mb-4">{env.description}</p>
                
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div className="flex items-center gap-2 text-gray-300">
                    <Grid3X3 className="w-4 h-4" />
                    <span>{env.gridSize}×{env.gridSize}</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-300">
                    <Users className="w-4 h-4" />
                    <span>{env.numAgents} agents</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-300">
                    <Box className="w-4 h-4" />
                    <span>{env.numObstacles} obstacles</span>
                  </div>
                </div>

                {env.hasDynamicObstacles && (
                  <div className="mt-3 flex items-center gap-2 text-orange-400 text-sm">
                    <Zap className="w-4 h-4" />
                    <span>Dynamic Obstacles</span>
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Configuration Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 h-fit"
          >
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-5 h-5 text-blue-400" />
              <h2 className="text-xl font-semibold">Training Configuration</h2>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Episodes</label>
                <input
                  type="number"
                  value={config.numEpisodes}
                  onChange={(e) => setConfig({...config, numEpisodes: parseInt(e.target.value)})}
                  className="w-full px-4 py-2 bg-gray-900 rounded-lg border border-gray-700 text-white focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.learningRate}
                  onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                  className="w-full px-4 py-2 bg-gray-900 rounded-lg border border-gray-700 text-white focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Gamma (Discount Factor)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={config.gamma}
                  onChange={(e) => setConfig({...config, gamma: parseFloat(e.target.value)})}
                  className="w-full px-4 py-2 bg-gray-900 rounded-lg border border-gray-700 text-white focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                <div>
                  <p className="font-medium">RAG Memory</p>
                  <p className="text-xs text-gray-400">Store successful trajectories</p>
                </div>
                <button
                  onClick={() => setConfig({...config, useRAG: !config.useRAG})}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    config.useRAG ? 'bg-blue-500' : 'bg-gray-600'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    config.useRAG ? 'translate-x-6' : 'translate-x-1'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                <div>
                  <p className="font-medium">LLM Reward Shaping</p>
                  <p className="text-xs text-gray-400">Intelligent reward feedback</p>
                </div>
                <button
                  onClick={() => setConfig({...config, useLLMShaping: !config.useLLMShaping})}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    config.useLLMShaping ? 'bg-blue-500' : 'bg-gray-600'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    config.useLLMShaping ? 'translate-x-6' : 'translate-x-1'
                  }`} />
                </button>
              </div>

              <button
                onClick={handleInitialize}
                disabled={!selectedEnv || isInitializing}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl font-semibold flex items-center justify-center gap-2 hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isInitializing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Initializing...
                  </>
                ) : (
                  <>
                    Initialize & Start
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
