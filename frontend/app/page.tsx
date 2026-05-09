'use client'

import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  Bot, 
  Map, 
  Brain, 
  Database, 
  ArrowRight, 
  Cpu,
  Network,
  Target,
  BarChart3
} from 'lucide-react'

export default function LandingPage() {
  const router = useRouter()

  const features = [
    {
      icon: <Bot className="w-8 h-8" />,
      title: "MAPPO Algorithm",
      description: "Multi-Agent Proximal Policy Optimization with centralized critic for coordinated navigation"
    },
    {
      icon: <Brain className="w-8 h-8" />,
      title: "LLM Reward Shaping",
      description: "Intelligent reward shaping using Large Language Model feedback for better decisions"
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: "RAG Memory",
      description: "Retrieval-Augmented Generation for storing and retrieving successful navigation patterns"
    },
    {
      icon: <Network className="w-8 h-8" />,
      title: "Multi-Agent Coordination",
      description: "Seamless coordination between multiple agents to avoid collisions and optimize paths"
    },
    {
      icon: <Map className="w-8 h-8" />,
      title: "4 Environments",
      description: "Train and test on Simple, Medium, Complex, and Dynamic warehouse environments"
    },
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: "Real-time Analytics",
      description: "Live dashboard with reward curves, success rates, collision metrics, and more"
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse delay-2000" />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex justify-between items-center px-8 py-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Cpu className="w-8 h-8 text-blue-400" />
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            MAPPO Navigator
          </span>
        </div>
        <div className="flex gap-6 text-sm text-gray-300">
          <span>v1.0.0</span>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 container mx-auto px-8 pt-12 pb-20">
        <div className="text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/20 border border-blue-500/30 mb-6">
              <Target className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-blue-200">Multi-Agent RL System</span>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
          >
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400 bg-clip-text text-transparent">
              Multi-Agent
            </span>
            <br />
            <span className="text-white">Warehouse Navigation</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl text-gray-300 mb-4 max-w-2xl mx-auto"
          >
            Advanced reinforcement learning system using MAPPO algorithm with 
            LLM reward shaping and RAG memory for intelligent multi-robot navigation.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex justify-center gap-4 mt-8"
          >
            <button
              onClick={() => router.push('/environments')}
              className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl font-semibold text-lg flex items-center gap-3 hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300"
            >
              Start Simulation
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
            <a
              href="#features"
              className="px-8 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl font-semibold text-lg hover:bg-white/20 transition-all duration-300"
            >
              Learn More
            </a>
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div
          id="features"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-24 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
              className="p-6 bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl hover:bg-white/10 transition-all duration-300 group"
            >
              <div className="p-3 bg-blue-500/20 rounded-xl w-fit mb-4 group-hover:bg-blue-500/30 transition-colors">
                <div className="text-blue-400">{feature.icon}</div>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">{feature.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Problem Statement */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="mt-20 p-8 bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-3xl border border-blue-500/20"
        >
          <h2 className="text-2xl font-bold mb-6 text-center">Problem Statement</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-blue-300">Challenge</h3>
              <p className="text-gray-300 leading-relaxed">
                Multiple robots must navigate from start positions to goals in a shared warehouse 
                environment while avoiding collisions with each other, static obstacles, and dynamic 
                obstacles. Traditional approaches struggle with coordination and path optimization 
                in complex scenarios.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-300">Our Solution</h3>
              <p className="text-gray-300 leading-relaxed">
                We combine Multi-Agent PPO (MAPPO) with centralized critics for coordination, 
                LLM-based reward shaping for intelligent navigation decisions, and RAG memory 
                to learn from past successful trajectories and avoid historical collision patterns.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.4 }}
          className="mt-16 text-center"
        >
          <h2 className="text-2xl font-bold mb-8">Technology Stack</h2>
          <div className="flex flex-wrap justify-center gap-4">
            {['Python', 'PyTorch', 'FastAPI', 'Next.js', 'MongoDB', 'Three.js', 'Tailwind CSS', 'WebSocket'].map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-sm text-gray-300 border border-white/10"
              >
                {tech}
              </span>
            ))}
          </div>
        </motion.div>
      </main>
    </div>
  )
}
