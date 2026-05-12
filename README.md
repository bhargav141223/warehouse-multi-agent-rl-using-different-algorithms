# Multi-Agent Warehouse Navigation System

## MAPPO + Heterogeneous Training + LLM Reward Shaping + RAG Memory

A comprehensive full-stack application for multi-agent warehouse navigation using Multi-Agent Proximal Policy Optimization (MAPPO) with support for heterogeneous training (different algorithms for different agents), Large Language Model (LLM) reward shaping, and Retrieval-Augmented Generation (RAG) memory.

## Features

### Core Algorithm
- **Heterogeneous Multi-Agent Training**: Train agents with different algorithms (MAPPO, DQN, A2C, PPO, SAC)
- **MAPPO (Multi-Agent PPO)**: Multi-agent reinforcement learning with centralized critic
- **DQN (Deep Q-Network)**: Value-based learning with experience replay
- **A2C (Advantage Actor-Critic)**: Synchronous actor-critic with advantage estimation
- **PPO (Proximal Policy Optimization)**: Independent PPO for individual agents
- **SAC (Soft Actor-Critic)**: Maximum entropy reinforcement learning
- **LLM Reward Shaping**: Intelligent reward feedback for better navigation decisions
- **RAG Memory**: Store and retrieve successful trajectories and collision avoidance patterns
- **Real-time Training**: Live visualization of agent movements and training metrics

### Environments
1. **Simple**: 5x5 grid, 2 agents, no obstacles
2. **Medium**: 8x8 grid, 3 agents, static obstacles
3. **Complex**: 10x10 grid, 4 agents, narrow corridors
4. **Dynamic**: 8x8 grid, 4 agents, moving obstacles

### Dashboard Features
- Real-time warehouse animation (HTML5 Canvas)
- Live reward curves, success rates, collision metrics
- RAG memory statistics
- Agent coordination visualization
- Model export and report generation
- Inference mode for testing trained models

## Tech Stack

### Backend
- Python 3.8+
- FastAPI (Web framework)
- PyTorch (Deep learning)
- Gymnasium (RL environments)
- MongoDB (Data storage)
- WebSocket (Real-time communication)

### Frontend
- Next.js 14 (React framework)
- TypeScript
- Tailwind CSS (Styling)
- Recharts (Data visualization)
- Framer Motion (Animations)
- Lucide React (Icons)

## Project Structure

```
warehouse-navigation/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── requirements.txt       # Python dependencies
│   ├── env/                   # Warehouse environments
│   │   ├── warehouse_env.py   # Base environment
│   │   ├── simple_env.py      # Simple environment
│   │   ├── medium_env.py      # Medium environment
│   │   ├── complex_env.py     # Complex environment
│   │   └── dynamic_env.py     # Dynamic environment
│   ├── agents/                # RL algorithm implementations
│   │   ├── mappo.py          # MAPPO trainer & agents
│   │   ├── dqn_agent.py      # DQN agent implementation
│   │   ├── a2c_agent.py      # A2C agent implementation
│   │   ├── ppo_agent.py      # Independent PPO agent
│   │   ├── sac_agent.py      # SAC agent implementation
│   │   ├── heterogeneous_trainer.py  # Heterogeneous training manager
│   │   └── networks.py        # Actor-Critic networks
│   ├── memory/                # RAG & LLM modules
│   │   ├── rag_memory.py      # RAG memory implementation
│   │   └── llm_reward_shaping.py # LLM reward shaping
│   └── database/
│       └── mongodb.py         # MongoDB manager
├── frontend/
│   ├── app/                   # Next.js app router
│   │   ├── page.tsx           # Landing page
│   │   ├── environments/      # Environment selection
│   │   ├── training/          # Training dashboard
│   │   └── results/           # Results page
│   ├── package.json           # NPM dependencies
│   └── tailwind.config.ts     # Tailwind configuration
├── models/                    # Saved model checkpoints
├── results/                   # Training results
└── data/                      # Vector database storage
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 18 or higher
- MongoDB (optional, for data persistence)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Running the Application

### Option 1: Using Python Scripts

```bash
# Terminal 1: Start Backend
python start_backend.py

# Terminal 2: Start Frontend
python start_frontend.py
```

### Option 2: Manual Start

```bash
# Terminal 1: Backend
cd backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Option 3: Direct Commands

```bash
# Start Backend (from project root)
cd backend && venv\Scripts\python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Start Frontend (from project root)
cd frontend && npm run dev
```

## Accessing the Application

Once both servers are running:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Usage Guide

1. **Landing Page**: Learn about the project and click "Start Simulation"

2. **Environment Selection**: 
   - Choose one of 4 environments (Simple, Medium, Complex, Dynamic)
   - Configure training parameters (episodes, learning rate, etc.)
   - Enable/disable RAG memory and LLM reward shaping
   - **NEW**: Enable heterogeneous training and select algorithms for each agent
   - Click "Initialize & Start"

3. **Training Dashboard**:
   - Watch real-time warehouse animation
   - Monitor live charts (reward, success rate, collisions)
   - View RAG memory statistics
   - Control training (start, pause, stop)
   - Export model or report after training

4. **Results Page**:
   - View comprehensive training statistics
   - Compare performance across environments
   - Download trained models
   - Export detailed reports
   - Run inference with trained agents

## MAPPO Implementation Details

### Algorithm Components
- **Actor Network**: Each agent has its own policy network
- **Centralized Critic**: Single critic network evaluates joint state
- **PPO Clipping**: Prevents large policy updates
- **GAE (Generalized Advantage Estimation)**: Reduces variance in advantage estimation
- **Entropy Regularization**: Encourages exploration

## Heterogeneous Training

### Supported Algorithms
- **MAPPO**: Multi-agent PPO with centralized critic (best for coordination)
- **DQN**: Deep Q-Network with experience replay (value-based, off-policy)
- **A2C**: Advantage Actor-Critic (synchronous, on-policy)
- **PPO**: Independent PPO (decentralized, on-policy)
- **SAC**: Soft Actor-Critic (maximum entropy, off-policy)

### How to Use Heterogeneous Training

When initializing training, set `heterogeneous: true` and provide algorithm assignments:

```json
{
  "heterogeneous": true,
  "algorithms": ["mappo", "dqn", "a2c", "ppo"]
}
```

This assigns:
- Agent 0: MAPPO
- Agent 1: DQN
- Agent 2: A2C
- Agent 3: PPO

You can use any combination of algorithms including SAC.

### Benefits of Heterogeneous Training
- Algorithm diversity for robust performance
- Compare different algorithms in same environment
- Leverage strengths of each algorithm
- More flexible experimental setup

### Actions
- 0: Up
- 1: Down  
- 2: Left
- 3: Right
- 4: Stay

### Reward Structure
- Goal reached: +10
- Move closer to goal: +2
- Collision: -5
- Wandering: -1
- Good exploration: +0.3

## LLM Reward Shaping

The LLM module provides:
- Dense reward generation based on navigation progress
- Collision penalty reasoning
- Path optimization feedback
- Coordination hints for multiple agents

## RAG Memory

The RAG system stores:
- Successful trajectories
- Collision avoidance patterns
- Per-agent experiences

Retrieval uses:
- Vector embeddings for state representation
- Cosine similarity for matching
- Top-k retrieval for relevant experiences

## MongoDB Integration

Stores:
- Training history
- Episode data
- Collision logs
- Model checkpoints
- Training metrics

## API Endpoints

### Environment Management
- `GET /api/environments` - List available environments
- `POST /api/environment/initialize` - Initialize training session (supports heterogeneous training)
  - Parameters:
    - `environment_type`: Environment name (simple, medium, complex, dynamic)
    - `num_episodes`: Number of training episodes
    - `learning_rate`: Learning rate for optimization
    - `gamma`: Discount factor
    - `use_rag`: Enable RAG memory
    - `use_llm_shaping`: Enable LLM reward shaping
    - `heterogeneous`: Enable heterogeneous training (boolean)
    - `algorithms`: List of algorithms for each agent (e.g., ["mappo", "dqn", "a2c", "ppo"])

### Training Control
- `POST /api/training/control` - Start/pause/stop training
- `GET /api/training/stats/{session_id}` - Get training statistics
- `WebSocket /ws/training/{session_id}` - Real-time training updates

### Export & Inference
- `GET /api/export/model/{session_id}` - Download trained model
- `GET /api/export/report/{session_id}` - Download training report
- `POST /api/inference/{session_id}` - Run inference

### RAG Statistics
- `GET /api/rag/stats/{session_id}` - Get RAG memory statistics

## Troubleshooting

### Backend Issues

**MongoDB Connection Failed**
- MongoDB is optional; the application will continue without it
- To enable MongoDB, install and start MongoDB service

**CUDA/GPU Issues**
- PyTorch will automatically use CPU if GPU is not available
- No changes needed; training will work on CPU

**Port 8000 Already in Use**
```bash
# Find and kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**Module Not Found Errors**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Port 3000 Already in Use**
```bash
# Next.js will automatically try next available port
# Or set custom port:
npm run dev -- --port 3001
```

### Common Issues

**WebSocket Connection Failed**
- Ensure backend is running on port 8000
- Check firewall settings
- Try refreshing the page

**Training Not Starting**
- Verify backend is running
- Check browser console for errors
- Ensure environment was properly initialized

## Performance Tips

1. **For faster training**: Reduce number of episodes or use Simple environment
2. **For better results**: Enable RAG memory and LLM reward shaping
3. **For visualization**: Use smaller grid sizes for clearer animation

## Development

### Adding New Environments

1. Create new file in `backend/env/`
2. Inherit from `WarehouseEnvironment`
3. Override `_initialize_positions()` for custom layouts
4. Add to `ENVIRONMENTS` dict in `app.py`

### Adding New Algorithms

1. Create new agent file in `backend/agents/` (e.g., `new_algorithm_agent.py`)
2. Implement required methods: `select_action()`, `store_transition()`, `update()`, `save()`, `load()`
3. Add to `HeterogeneousTrainer` in `heterogeneous_trainer.py`
4. Add algorithm name to supported algorithms list

### Customizing Reward Function

Edit `llm_reward_shaping.py`:
- Modify `_calculate_dense_reward()` for custom rewards
- Update `_generate_reasoning()` for different feedback
- Adjust weights in `shape_reward()` method

### Extending Dashboard

Add new charts in `frontend/app/training/[sessionId]/page.tsx`:
- Add chart data preparation
- Include new Recharts component
- Style with Tailwind classes

## License

This project is created for educational purposes as a Final Year Project.

## Acknowledgments

- MAPPO algorithm based on "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- RAG implementation inspired by LangChain
- LLM reward shaping concept from recent research in RL + LLMs
