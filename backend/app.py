import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import io
import base64

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# Import our modules
from env import SimpleEnvironment, MediumEnvironment, ComplexEnvironment, DynamicEnvironment
from agents import MAPPOTrainer
from memory import RAGMemory, LLMRewardShaper
from database import MongoDBManager

app = FastAPI(title="Multi-Agent Warehouse Navigation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions = {}
trainers = {}
environments = {}
rag_memories = {}
reward_shapers = {}
db_manager = None

# Environment registry
ENVIRONMENTS = {
    'simple': SimpleEnvironment,
    'medium': MediumEnvironment,
    'complex': ComplexEnvironment,
    'dynamic': DynamicEnvironment
}

# Pydantic models
class EnvironmentConfig(BaseModel):
    environment_type: str
    num_episodes: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    use_rag: bool = True
    use_llm_shaping: bool = True

class TrainingConfig(BaseModel):
    session_id: str
    action: str  # 'start', 'pause', 'stop'

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    global db_manager
    # Connect to MongoDB Atlas
    try:
        db_manager = MongoDBManager()
        await db_manager.initialize_indexes()
        print("MongoDB Atlas connected successfully")
    except Exception as e:
        print(f"MongoDB Atlas connection failed: {e}")
        db_manager = None

@app.get("/")
async def root():
    return {"message": "Multi-Agent Warehouse Navigation API", "status": "running"}

@app.get("/api/environments")
async def get_environments():
    """Get available environment configurations"""
    return {
        "environments": [
            {
                "id": "simple",
                "name": "Simple Environment",
                "description": "5x5 grid, 2 agents, no obstacles",
                "grid_size": 5,
                "num_agents": 2,
                "num_obstacles": 0,
                "difficulty": "easy"
            },
            {
                "id": "medium",
                "name": "Medium Environment",
                "description": "8x8 grid, 3 agents, static obstacles",
                "grid_size": 8,
                "num_agents": 3,
                "num_obstacles": 8,
                "difficulty": "medium"
            },
            {
                "id": "complex",
                "name": "Complex Environment",
                "description": "10x10 grid, 4 agents, narrow corridors",
                "grid_size": 10,
                "num_agents": 4,
                "num_obstacles": 15,
                "difficulty": "hard"
            },
            {
                "id": "dynamic",
                "name": "Dynamic Environment",
                "description": "8x8 grid, 4 agents, moving obstacles",
                "grid_size": 8,
                "num_agents": 4,
                "num_obstacles": 6,
                "dynamic_obstacles": True,
                "difficulty": "hard"
            }
        ]
    }

@app.post("/api/environment/initialize")
async def initialize_environment(config: EnvironmentConfig):
    """Initialize environment and training session"""
    session_id = str(uuid.uuid4())
    
    # Create environment
    env_class = ENVIRONMENTS.get(config.environment_type, SimpleEnvironment)
    env = env_class()
    
    # Initialize RAG memory if enabled
    rag_memory = RAGMemory() if config.use_rag else None
    
    # Initialize LLM reward shaper if enabled
    reward_shaper = LLMRewardShaper(use_simulation=True) if config.use_llm_shaping else None
    
    # Initialize MAPPO trainer with fixed obs_dim=8 (feature vector representation)
    # Observation features: [agent_x, agent_y, goal_x, goal_y, nearest_obstacle_dist, nearest_agent_dist, collision_flag, remaining_steps]
    trainer = MAPPOTrainer(
        env=env,
        num_agents=env.num_agents,
        obs_dim=8,  # Fixed 8-dimensional feature vector
        action_dim=5,  # Up, Down, Left, Right, Stay
        lr=config.learning_rate,
        gamma=config.gamma,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Store in global state
    environments[session_id] = env
    trainers[session_id] = trainer
    rag_memories[session_id] = rag_memory
    reward_shapers[session_id] = reward_shaper
    
    active_sessions[session_id] = {
        'config': config.dict(),
        'status': 'initialized',
        'created_at': datetime.now().isoformat(),
        'current_episode': 0
    }
    
    # Get initial state
    obs, info = env.reset()
    
    return {
        "session_id": session_id,
        "status": "initialized",
        "environment": {
            "type": config.environment_type,
            "grid_size": env.grid_size,
            "num_agents": env.num_agents,
            "num_obstacles": env.num_obstacles
        },
        "initial_state": {
            "agent_positions": env.agent_positions,
            "goal_positions": env.goal_positions,
            "obstacle_positions": env.obstacle_positions,
            "dynamic_obstacles": env.dynamic_obstacle_positions if hasattr(env, 'dynamic_obstacle_positions') else []
        }
    }

@app.post("/api/training/control")
async def control_training(config: TrainingConfig):
    """Control training session"""
    session_id = config.session_id
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if config.action == 'start':
        active_sessions[session_id]['status'] = 'training'
    elif config.action == 'pause':
        active_sessions[session_id]['status'] = 'paused'
    elif config.action == 'stop':
        active_sessions[session_id]['status'] = 'stopped'
    
    return {
        "session_id": session_id,
        "status": active_sessions[session_id]['status']
    }

def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(v) for v in obj]
    return obj


@app.websocket("/ws/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time training updates"""
    await websocket.accept()
    print(f"WebSocket connected for session {session_id}")
    
    if session_id not in active_sessions:
        print(f"Session {session_id} not found")
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    trainer = trainers.get(session_id)
    env = environments.get(session_id)
    rag_memory = rag_memories.get(session_id)
    reward_shaper = reward_shapers.get(session_id)
    
    print(f"Trainer: {trainer is not None}, Env: {env is not None}")
    
    if not trainer or not env:
        print("Training components not initialized")
        await websocket.send_json({"error": "Training components not initialized"})
        await websocket.close()
        return
    
    config = active_sessions[session_id]['config']
    num_episodes = config.get('num_episodes', 1000)
    
    completed_episodes = 0
    
    try:
        print(f"Starting training loop for {num_episodes} episodes")
        for episode in range(num_episodes):
            print(f"Starting episode {episode + 1} of {num_episodes} (completed: {completed_episodes})")
            
            # Check if training was stopped
            if active_sessions[session_id]['status'] == 'stopped':
                print(f"Training stopped by user at episode {episode + 1}")
                break
            
            # Skip if paused
            if active_sessions[session_id]['status'] != 'training':
                print(f"Training paused at episode {episode + 1}")
                await asyncio.sleep(0.1)
                continue
            
            # Run episode with error handling
            try:
                print("Resetting environment...")
                obs, info = env.reset()
                print(f"Environment reset complete. Obs shape: {len(obs)}, {obs[0].shape if len(obs) > 0 else 'N/A'}")
                episode_reward = 0
                episode_collisions = 0
                step_count = 0
                trajectory_states = []
                trajectory_actions = []
                trajectory_rewards = []
                
                done = False
                print(f"Starting episode loop, max_steps: {env.max_steps}")
                while not done and step_count < env.max_steps:
                    actions = []
                    log_probs = []
                    
                    print(f"Step {step_count}: Getting actions...")
                    for i, agent in enumerate(trainer.agents):
                        action, log_prob, _ = agent.select_action(obs[i])
                        actions.append(action[0])
                        log_probs.append(log_prob[0])
                    print(f"Actions: {actions}")
                    
                    # Get RAG guidance if enabled and USE IT to improve rewards
                    rag_guidance = None
                    rag_action_bonuses = {}  # Action -> bonus mapping
                    if rag_memory:
                        rag_guidance = rag_memory.get_guidance_from_memory(
                            obs[0], env.agent_positions, env.goal_positions, env.obstacle_positions
                        )
                        
                        # ACTUALLY USE RAG: Create action bonuses from retrieved trajectories
                        if rag_guidance and 'suggested_actions' in rag_guidance:
                            for suggestion in rag_guidance['suggested_actions']:
                                action = suggestion.get('action')
                                confidence = suggestion.get('confidence', 0.5)
                                if action is not None and confidence > 0.6:  # High confidence only
                                    # Convert action name to index if needed
                                    action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4}
                                    if isinstance(action, str) and action.lower() in action_map:
                                        action_idx = action_map[action.lower()]
                                    else:
                                        action_idx = int(action)
                                    # Bonus proportional to confidence (0.5 to 2.0)
                                    rag_action_bonuses[action_idx] = confidence * 2.0
                    
                    # Compute joint value using centralized critic (ALL agents' observations)
                    import torch
                    obs_tensors = [torch.FloatTensor(o).unsqueeze(0).to(trainer.device) for o in obs]
                    with torch.no_grad():
                        joint_value = trainer.critic(obs_tensors).cpu().numpy()[0, 0]
                    print(f"Joint value from critic: {joint_value}")
                    
                    # Step environment
                    print(f"Stepping environment...")
                    next_obs, rewards, terminated, truncated, info = env.step(actions)
                    print(f"Step complete. Rewards: {rewards}")
                    done = terminated or truncated
                    
                    # Apply LLM reward shaping AND RAG bonuses if enabled
                    shaped_rewards = []
                    for i, (agent, reward) in enumerate(zip(trainer.agents, rewards)):
                        # Add RAG guidance bonus for taking suggested actions
                        rag_bonus = 0.0
                        if rag_action_bonuses and actions[i] in rag_action_bonuses:
                            rag_bonus = rag_action_bonuses[actions[i]]
                            print(f"  Agent {i}: RAG bonus {rag_bonus:.2f} for action {actions[i]}")
                        
                        if reward_shaper:
                            # Calculate ACTUALLY nearby agents (within 3 squares)
                            agent_pos = env.agent_positions[i]
                            nearby_agents = []
                            for j in range(env.num_agents):
                                if j != i:
                                    other_pos = env.agent_positions[j]
                                    dist = abs(agent_pos[0] - other_pos[0]) + abs(agent_pos[1] - other_pos[1])
                                    if dist <= 3:  # Only truly nearby agents
                                        nearby_agents.append(j)
                            
                            # Calculate ACTUALLY nearby obstacles (within 3 squares)
                            nearby_obstacles = []
                            for obs_pos in env.obstacle_positions:
                                dist = abs(agent_pos[0] - obs_pos[0]) + abs(agent_pos[1] - obs_pos[1])
                                if dist <= 3:
                                    nearby_obstacles.append(obs_pos)
                            
                            context = {
                                'agent_position': agent_pos,
                                'goal_position': env.goal_positions[i],
                                'nearby_agents': nearby_agents,  # Only truly nearby
                                'nearby_obstacles': nearby_obstacles,  # Only truly nearby
                                'collision_occurred': info.get('collisions', 0) > episode_collisions,
                                'path_efficiency': 1.0 - step_count / env.max_steps
                            }
                            shaped_reward, reward_info = reward_shaper.shape_reward(
                                reward + rag_bonus, i, obs[i], actions[i], next_obs[i], context
                            )
                            shaped_rewards.append(shaped_reward)
                        else:
                            # Even without LLM, add RAG bonuses
                            shaped_rewards.append(reward + rag_bonus)
                    
                    # Store in RAG memory
                    trajectory_states.append(obs[0])
                    trajectory_actions.append(actions[0])
                    trajectory_rewards.append(shaped_rewards[0])
                    
                    # Store transitions with joint value from critic
                    for i, agent in enumerate(trainer.agents):
                        agent.store_transition(
                            obs[i], actions[i], shaped_rewards[i], joint_value, log_probs[i], done, next_obs[i]
                        )
                    
                    episode_reward += sum(shaped_rewards)
                    episode_collisions = info.get('collisions', 0)
                    step_count += 1
                    obs = next_obs
                    
                    # Send update every 5 steps
                    if step_count % 5 == 0:
                        await websocket.send_json(convert_to_serializable({
                            'type': 'step_update',
                            'episode': episode,
                            'step': step_count,
                            'agent_positions': env.agent_positions,
                            'goals_reached': env.goals_reached,
                            'dynamic_obstacles': env.dynamic_obstacle_positions if hasattr(env, 'dynamic_obstacle_positions') else [],
                            'current_reward': episode_reward,
                            'collisions': episode_collisions,
                            'rag_guidance': rag_guidance
                        }))
                    
                    await asyncio.sleep(0.01)  # Small delay for visualization
                
                # Update agents after episode
                update_info = {}
                for i, agent in enumerate(trainer.agents):
                    agent_info = agent.update()
                    update_info[f'agent_{i}'] = agent_info
                
                # Store trajectory in RAG memory
                if rag_memory and len(trajectory_states) > 10:
                    rag_memory.store_trajectory(
                        trajectory_states, trajectory_actions, trajectory_rewards,
                        success=info.get('full_success', all(env.goals_reached)),
                        environment_type=config['environment_type']
                    )
                
                # Get correct success rate from environment info
                episode_success_rate = info.get('success_rate', sum(env.goals_reached) / env.num_agents)
                
                # Store episode data
                episode_data = {
                    'episode': episode,
                    'total_reward': episode_reward,
                    'steps': step_count,
                    'collisions': episode_collisions,
                    'success_rate': episode_success_rate,
                    'success': info.get('full_success', episode_success_rate >= 1.0),  # Full success boolean for DB
                    'update_info': update_info
                }
                
                trainer.episode_rewards.append(episode_reward)
                trainer.episode_lengths.append(step_count)
                trainer.collision_counts.append(episode_collisions)
                trainer.success_rates.append(episode_success_rate)
                
                # Track per-agent metrics
                for i in range(env.num_agents):
                    agent_reward = sum(shaped_rewards[i] for _ in range(step_count)) / max(step_count, 1)
                    trainer.per_agent_rewards[i].append(agent_reward)
                    trainer.per_agent_success[i].append(1 if env.goals_reached[i] else 0)
                    # Approximate per-agent collisions (shared for now)
                    trainer.per_agent_collisions[i].append(episode_collisions)
                
            except Exception as episode_error:
                print(f"ERROR in episode {episode + 1}: {episode_error}")
                import traceback
                print(traceback.format_exc())
                print(f"Continuing to next episode...")
                continue  # Skip to next episode instead of stopping training
            
            # Calculate overall metrics across all episodes
            overall_avg_reward = float(np.mean(trainer.episode_rewards))
            overall_avg_success_rate = float(np.mean(trainer.success_rates))
            overall_total_collisions = int(sum(trainer.collision_counts))
            overall_avg_collisions = float(np.mean(trainer.collision_counts))
            
            # Send episode summary
            await websocket.send_json(convert_to_serializable({
                'type': 'episode_complete',
                'episode': episode,
                'data': episode_data,
                'metrics': {
                    # Rolling 10-episode averages
                    'avg_reward_10': float(np.mean(trainer.episode_rewards[-10:])) if len(trainer.episode_rewards) >= 10 else float(episode_reward),
                    'avg_success_rate_10': float(np.mean(trainer.success_rates[-10:])) if len(trainer.success_rates) >= 10 else float(sum(env.goals_reached) / env.num_agents),
                    'avg_collisions_10': float(np.mean(trainer.collision_counts[-10:])) if len(trainer.collision_counts) >= 10 else float(episode_collisions),
                    # Overall metrics (all episodes)
                    'overall_avg_reward': overall_avg_reward,
                    'overall_avg_success_rate': overall_avg_success_rate,
                    'overall_total_collisions': overall_total_collisions,
                    'overall_avg_collisions': overall_avg_collisions,
                    'episodes_completed': len(trainer.episode_rewards)
                }
            }))
            
            active_sessions[session_id]['current_episode'] = episode
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0 and episode > 0:
                checkpoint_path = f"./models/checkpoint_{session_id}_ep{episode}.pt"
                os.makedirs("./models", exist_ok=True)
                trainer.save(checkpoint_path)
            
            completed_episodes += 1
            await asyncio.sleep(0.05)
        
        print(f"Training loop ended. Completed {completed_episodes} of {num_episodes} episodes")
        
        # Training complete - calculate comprehensive overall metrics
        total_reward = sum(trainer.episode_rewards) if trainer.episode_rewards else 0
        total_collisions = int(sum(trainer.collision_counts)) if trainer.collision_counts else 0
        
        await websocket.send_json(convert_to_serializable({
            'type': 'training_complete',
            'final_stats': {
                'total_episodes': num_episodes,
                # Overall averages across all episodes
                'overall_avg_reward': float(np.mean(trainer.episode_rewards)),
                'overall_avg_success_rate': float(np.mean(trainer.success_rates)),
                'overall_avg_collisions': float(np.mean(trainer.collision_counts)),
                # Totals
                'total_reward': float(total_reward),
                'total_collisions': total_collisions,
                # Best and worst performance
                'best_episode_reward': float(max(trainer.episode_rewards)) if trainer.episode_rewards else 0,
                'worst_episode_reward': float(min(trainer.episode_rewards)) if trainer.episode_rewards else 0,
                # Success tracking
                'episodes_with_all_goals_reached': int(sum(1 for s in trainer.success_rates if s >= 1.0)),
                'overall_goals_reached_percentage': float(np.mean(trainer.success_rates) * 100)
            }
        }))
        
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        import traceback
        print(f"Error in WebSocket for session {session_id}:")
        print(traceback.format_exc())
        try:
            await websocket.send_json(convert_to_serializable({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass
    finally:
        # Save final model
        if trainer:
            final_path = f"./models/final_{session_id}.pt"
            os.makedirs("./models", exist_ok=True)
            trainer.save(final_path)
            
            # Save RAG memory
            if rag_memory:
                rag_memory.save_memories()

@app.get("/api/training/stats/{session_id}")
async def get_training_stats(session_id: str):
    """Get training statistics"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    trainer = trainers.get(session_id)
    if not trainer:
        return {"error": "Trainer not found"}
    
    # Get RAG stats if available (convert snake_case to camelCase for frontend)
    rag_memory = rag_memories.get(session_id)
    if rag_memory:
        raw_stats = rag_memory.get_stats()
        rag_stats = {
            "totalQueries": raw_stats.get('trajectory_queries', 0) + raw_stats.get('collision_queries', 0) + 
                          raw_stats.get('success_queries', 0) + raw_stats.get('experience_queries', 0),
            "successfulRetrievals": raw_stats.get('successful_retrievals', 0),
            "retrievalSuccessRate": raw_stats.get('retrieval_success_rate', 0),
            "totalMemories": raw_stats.get('total_memories', 0)
        }
    else:
        rag_stats = {
            "totalQueries": 0,
            "successfulRetrievals": 0,
            "retrievalSuccessRate": 0,
            "totalMemories": 0
        }
    
    # Build per-agent performance data
    agent_performance = []
    for i in range(trainer.num_agents):
        avg_reward = float(np.mean(trainer.per_agent_rewards[i])) if trainer.per_agent_rewards[i] else 0
        success_rate = float(np.mean(trainer.per_agent_success[i])) if trainer.per_agent_success[i] else 0
        total_collisions = int(sum(trainer.per_agent_collisions[i])) if trainer.per_agent_collisions[i] else 0
        agent_performance.append({
            "agentId": i,
            "avgReward": avg_reward,
            "successRate": success_rate,
            "collisions": total_collisions
        })
    
    return {
        "session_id": session_id,
        "current_episode": active_sessions[session_id]['current_episode'],
        "status": active_sessions[session_id]['status'],
        "rewards": trainer.episode_rewards,
        "episode_lengths": trainer.episode_lengths,
        "success_rates": trainer.success_rates,
        "collision_counts": trainer.collision_counts,
        "rag_stats": rag_stats,
        "agent_performance": agent_performance
    }

@app.get("/api/rag/stats/{session_id}")
async def get_rag_stats(session_id: str):
    """Get RAG memory statistics"""
    rag_memory = rag_memories.get(session_id)
    if not rag_memory:
        return {"error": "RAG memory not enabled for this session"}
    
    return rag_memory.get_stats()

@app.post("/api/inference/{session_id}")
async def run_inference(session_id: str, num_episodes: int = 5):
    """Run inference with trained model"""
    trainer = trainers.get(session_id)
    env = environments.get(session_id)
    
    if not trainer or not env:
        raise HTTPException(status_code=404, detail="Session or components not found")
    
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        trajectory = []
        
        done = False
        while not done and step_count < env.max_steps:
            actions = []
            for i, agent in enumerate(trainer.agents):
                action, _, _ = agent.select_action(obs[i], deterministic=True)
                actions.append(action[0])
            
            trajectory.append({
                'step': step_count,
                'agent_positions': env.agent_positions.copy(),
                'actions': actions.copy()
            })
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_reward += sum(rewards)
            step_count += 1
            obs = next_obs
        
        results.append({
            'episode': episode,
            'total_reward': episode_reward,
            'steps': step_count,
            'success': all(env.goals_reached),
            'collisions': info.get('collisions', 0),
            'trajectory': trajectory
        })
    
    return {
        "session_id": session_id,
        "num_episodes": num_episodes,
        "results": results,
        "avg_reward": np.mean([r['total_reward'] for r in results]),
        "success_rate": sum([r['success'] for r in results]) / num_episodes
    }

@app.get("/api/export/model/{session_id}")
async def export_model(session_id: str):
    """Export trained model"""
    model_path = f"./models/final_{session_id}.pt"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        model_path,
        media_type='application/octet-stream',
        filename=f'mappo_model_{session_id}.pt'
    )

@app.get("/api/export/report/{session_id}")
async def export_report(session_id: str):
    """Export training report"""
    trainer = trainers.get(session_id)
    
    if not trainer:
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    # Generate report data
    report = {
        "session_id": session_id,
        "config": active_sessions.get(session_id, {}).get('config', {}),
        "training_stats": {
            "total_episodes": len(trainer.episode_rewards),
            "avg_reward": np.mean(trainer.episode_rewards),
            "max_reward": max(trainer.episode_rewards) if trainer.episode_rewards else 0,
            "min_reward": min(trainer.episode_rewards) if trainer.episode_rewards else 0,
            "avg_success_rate": np.mean(trainer.success_rates) if trainer.success_rates else 0,
            "avg_episode_length": np.mean(trainer.episode_lengths) if trainer.episode_lengths else 0,
            "total_collisions": sum(trainer.collision_counts)
        },
        "rewards_per_episode": trainer.episode_rewards,
        "success_rates": trainer.success_rates,
        "collision_counts": trainer.collision_counts,
        "episode_lengths": trainer.episode_lengths
    }
    
    # Convert to JSON
    json_data = json.dumps(report, indent=2)
    
    return StreamingResponse(
        io.BytesIO(json_data.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=training_report_{session_id}.json"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
