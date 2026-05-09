from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import numpy as np

class MongoDBManager:
    """MongoDB manager for training data and model checkpoints"""
    
    def __init__(self, connection_string: str = "mongodb+srv://bhargavreddy1223_db_user:1234>@cluster0.edm1qrz.mongodb.net/?appName=Cluster0", 
                 database_name: str = "warehouse_navigation"):
        self.client = AsyncIOMotorClient(connection_string, serverSelectionTimeoutMS=5000)
        self.db = self.client[database_name]
        
        # Collections
        self.training_history = self.db.training_history
        self.episodes = self.db.episodes
        self.collision_logs = self.db.collision_logs
        self.model_checkpoints = self.db.model_checkpoints
        self.metrics = self.db.metrics
        self.rewards = self.db.rewards
        
    async def initialize_indexes(self):
        """Create indexes for efficient queries"""
        # Training history indexes
        await self.training_history.create_index([("session_id", ASCENDING)])
        await self.training_history.create_index([("timestamp", DESCENDING)])
        
        # Episodes indexes
        await self.episodes.create_index([("session_id", ASCENDING)])
        await self.episodes.create_index([("episode_number", ASCENDING)])
        await self.episodes.create_index([("environment_type", ASCENDING)])
        
        # Collision logs indexes
        await self.collision_logs.create_index([("session_id", ASCENDING)])
        await self.collision_logs.create_index([("timestamp", DESCENDING)])
        
        # Model checkpoints indexes
        await self.model_checkpoints.create_index([("session_id", ASCENDING)])
        await self.model_checkpoints.create_index([("episode", DESCENDING)])
    
    async def save_training_session(self, session_data: Dict) -> str:
        """Save training session metadata"""
        session_data['timestamp'] = datetime.now()
        result = await self.training_history.insert_one(session_data)
        return str(result.inserted_id)
    
    async def save_episode(self, session_id: str, episode_data: Dict):
        """Save episode data"""
        episode_data['session_id'] = session_id
        episode_data['timestamp'] = datetime.now()
        
        # Convert numpy arrays to lists for JSON serialization
        episode_data = self._convert_numpy(episode_data)
        
        await self.episodes.insert_one(episode_data)
    
    async def save_collision(self, session_id: str, collision_data: Dict):
        """Save collision log"""
        collision_data['session_id'] = session_id
        collision_data['timestamp'] = datetime.now()
        collision_data = self._convert_numpy(collision_data)
        await self.collision_logs.insert_one(collision_data)
    
    async def save_model_checkpoint(
        self, 
        session_id: str, 
        episode: int, 
        model_data: bytes,
        metrics: Dict
    ):
        """Save model checkpoint"""
        checkpoint = {
            'session_id': session_id,
            'episode': episode,
            'model_data': model_data,
            'metrics': self._convert_numpy(metrics),
            'timestamp': datetime.now()
        }
        await self.model_checkpoints.insert_one(checkpoint)
    
    async def save_metrics(self, session_id: str, episode: int, metrics: Dict):
        """Save training metrics"""
        metrics_data = {
            'session_id': session_id,
            'episode': episode,
            'metrics': self._convert_numpy(metrics),
            'timestamp': datetime.now()
        }
        await self.metrics.insert_one(metrics_data)
    
    async def save_reward(self, session_id: str, episode: int, agent_id: int, 
                          step: int, reward: float, reward_type: str):
        """Save reward information"""
        reward_data = {
            'session_id': session_id,
            'episode': episode,
            'agent_id': agent_id,
            'step': step,
            'reward': reward,
            'reward_type': reward_type,
            'timestamp': datetime.now()
        }
        await self.rewards.insert_one(reward_data)
    
    async def get_training_history(self, session_id: Optional[str] = None, 
                                   limit: int = 100) -> List[Dict]:
        """Get training history"""
        query = {'session_id': session_id} if session_id else {}
        cursor = self.training_history.find(query).sort('timestamp', DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def get_episodes(self, session_id: str, limit: int = 1000) -> List[Dict]:
        """Get episodes for a session"""
        cursor = self.episodes.find({'session_id': session_id}).sort('episode_number', ASCENDING).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def get_best_episodes(self, session_id: str, metric: str = 'total_reward', 
                                top_k: int = 10) -> List[Dict]:
        """Get best performing episodes"""
        cursor = self.episodes.find({'session_id': session_id}).sort(metric, DESCENDING).limit(top_k)
        return await cursor.to_list(length=top_k)
    
    async def get_collision_stats(self, session_id: str) -> Dict:
        """Get collision statistics"""
        pipeline = [
            {'$match': {'session_id': session_id}},
            {'$group': {
                '_id': '$collision_type',
                'count': {'$sum': 1},
                'avg_severity': {'$avg': '$severity'}
            }}
        ]
        cursor = self.collision_logs.aggregate(pipeline)
        results = await cursor.to_list(length=100)
        
        return {
            'total_collisions': sum(r['count'] for r in results),
            'by_type': {r['_id']: r['count'] for r in results},
            'avg_severity_by_type': {r['_id']: r['avg_severity'] for r in results}
        }
    
    async def get_metrics_timeseries(self, session_id: str, 
                                     metric_names: List[str]) -> Dict[str, List]:
        """Get time series data for specified metrics"""
        cursor = self.metrics.find({'session_id': session_id}).sort('episode', ASCENDING)
        documents = await cursor.to_list(length=10000)
        
        timeseries = {name: [] for name in metric_names}
        timeseries['episodes'] = []
        
        for doc in documents:
            timeseries['episodes'].append(doc['episode'])
            for name in metric_names:
                timeseries[name].append(doc['metrics'].get(name, 0))
        
        return timeseries
    
    async def get_latest_checkpoint(self, session_id: str) -> Optional[Dict]:
        """Get latest model checkpoint"""
        cursor = self.model_checkpoints.find({'session_id': session_id}).sort('episode', DESCENDING).limit(1)
        checkpoints = await cursor.to_list(length=1)
        return checkpoints[0] if checkpoints else None
    
    async def get_rag_stats(self, session_id: str) -> Dict:
        """Get RAG memory statistics"""
        # This would be populated from training logs
        cursor = self.episodes.find({'session_id': session_id})
        episodes = await cursor.to_list(length=10000)
        
        total_retrievals = sum(e.get('rag_retrievals', 0) for e in episodes)
        successful_retrievals = sum(e.get('successful_retrievals', 0) for e in episodes)
        
        return {
            'total_queries': total_retrievals,
            'successful_retrievals': successful_retrievals,
            'success_rate': successful_retrievals / max(total_retrievals, 1)
        }
    
    def _convert_numpy(self, data: Any) -> Any:
        """Convert numpy types to Python native types"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif isinstance(data, dict):
            return {k: self._convert_numpy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy(item) for item in data]
        return data
    
    async def export_session_data(self, session_id: str) -> Dict:
        """Export all data for a session"""
        episodes = await self.get_episodes(session_id, limit=10000)
        collision_stats = await self.get_collision_stats(session_id)
        
        return {
            'session_id': session_id,
            'episodes': episodes,
            'collision_stats': collision_stats,
            'total_episodes': len(episodes)
        }
    
    async def get_environment_comparison(self, session_ids: List[str]) -> Dict:
        """Compare performance across environments"""
        comparison = {}
        
        for session_id in session_ids:
            cursor = self.episodes.find({'session_id': session_id})
            episodes = await cursor.to_list(length=1000)
            
            if episodes:
                # Calculate metrics
                rewards = [e.get('total_reward', 0) for e in episodes]
                success_flags = [e.get('success', False) for e in episodes]
                collision_counts = [e.get('collisions', 0) for e in episodes]
                steps = [e.get('steps', 0) for e in episodes]
                
                env_type = episodes[0].get('environment_type', 'unknown')
                
                comparison[env_type] = {
                    'success_rate': sum(success_flags) / len(success_flags),
                    'avg_reward': sum(rewards) / len(rewards),
                    'avg_collisions': sum(collision_counts) / len(collision_counts),
                    'avg_steps': sum(steps) / len(steps)
                }
        
        return comparison
    
    async def close(self):
        """Close database connection"""
        self.client.close()
