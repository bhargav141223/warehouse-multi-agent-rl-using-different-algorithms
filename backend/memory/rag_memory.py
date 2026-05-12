import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """Simple vector store for embeddings using cosine similarity"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension))
        self.metadata: List[Dict] = []
    
    def add(self, vector: np.ndarray, metadata: Dict):
        """Add a vector with metadata"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Expected vector of dimension {self.dimension}, got {vector.shape[0]}")
        
        vector = vector.reshape(1, -1)
        if self.vectors.size == 0:
            self.vectors = vector
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        self.metadata.append(metadata)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for similar vectors using cosine similarity"""
        if self.vectors.size == 0:
            return []
        
        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((float(similarities[idx]), self.metadata[idx]))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.vectors = data['vectors']
        self.metadata = data['metadata']
        self.dimension = data['dimension']


class RAGMemory:
    """Retrieval-Augmented Generation Memory for Multi-Agent RL"""
    
    def __init__(
        self,
        embedding_dim: int = 384,
        memory_dir: str = "./data/vector_db",
        similarity_threshold: float = 0.7
    ):
        self.embedding_dim = embedding_dim
        self.memory_dir = memory_dir
        self.similarity_threshold = similarity_threshold
        
        # Create memory directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize vector stores for different memory types
        self.trajectory_store = VectorStore(embedding_dim)
        self.collision_store = VectorStore(embedding_dim)
        self.success_store = VectorStore(embedding_dim)
        self.experience_store = VectorStore(embedding_dim)
        
        # Statistics
        self.access_stats = {
            'trajectory_queries': 0,
            'collision_queries': 0,
            'success_queries': 0,
            'experience_queries': 0,
            'successful_retrievals': 0
        }
        
        # Try to load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load existing memory stores"""
        stores = [
            (self.trajectory_store, 'trajectories.pkl'),
            (self.collision_store, 'collisions.pkl'),
            (self.success_store, 'successes.pkl'),
            (self.experience_store, 'experiences.pkl')
        ]
        
        for store, filename in stores:
            filepath = os.path.join(self.memory_dir, filename)
            if os.path.exists(filepath):
                try:
                    store.load(filepath)
                    print(f"Loaded {filename} with {len(store.metadata)} entries")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def save_memories(self):
        """Save all memory stores"""
        stores = [
            (self.trajectory_store, 'trajectories.pkl'),
            (self.collision_store, 'collisions.pkl'),
            (self.success_store, 'successes.pkl'),
            (self.experience_store, 'experiences.pkl')
        ]
        
        for store, filename in stores:
            filepath = os.path.join(self.memory_dir, filename)
            store.save(filepath)
    
    def _state_to_embedding(self, state: np.ndarray) -> np.ndarray:
        """Convert state to embedding vector"""
        # Flatten and normalize state
        flat_state = state.flatten()
        
        # Pad or truncate to embedding dimension
        if len(flat_state) > self.embedding_dim:
            embedding = flat_state[:self.embedding_dim]
        else:
            embedding = np.zeros(self.embedding_dim)
            embedding[:len(flat_state)] = flat_state
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def store_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        success: bool,
        environment_type: str
    ):
        """Store a successful trajectory"""
        # Use final state as key
        final_state_embedding = self._state_to_embedding(states[-1])
        
        trajectory_data = {
            'states': [s.tolist() for s in states],
            'actions': actions,
            'rewards': rewards,
            'total_reward': sum(rewards),
            'success': success,
            'environment_type': environment_type,
            'timestamp': datetime.now().isoformat(),
            'length': len(states)
        }
        
        self.trajectory_store.add(final_state_embedding, trajectory_data)
    
    def store_collision_experience(
        self,
        state: np.ndarray,
        action: int,
        collision_type: str,
        agents_involved: List[int]
    ):
        """Store collision experience for avoidance learning"""
        state_embedding = self._state_to_embedding(state)
        
        collision_data = {
            'state': state.tolist(),
            'action': action,
            'collision_type': collision_type,
            'agents_involved': agents_involved,
            'timestamp': datetime.now().isoformat()
        }
        
        self.collision_store.add(state_embedding, collision_data)
    
    def store_success_pattern(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        strategy: str,
        num_agents: int,
        avg_reward: float
    ):
        """Store successful navigation patterns"""
        # Combine initial and goal state
        combined = np.concatenate([initial_state.flatten(), goal_state.flatten()])
        state_embedding = self._state_to_embedding(combined)
        
        success_data = {
            'initial_state': initial_state.tolist(),
            'goal_state': goal_state.tolist(),
            'strategy': strategy,
            'num_agents': num_agents,
            'avg_reward': avg_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        self.success_store.add(state_embedding, success_data)
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        agent_id: int,
        episode_id: int
    ):
        """Store general experience"""
        state_embedding = self._state_to_embedding(state)
        
        experience_data = {
            'state': state.tolist(),
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist(),
            'agent_id': agent_id,
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experience_store.add(state_embedding, experience_data)
    
    def retrieve_similar_trajectories(
        self,
        current_state: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """Retrieve similar successful trajectories"""
        self.access_stats['trajectory_queries'] += 1
        
        state_embedding = self._state_to_embedding(current_state)
        results = self.trajectory_store.search(state_embedding, top_k)
        
        # Filter by similarity threshold
        filtered = [meta for sim, meta in results if sim >= self.similarity_threshold]
        
        if filtered:
            self.access_stats['successful_retrievals'] += len(filtered)
        
        return filtered
    
    def retrieve_collision_avoidance(
        self,
        current_state: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve collision avoidance patterns"""
        self.access_stats['collision_queries'] += 1
        
        state_embedding = self._state_to_embedding(current_state)
        results = self.collision_store.search(state_embedding, top_k)
        
        filtered = [meta for sim, meta in results if sim >= self.similarity_threshold]
        
        if filtered:
            self.access_stats['successful_retrievals'] += len(filtered)
        
        return filtered
    
    def retrieve_success_patterns(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """Retrieve similar success patterns"""
        self.access_stats['success_queries'] += 1
        
        combined = np.concatenate([initial_state.flatten(), goal_state.flatten()])
        state_embedding = self._state_to_embedding(combined)
        results = self.success_store.search(state_embedding, top_k)
        
        filtered = [meta for sim, meta in results if sim >= self.similarity_threshold]
        
        if filtered:
            self.access_stats['successful_retrievals'] += len(filtered)
        
        return filtered
    
    def retrieve_similar_experiences(
        self,
        current_state: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve similar past experiences"""
        self.access_stats['experience_queries'] += 1
        
        state_embedding = self._state_to_embedding(current_state)
        results = self.experience_store.search(state_embedding, top_k)
        
        filtered = [meta for sim, meta in results if sim >= self.similarity_threshold]
        
        if filtered:
            self.access_stats['successful_retrievals'] += len(filtered)
        
        return filtered
    
    def get_guidance_from_memory(
        self,
        current_state: np.ndarray,
        agent_positions: List[Tuple[int, int]],
        goal_positions: List[Tuple[int, int]],
        obstacle_positions: List[Tuple[int, int]]
    ) -> Dict:
        """Get comprehensive guidance from memory"""
        guidance = {
            'similar_trajectories': [],
            'collision_warnings': [],
            'suggested_actions': [],
            'coordination_hints': [],
            'retrieval_stats': {}
        }
        
        # Retrieve similar trajectories
        trajectories = self.retrieve_similar_trajectories(current_state, top_k=3)
        guidance['similar_trajectories'] = trajectories
        
        # Retrieve collision warnings
        collisions = self.retrieve_collision_avoidance(current_state, top_k=5)
        guidance['collision_warnings'] = collisions
        
        # Analyze patterns
        if trajectories:
            # Extract common successful actions
            action_counts = {}
            for traj in trajectories:
                for action in traj.get('actions', []):
                    action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts:
                best_action = max(action_counts, key=action_counts.get)
                guidance['suggested_actions'].append({
                    'action': best_action,
                    'confidence': action_counts[best_action] / sum(action_counts.values()),
                    'source': 'historical_trajectories'
                })
        
        # Collision avoidance suggestions
        if collisions:
            dangerous_actions = set()
            for coll in collisions:
                dangerous_actions.add(coll.get('action'))
            
            guidance['collision_warnings'] = [
                {
                    'action': action,
                    'warning': 'Historical collision detected with this action'
                }
                for action in dangerous_actions
            ]
        
        # Add retrieval stats
        guidance['retrieval_stats'] = {
            'trajectories_found': len(trajectories),
            'collisions_found': len(collisions),
            'total_queries': sum(self.access_stats.values())
        }
        
        return guidance
    
    def get_stats(self) -> Dict:
        """Get memory access statistics"""
        total_queries = (
            self.access_stats.get('trajectory_queries', 0) +
            self.access_stats.get('collision_queries', 0) +
            self.access_stats.get('success_queries', 0) +
            self.access_stats.get('experience_queries', 0)
        )
        
        successful_retrievals = self.access_stats.get('successful_retrievals', 0)
        
        stats = {
            'trajectory_queries': self.access_stats.get('trajectory_queries', 0),
            'collision_queries': self.access_stats.get('collision_queries', 0),
            'success_queries': self.access_stats.get('success_queries', 0),
            'experience_queries': self.access_stats.get('experience_queries', 0),
            'successful_retrievals': successful_retrievals,
            'total_memories': (
                len(self.trajectory_store.metadata) +
                len(self.collision_store.metadata) +
                len(self.success_store.metadata) +
                len(self.experience_store.metadata)
            ),
            'retrieval_success_rate': (
                successful_retrievals / max(total_queries, 1)
            ) if total_queries > 0 else 0.0
        }
        
        return stats
    
    def clear_all_memories(self):
        """Clear all memory stores"""
        self.trajectory_store = VectorStore(self.embedding_dim)
        self.collision_store = VectorStore(self.embedding_dim)
        self.success_store = VectorStore(self.embedding_dim)
        self.experience_store = VectorStore(self.embedding_dim)
        
        self.access_stats = {
            'trajectory_queries': 0,
            'collision_queries': 0,
            'success_queries': 0,
            'experience_queries': 0,
            'successful_retrievals': 0
        }
