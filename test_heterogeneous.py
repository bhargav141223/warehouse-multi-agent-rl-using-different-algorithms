import requests
import json

# Test heterogeneous training initialization
url = "http://localhost:8000/api/environment/initialize"

# Configure heterogeneous training with 4 different algorithms
config = {
    "environment_type": "complex",  # 4 agents
    "num_episodes": 50,  # Short test
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "use_rag": True,
    "use_llm_shaping": True,
    "heterogeneous": True,
    "algorithms": ["mappo", "dqn", "a2c", "ppo"]  # 4 different algorithms
}

print("Testing heterogeneous training initialization...")
print(f"Configuration: {json.dumps(config, indent=2)}")

try:
    response = requests.post(url, json=config)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Heterogeneous training initialized successfully!")
        print(f"Session ID: {result['session_id']}")
        print(f"Environment: {result['environment']}")
        
        if 'algorithms' in result:
            print("\nAlgorithm Assignment:")
            for algo_info in result['algorithms']:
                print(f"  Agent {algo_info['agent_id']}: {algo_info['algorithm'].upper()}")
        
        print(f"\nYou can now start training with session ID: {result['session_id']}")
    else:
        print(f"✗ Error: {response.text}")
        
except Exception as e:
    print(f"✗ Exception: {e}")
