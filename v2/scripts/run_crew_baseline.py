import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.env.sre_env import SreDecisionEnvV2
from v2.agents.tools import set_env
from v2.agents.crew import create_sre_crew

def run_baseline(num_episodes=5):
    env = SreDecisionEnvV2()
    set_env(env)
    
    results = []
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    for i in range(num_episodes):
        print(f"\n{'='*20} EPISODE {i+1} {'='*20}")
        obs = env.reset(difficulty="medium")
        
        crew = create_sre_crew()
        
        start_time = datetime.now()
        crew_output = crew.kickoff()
        end_time = datetime.now()
        
        # Collect episode metrics
        total_reward = sum(env.reward_system.calculate(env.state, {}).values()) # Get final state rewards if any
        # Note: In sequential CrewAI, steps are handled inside tools. 
        # We can pull the final state from the env instance.
        
        episode_data = {
            "episode": i + 1,
            "root_cause": env.state.root_cause,
            "resolved": env.state.is_resolved,
            "steps": env.state.steps,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "crew_output": str(crew_output),
            "history": [m.dict() if hasattr(m, 'dict') else m for m in env.state.message_bus.get_messages()]
        }
        results.append(episode_data)
        
        # Save results immediately after each episode
        filename = f"data/baseline_crew_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{i+1}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEpisode {i+1} Result: {'SUCCESS' if env.state.is_resolved else 'FAILURE'}")
        print(f"Root Cause: {env.state.root_cause}")
        print(f"Steps: {env.state.steps}")

    print(f"\nBaseline complete. Results saved to data/")

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in environment.")
        sys.exit(1)
        
    run_baseline(num_episodes=3) # Small run for baseline check
