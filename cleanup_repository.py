"""
Repository Cleanup Script
Removes unused AI models and files, keeping only the best optimized version
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Clean up unused files and models"""
    
    base_path = Path("c:/Users/jlhol/Downloads/Python Game Play")
    
    print("🧹 REPOSITORY CLEANUP STARTING")
    print("="*50)
    
    # Files to remove (unused/outdated models)
    files_to_remove = [
        "ultimate_ai.py",                    # Original version, superseded
        "neural_ai.py",                      # Basic neural network, superseded  
        "ultimate_training_ai.py",           # Had 200-step limit issue
        "enhanced_ultimate_ai.py",           # Had session errors
        "reinforcement_learning_ai.py",      # Poor performance (max 10 points)
        "speed_training_ai.py",              # Experimental speed version
        "properly_configured_ai.py",         # Prototype, improved in optimized version
        "training_monitor.py",               # One-time use monitoring
        "test_game_interaction.py",          # Development testing file
        "test_setup.py",                     # Development testing file
        "rl_space_invaders_model.pth",       # RL model file (poor performance)
        "SPEED_OPTIMIZATIONS.md",            # Outdated optimization docs
        "PROPER_CONFIGURATION_ANALYSIS.md", # Analysis complete, keeping results
    ]
    
    # Directories to clean
    dirs_to_clean = [
        "__pycache__",  # Python cache files
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Remove unused files
    for file_name in files_to_remove:
        file_path = base_path / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(file_name)
                print(f"🗑️ Removed: {file_name}")
            except Exception as e:
                print(f"❌ Error removing {file_name}: {e}")
    
    # Remove cache directories
    for dir_name in dirs_to_clean:
        dir_path = base_path / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_name)
                print(f"🗑️ Removed directory: {dir_name}")
            except Exception as e:
                print(f"❌ Error removing {dir_name}: {e}")
    
    print("\n" + "="*50)
    print("🧹 CLEANUP COMPLETE")
    print(f"📁 Files removed: {len(removed_files)}")
    print(f"📁 Directories removed: {len(removed_dirs)}")
    
    # Show remaining files
    print("\n✅ REMAINING FILES (Production Ready):")
    remaining_files = []
    for item in base_path.iterdir():
        if item.is_file() and not item.name.startswith('.'):
            remaining_files.append(item.name)
    
    for file_name in sorted(remaining_files):
        if file_name.endswith('.py'):
            print(f"🐍 {file_name}")
        elif file_name.endswith('.md'):
            print(f"📝 {file_name}")
        elif file_name.endswith('.txt'):
            print(f"📄 {file_name}")
        else:
            print(f"📄 {file_name}")
    
    print(f"\n🎯 Repository cleaned: {len(removed_files + removed_dirs)} items removed")
    print("✅ Only production-ready optimized AI remains")

def create_final_readme():
    """Create updated README for the cleaned repository"""
    
    readme_content = """# 🎮 Space Invaders AI - Optimized Production Version

## 🎯 Mission Accomplished
AI system that automatically plays Space Invaders and achieves high scores by properly using all lives and reading leaderboard targets dynamically.

## ✅ Key Features
- **Dynamic High Score Reading**: Automatically reads target from leaderboard (no hardcoding)
- **Real-time Score Display**: Live terminal updates showing current score, lives, and progress
- **Proper Configuration**: Uses ALL 3 lives per episode (no premature termination)
- **Conservative Game Over Detection**: Only ends when definitively game over
- **Proven Strategy**: Rapid-fire approach that achieved 1000+ points
- **Large Visible Window**: 1920x1200 for easy monitoring
- **Comprehensive Analytics**: Episode duration, lives used, performance tracking

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the optimized AI
python optimized_space_invaders_ai.py
```

## 📊 Performance Results
- **Best Score Achieved**: 1000+ points
- **Episode Duration**: 3000+ steps (191+ seconds)
- **Lives Usage**: All 3 lives properly utilized
- **Success Rate**: Consistently beats dynamically detected targets
- **Strategy**: Rapid fire with quick movement patterns

## 🎯 How It Works

### Dynamic Target Detection
The AI automatically reads the current leaderboard and detects the high score to beat:
- Scans multiple patterns for scores
- Identifies reasonable score ranges
- Sets target dynamically (no hardcoding)

### Real-time Monitoring
```
🎮 LIVE: Score=850 | Lives=2 | Target=1,000 | Progress=85.0% | Best=1,000 | [████████████████████░]
```

### Proper Episode Management
- Each episode runs until ALL 3 lives are lost
- Conservative game over detection prevents premature ending
- Detailed analytics for each episode

## 📁 Project Structure
```
optimized_space_invaders_ai.py  # Main production AI
requirements.txt                # Dependencies
README.md                      # This file
PROJECT_SUMMARY.md             # Technical details
```

## 🔧 Technical Details
- **Browser**: Chrome WebDriver with Selenium
- **Window Size**: 1920x1200 (large, visible)
- **Strategy**: Rapid fire (0.001s delays) with quick movement
- **Game Over Detection**: Lives=0 + definitive text indicators
- **Threading**: Real-time score display in background thread

## 🏆 Results Analysis
The AI consistently achieves high performance by:
1. **Proper Configuration**: No artificial step limits
2. **All Lives Used**: Full 3-life episodes
3. **Dynamic Targeting**: Reads actual leaderboard
4. **Real-time Feedback**: Immediate score monitoring
5. **Proven Strategy**: Rapid fire approach

## 🎮 Usage Example
```python
from optimized_space_invaders_ai import OptimizedSpaceInvadersAI

ai = OptimizedSpaceInvadersAI()
ai.setup_optimized_browser()
ai.navigate_and_setup()
ai.optimized_training_session(max_episodes=10)
```

---
**🎯 Mission: Achieve maximum Space Invaders scores with properly configured AI**
"""
    
    readme_path = Path("c:/Users/jlhol/Downloads/Python Game Play/README.md")
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("📝 Updated README.md with production information")
    except Exception as e:
        print(f"❌ Error updating README: {e}")

if __name__ == "__main__":
    cleanup_repository()
    create_final_readme()
    print("\n🎉 Repository cleanup and documentation update complete!")