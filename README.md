# Space Invaders AI Bot 🚀

An advanced AI bot that automatically plays the Space Invaders game on https://jordancota.site/ to achieve the highest score possible.

## 🎯 Current Target
- **High Score to Beat**: 25,940 points (Level 8) by "John H"
- **Game URL**: https://jordancota.site/

## 🤖 Features

### Advanced AI Strategy
- **Rapid Fire**: Continuous shooting with optimal timing
- **Adaptive Movement**: Different movement patterns based on game level
- **Learning System**: Saves statistics and improves over time
- **Multi-Strategy Approach**: Zigzag, defensive, and aggressive patterns

### Robust Automation
- **Smart Game Detection**: Multiple methods to find and start the game
- **Error Recovery**: Handles network issues and game state changes
- **Progress Monitoring**: Real-time score, level, and lives tracking
- **Statistics Tracking**: Saves best scores and game history

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Chrome browser
- Windows/Linux/macOS

### Setup
```bash
# Install required packages (already done)
pip install selenium webdriver-manager opencv-python numpy pillow pyautogui

# All packages are already installed in the virtual environment
```

## 🚀 Usage

### Option 1: Simple Run
```bash
# Run the advanced AI bot
python advanced_space_invaders_ai.py
```

### Option 2: Interactive Runner
```bash
# Run with options menu
python run_bot.py
```

### Option 3: Basic Version
```bash
# Run the basic version
python space_invaders_ai.py
```

## 🎮 Game Strategy

### Level 1-2: Early Game
- Continuous rapid fire
- Zigzag movement pattern
- Focus on clearing invaders quickly

### Level 3-5: Mid Game  
- More defensive positioning
- Calculated movements
- Maintain rapid fire rate

### Level 6+: High Levels
- Aggressive movement
- Maximum fire rate
- Risk/reward optimization

## 📊 AI Features

### Computer Vision
- Screen capture and analysis
- Object detection for invaders and bullets
- Safe zone calculation
- Optimal positioning algorithms

### Machine Learning
- Pattern recognition
- Strategy adaptation
- Performance optimization
- Historical data analysis

### Game Analysis
- Real-time score monitoring
- Level progression tracking
- Lives management
- Game over detection

## 🎯 Strategy Details

### Shooting Strategy
- **Fire Rate**: 12-20 shots per second
- **Accuracy**: Continuous fire for maximum hit rate
- **Timing**: No wasted shots, optimal cooldown management

### Movement Strategy
- **Early Levels**: Wide zigzag patterns
- **Mid Levels**: Defensive positioning with calculated moves
- **High Levels**: Aggressive side-to-side with quick stops

### Survival Strategy
- **Bullet Dodging**: Predictive movement patterns
- **Safe Positioning**: Analysis of bullet trajectories
- **Risk Management**: Balance between offense and defense

## 📈 Performance Optimization

### Browser Settings
- Optimized Chrome configuration
- Disabled unnecessary features
- Maximum window size for better detection
- Reduced input lag

### Game Loop Optimization
- Adaptive timing based on game level
- Efficient action execution
- Minimal delay between commands
- Error handling and recovery

## 🏆 Results Tracking

The bot automatically saves:
- Best scores achieved
- Games played
- Successful strategies
- Performance statistics

Check `game_stats.json` for detailed analytics.

## 🔧 Troubleshooting

### Common Issues

1. **Game not found**
   - Check internet connection
   - Verify website is accessible
   - Try different browser window size

2. **Canvas detection failed**
   - Website may have changed structure
   - Try refreshing the page manually
   - Check for browser popups/ads

3. **Controls not working**
   - Ensure game window has focus
   - Check for browser permission issues
   - Try running as administrator

### Debug Mode
Add debug logging by modifying the print statements in the code.

## 🎮 Game Controls (Manual Reference)
- **←/→ Arrow Keys**: Move left/right
- **Spacebar**: Shoot
- **Refresh Page**: Restart game

## 📝 Code Structure

### Main Files
- `advanced_space_invaders_ai.py`: Main AI bot with advanced features
- `space_invaders_ai.py`: Basic version with core functionality
- `run_bot.py`: Interactive runner with multiple game modes
- `game_stats.json`: Statistics and performance data (auto-generated)

### Key Classes
- `AdvancedSpaceInvadersAI`: Main AI controller
- Game state management
- Strategy execution
- Performance monitoring

## 🚀 Future Enhancements

Potential improvements:
- Neural network for pattern recognition
- Reinforcement learning for strategy optimization
- Multiple browser support
- Real-time strategy adjustment
- Tournament mode with multiple bots

## ⚠️ Disclaimer

This bot is for educational and entertainment purposes. Use responsibly and respect the website's terms of service.

## 🎯 Goal

**Beat the current high score of 25,940 points and reach Level 9+!**

---

**Good luck! May the AI be with you! 🤖🚀**