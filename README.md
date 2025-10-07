# Ultimate Space Invaders AI 🚀

The ultimate AI bot that automatically plays Space Invaders on https://jordancota.site/ and only saves high scores when they beat the current leaderboard.

## 🎯 Mission
- **Target**: Beat the current leaderboard high score
- **Champion Name**: "John H" (only saved if leaderboard is beaten)
- **Game URL**: https://jordancota.site/

## 🤖 Ultimate AI Features

### Intelligent Score Management
- **Leaderboard Detection**: Automatically reads current high score from leaderboard
- **Conditional Save**: Only enters name if score beats the leaderboard
- **Champion Name**: Uses "John H" as the champion name
- **Victory Screenshots**: Captures screenshots only for record-breaking scores

### Optimized Gaming Strategy
- **Ultra-Rapid Fire**: Maximum shooting rate for highest scores
- **Adaptive Movement**: Level-based movement patterns (early/mid/expert)
- **Extended Sessions**: Up to 75,000 game loops for maximum score attempts
- **Real-time Monitoring**: Live score tracking and milestone alerts

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

### Run the Ultimate AI
```bash
# Run the ultimate AI bot
python ultimate_ai.py
```

### Test Setup
```bash
# Verify everything is working
python test_setup.py
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
- `ultimate_ai.py`: The ultimate AI bot (only saves leaderboard-beating scores)
- `test_setup.py`: Setup verification and testing
- `test_game_interaction.py`: Interactive game testing
- `requirements.txt`: Python dependencies

### Key Components
- `UltimateSpaceInvadersAI`: Main AI controller
- Leaderboard detection and score comparison
- Conditional high score saving
- Optimized gaming strategies

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