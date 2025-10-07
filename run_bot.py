"""
Space Invaders AI Bot Runner
Simple script to run the AI bot with different strategies
"""

import sys
import time
from advanced_space_invaders_ai import AdvancedSpaceInvadersAI

def run_multiple_games(num_games=5):
    """Run multiple games to maximize score"""
    print(f"🎮 Running {num_games} games to achieve maximum score...")
    
    best_score = 0
    best_level = 0
    
    for game_num in range(1, num_games + 1):
        print(f"\n🚀 Starting Game {game_num}/{num_games}")
        print("="*50)
        
        bot = AdvancedSpaceInvadersAI()
        
        try:
            bot.run()
            
            if bot.score > best_score:
                best_score = bot.score
                best_level = bot.level
                print(f"🏆 NEW BEST SCORE: {best_score} (Level {best_level})")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping all games...")
            break
        except Exception as e:
            print(f"❌ Game {game_num} failed: {e}")
        
        # Wait between games
        if game_num < num_games:
            print(f"⏳ Waiting 3 seconds before next game...")
            time.sleep(3)
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"🏆 Best Score: {best_score}")
    print(f"📊 Best Level: {best_level}")
    print(f"🎯 Target to Beat: 25,940 points")
    
    if best_score > 25940:
        print("🎉 CONGRATULATIONS! You beat the high score!")
    else:
        print(f"📈 Keep trying! You need {25940 - best_score} more points.")

def main():
    print("🎮 Space Invaders AI Bot Runner")
    print("Choose an option:")
    print("1. Run single game")
    print("2. Run 3 games")
    print("3. Run 5 games")
    print("4. Run 10 games (marathon mode)")
    print("5. Run until high score is beaten")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            bot = AdvancedSpaceInvadersAI()
            bot.run()
        elif choice == "2":
            run_multiple_games(3)
        elif choice == "3":
            run_multiple_games(5)
        elif choice == "4":
            run_multiple_games(10)
        elif choice == "5":
            # Run until we beat the high score
            game_count = 0
            target_score = 25940
            
            while True:
                game_count += 1
                print(f"\n🚀 Attempt {game_count} to beat {target_score}")
                
                bot = AdvancedSpaceInvadersAI()
                try:
                    bot.run()
                    
                    if bot.score > target_score:
                        print(f"🎉 SUCCESS! Beat high score with {bot.score} points!")
                        break
                    else:
                        print(f"📊 Score: {bot.score}, Need: {target_score - bot.score} more")
                        
                except KeyboardInterrupt:
                    print("\n🛑 Stopping marathon mode...")
                    break
                    
                time.sleep(2)
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Goodbye!")

if __name__ == "__main__":
    main()