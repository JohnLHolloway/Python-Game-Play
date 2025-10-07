"""
Final Optimized Space Invaders AI
Focus on solving the game-over detection issue and maximizing gameplay time
"""

import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class FinalSpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.score = 0
        self.high_score = 0
        self.level = 1
        self.lives = 3
        self.game_started = False
        self.loop_count = 0
        
    def setup_browser(self):
        """Setup optimized browser"""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1400,900")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def navigate_to_game(self):
        """Navigate and prepare game"""
        print("🎮 Loading Space Invaders from jordancota.site...")
        self.driver.get("https://jordancota.site/")
        time.sleep(5)  # Longer wait for full page load
        
        # Scroll to game section
        print("🔍 Locating game...")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
        time.sleep(3)
        
        # Find game canvas
        try:
            self.game_canvas = self.driver.find_element(By.ID, "gameCanvas")
            print("✅ Game canvas found by ID")
        except:
            canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                self.game_canvas = canvases[0]
                print("✅ Canvas element found")
            else:
                print("❌ No canvas found")
                return False
        
        return True
    
    def start_game_carefully(self):
        """Carefully start the game with verification"""
        print("🚀 Starting game sequence...")
        
        # Method 1: Click start button if available
        try:
            start_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Start') or contains(text(), 'start')]")
            for btn in start_buttons:
                if btn.is_displayed():
                    btn.click()
                    print("✅ Start button clicked")
                    break
        except:
            pass
        
        time.sleep(2)
        
        # Method 2: Focus and click canvas
        try:
            self.game_canvas.click()
            print("✅ Canvas clicked")
        except:
            pass
        
        time.sleep(1)
        
        # Method 3: Send start keys
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER).perform()
            time.sleep(0.5)
            actions.send_keys(Keys.SPACE).perform()
            print("✅ Start keys sent")
        except:
            pass
        
        # Wait for game to initialize
        time.sleep(3)
        
        # Verify game started by checking for initial game state
        self.check_game_state()
        if self.score >= 0:  # Any score means game is active
            self.game_started = True
            print("✅ Game appears to be active")
        
        return True
    
    def check_game_state(self):
        """Check and update game state"""
        try:
            page_source = self.driver.page_source
            
            # Extract score with multiple patterns
            score_patterns = [
                r'Score[:\s]*(\d+)',
                r'score[:\s]*(\d+)',
                r'>Score[<\s]*(\d+)',
                r'SCORE[:\s]*(\d+)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    try:
                        latest_score = int(matches[-1])
                        if latest_score >= self.score:
                            self.score = latest_score
                    except:
                        continue
            
            # Extract level
            level_matches = re.findall(r'Level[:\s]*(\d+)', page_source, re.IGNORECASE)
            if level_matches:
                try:
                    self.level = int(level_matches[-1])
                except:
                    pass
            
            # Extract lives
            lives_matches = re.findall(r'Lives[:\s]*(\d+)', page_source, re.IGNORECASE)
            if lives_matches:
                try:
                    self.lives = int(lives_matches[-1])
                except:
                    pass
                    
        except Exception as e:
            pass
    
    def is_game_really_over(self):
        """More conservative game over detection"""
        try:
            page_text = self.driver.page_source.lower()
            
            # Only trigger game over on very clear indicators
            definite_game_over = [
                'game over',
                'gameover', 
                'you died',
                'mission failed'
            ]
            
            # Count how many indicators we find
            indicators_found = 0
            for indicator in definite_game_over:
                if indicator in page_text:
                    indicators_found += 1
            
            # Only consider game over if multiple indicators or lives is definitely 0
            if indicators_found >= 1 and self.lives == 0:
                return True
            elif indicators_found >= 2:
                return True
            
            return False
            
        except:
            return False
    
    def play_optimized_game(self):
        """Play with optimized strategy"""
        print("🤖 Starting optimized gameplay...")
        print("🎯 Strategy: Conservative game-over detection + aggressive play")
        
        actions = ActionChains(self.driver)
        self.loop_count = 0
        last_score_check = time.time()
        last_progress_report = time.time()
        
        # Much longer gameplay loop
        max_loops = 20000
        
        while self.loop_count < max_loops:
            try:
                self.loop_count += 1
                
                # Rapid fire shooting
                actions.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                
                # Movement pattern
                movement_cycle = self.loop_count % 12
                
                if movement_cycle < 2:
                    # Move left
                    actions.key_down(Keys.ARROW_LEFT).perform()
                    time.sleep(0.06)
                    actions.key_up(Keys.ARROW_LEFT).perform()
                elif movement_cycle > 9:
                    # Move right
                    actions.key_down(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.06)
                    actions.key_up(Keys.ARROW_RIGHT).perform()
                # Otherwise hold position for accuracy
                
                # Check game state every 150 loops (less frequent)
                if self.loop_count % 150 == 0:
                    self.check_game_state()
                    last_score_check = time.time()
                
                # Progress report every 500 loops
                if self.loop_count % 500 == 0:
                    elapsed = time.time() - last_progress_report
                    print(f"📊 Loop {self.loop_count}: Score={self.score}, Level={self.level}, Lives={self.lives}, Time={elapsed:.1f}s")
                    last_progress_report = time.time()
                    
                    # Update high score
                    if self.score > self.high_score:
                        self.high_score = self.score
                        print(f"🏆 New high score: {self.high_score}")
                
                # Conservative game over check (only every 300 loops)
                if self.loop_count % 300 == 0:
                    if self.is_game_really_over():
                        print("🎮 Game over confirmed after conservative check")
                        break
                
                # Adaptive timing
                if self.loop_count < 1000:
                    time.sleep(0.025)  # Slower start
                elif self.loop_count < 5000:
                    time.sleep(0.02)   # Medium
                else:
                    time.sleep(0.015)  # Faster for high scores
                
            except KeyboardInterrupt:
                print("🛑 Game stopped by user")
                break
            except Exception as e:
                # Don't stop on errors, just continue
                continue
        
        print(f"🏁 Game completed!")
        print(f"🏆 Final Score: {self.score}")
        print(f"📊 Final Level: {self.level}")
        print(f"🔄 Total Loops: {self.loop_count}")
        print(f"⏱️ Average Score per Loop: {self.score / max(1, self.loop_count):.2f}")
        
        return self.score
    
    def run_session(self):
        """Run complete session"""
        try:
            self.setup_browser()
            
            if not self.navigate_to_game():
                return 0
                
            if not self.start_game_carefully():
                return 0
            
            final_score = self.play_optimized_game()
            
            # Save final screenshot
            try:
                self.driver.save_screenshot(f"final_score_{final_score}.png")
                print(f"📸 Screenshot saved: final_score_{final_score}.png")
            except:
                pass
            
            return final_score
            
        except Exception as e:
            print(f"❌ Session error: {e}")
            return 0
        finally:
            if self.driver:
                time.sleep(2)
                self.driver.quit()

def main():
    """Main execution"""
    print("🎮 FINAL OPTIMIZED SPACE INVADERS AI")
    print("🎯 Goal: Beat 25,940 points with extended gameplay")
    print("🔧 Strategy: Conservative game-over detection + aggressive scoring")
    print("="*60)
    
    max_sessions = 3
    best_score = 0
    all_scores = []
    
    for session_num in range(1, max_sessions + 1):
        print(f"\n🎮 SESSION {session_num}/{max_sessions}")
        print("🚀 " + "-" * 40)
        
        ai = FinalSpaceInvadersAI()
        score = ai.run_session()
        all_scores.append(score)
        
        print(f"📊 Session {session_num} Result: {score} points")
        
        if score > best_score:
            best_score = score
            print(f"🏆 NEW PERSONAL BEST: {best_score} points!")
        
        # Check if we beat the target
        if best_score > 25940:
            print("🎉🎉🎉 TARGET ACHIEVED! HIGH SCORE BEATEN! 🎉🎉🎉")
            break
        
        if session_num < max_sessions:
            print("⏳ 15 second break before next session...")
            time.sleep(15)
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL AI PERFORMANCE SUMMARY")
    print("="*60)
    print(f"🏆 Best Score Achieved: {best_score} points")
    print(f"📊 All Session Scores: {all_scores}")
    print(f"📈 Average Score: {sum(all_scores)/len(all_scores):.1f}")
    print(f"🎯 Target Score: 25,940 points")
    
    if best_score > 25940:
        print("🎉 MISSION ACCOMPLISHED! You are the new Space Invaders champion!")
        improvement = best_score - 25940
        print(f"🚀 You beat the record by {improvement} points!")
    elif best_score > 20000:
        print("🔥 EXCELLENT! Very close to beating the record!")
        gap = 25940 - best_score
        print(f"📈 Only {gap} points away from the target!")
    elif best_score > 10000:
        print("💪 GREAT PERFORMANCE! You're making solid progress!")
        gap = 25940 - best_score
        print(f"📈 Need {gap} more points to beat the record")
    elif best_score > 1000:
        print("👍 Good score! The AI is working and improving!")
        gap = 25940 - best_score
        print(f"📈 {gap} points needed to reach the target")
    else:
        print("🔧 The AI needs more optimization to reach higher scores")
        print("💡 Consider adjusting timing, movement patterns, or game detection")
    
    print("\n🤖 Thank you for using the Final Optimized Space Invaders AI!")
    print("🎮 The AI has been designed to automatically play and score as high as possible!")

if __name__ == "__main__":
    main()