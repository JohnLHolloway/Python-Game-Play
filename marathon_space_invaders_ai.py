"""
Extended Play Space Invaders AI - Focused on Long Gameplay Sessions
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

class ExtendedPlayAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.score = 0
        self.level = 1
        self.lives = 3
        self.game_time = 0
        self.start_time = 0
        
    def setup_browser(self):
        """Setup browser for extended play"""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1200,800")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--no-sandbox")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def navigate_and_start(self):
        """Navigate to game and start"""
        print("🎮 Connecting to Jordan Cota's Space Invaders...")
        self.driver.get("https://jordancota.site/")
        time.sleep(4)
        
        # Scroll to game
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
        time.sleep(2)
        
        # Find canvas
        try:
            self.game_canvas = self.driver.find_element(By.ID, "gameCanvas")
            print("✅ Game canvas located")
        except:
            canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                self.game_canvas = canvases[0]
                print("✅ Canvas found")
            else:
                return False
        
        # Start game
        try:
            start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            start_btn.click()
            print("✅ Start button clicked")
        except:
            self.game_canvas.click()
            print("✅ Canvas clicked to start")
        
        # Send start keys
        ActionChains(self.driver).send_keys(Keys.SPACE).perform()
        time.sleep(1)
        
        self.start_time = time.time()
        return True
    
    def play_extended_session(self):
        """Play extended session focusing on survival and score"""
        print("🚀 Starting EXTENDED PLAY session...")
        print("🎯 Goal: Survive as long as possible for maximum score")
        
        actions = ActionChains(self.driver)
        loop_count = 0
        last_update = time.time()
        
        # Extended play loop - much longer than previous versions
        while loop_count < 30000:  # 30,000 loops for extended play
            try:
                loop_count += 1
                current_time = time.time()
                self.game_time = current_time - self.start_time
                
                # Continuous rapid fire
                actions.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                
                # Varied movement pattern
                movement_pattern = loop_count % 20
                
                if movement_pattern < 3:
                    # Move left
                    actions.key_down(Keys.ARROW_LEFT).perform()
                    time.sleep(0.05)
                    actions.key_up(Keys.ARROW_LEFT).perform()
                elif movement_pattern > 16:
                    # Move right
                    actions.key_down(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.05)
                    actions.key_up(Keys.ARROW_RIGHT).perform()
                # Otherwise stay still for better accuracy
                
                # Update score every 100 loops
                if loop_count % 100 == 0:
                    self.update_game_state()
                    
                    if current_time - last_update > 10:  # Every 10 seconds
                        print(f"📊 Time: {int(self.game_time)}s | Score: {self.score} | Level: {self.level} | Loops: {loop_count}")
                        last_update = current_time
                
                # Check for game over every 200 loops
                if loop_count % 200 == 0:
                    if self.is_game_over():
                        print("🎮 Game over detected")
                        break
                
                # Adaptive timing
                if loop_count < 5000:
                    time.sleep(0.02)  # Slower start
                elif loop_count < 15000:
                    time.sleep(0.015)  # Medium speed
                else:
                    time.sleep(0.01)  # Maximum speed
                
            except KeyboardInterrupt:
                print("🛑 Extended play stopped by user")
                break
            except:
                continue
        
        print(f"🏁 Extended play completed!")
        print(f"⏱️ Total play time: {int(self.game_time)} seconds")
        print(f"🏆 Final score: {self.score}")
        print(f"📊 Final level: {self.level}")
        print(f"🔄 Total loops: {loop_count}")
        
        return self.score
    
    def update_game_state(self):
        """Update game state from page"""
        try:
            page_source = self.driver.page_source
            
            # Find score
            score_match = re.search(r'Score[:\s]*(\d+)', page_source, re.IGNORECASE)
            if score_match:
                new_score = int(score_match.group(1))
                if new_score > self.score:
                    self.score = new_score
            
            # Find level
            level_match = re.search(r'Level[:\s]*(\d+)', page_source, re.IGNORECASE)
            if level_match:
                self.level = int(level_match.group(1))
            
            # Find lives
            lives_match = re.search(r'Lives[:\s]*(\d+)', page_source, re.IGNORECASE)
            if lives_match:
                self.lives = int(lives_match.group(1))
                
        except:
            pass
    
    def is_game_over(self):
        """Check if game is over"""
        try:
            page_text = self.driver.page_source.lower()
            game_over_signs = ['game over', 'gameover', 'you died', 'try again']
            
            for sign in game_over_signs:
                if sign in page_text:
                    return True
            
            return self.lives <= 0
            
        except:
            return False
    
    def run_marathon(self):
        """Run marathon session"""
        try:
            self.setup_browser()
            
            if not self.navigate_and_start():
                print("❌ Failed to start game")
                return 0
            
            score = self.play_extended_session()
            
            # Take final screenshot
            self.driver.save_screenshot("marathon_final.png")
            
            return score
            
        except Exception as e:
            print(f"❌ Marathon error: {e}")
            return 0
        finally:
            if self.driver:
                self.driver.quit()

def main():
    """Run marathon session"""
    print("🏃‍♂️ SPACE INVADERS MARATHON MODE")
    print("🎯 Target: 25,940+ points through extended play")
    print("⏱️ Strategy: Long survival sessions")
    print("="*50)
    
    # Run multiple marathon sessions
    best_score = 0
    sessions = 2
    
    for session in range(1, sessions + 1):
        print(f"\n🏃‍♂️ MARATHON SESSION {session}/{sessions}")
        print("-" * 30)
        
        ai = ExtendedPlayAI()
        score = ai.run_marathon()
        
        print(f"📊 Session {session} final score: {score}")
        
        if score > best_score:
            best_score = score
            print(f"🏆 NEW BEST: {best_score} points!")
        
        if best_score > 25940:
            print("🎉 TARGET ACHIEVED!")
            break
        
        if session < sessions:
            print("⏳ 10 second break before next session...")
            time.sleep(10)
    
    print(f"\n🏁 MARATHON RESULTS:")
    print(f"🏆 Best Score: {best_score}")
    print(f"🎯 Target: 25,940")
    
    if best_score > 25940:
        print("🎉 HIGH SCORE BEATEN! CONGRATULATIONS!")
    else:
        print(f"📈 {25940 - best_score} points needed to beat record")

if __name__ == "__main__":
    main()