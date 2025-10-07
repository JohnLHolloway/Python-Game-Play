"""
Enhanced Space Invaders AI with better game detection and interaction
"""

import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class RobustSpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_element = None
        self.game_active = False
        self.score = 0
        self.level = 1
        self.lives = 3
        self.last_shot_time = 0
        self.shot_cooldown = 0.1
        self.game_started = False
        
    def setup_browser(self):
        """Setup Chrome browser with game-optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1400,900")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def navigate_and_find_game(self):
        """Navigate to the website and locate the game"""
        print("🎮 Navigating to https://jordancota.site/...")
        self.driver.get("https://jordancota.site/")
        
        # Wait for page to fully load
        time.sleep(5)
        
        # Scroll to find the Space Invaders section
        print("🔍 Looking for Space Invaders game...")
        
        # Try to find the game section by text
        try:
            game_heading = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Space Invaders')]")
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", game_heading)
            time.sleep(2)
            print("✅ Found Space Invaders section")
        except:
            print("⚠️ Couldn't find game heading, scrolling to middle...")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
            time.sleep(2)
        
        # Look for game canvas or container
        game_found = False
        
        # Method 1: Canvas with ID
        try:
            self.game_element = self.driver.find_element(By.ID, "gameCanvas")
            game_found = True
            print("✅ Found game canvas with ID")
        except:
            pass
        
        # Method 2: Any canvas element
        if not game_found:
            try:
                canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
                if canvases:
                    self.game_element = canvases[0]
                    game_found = True
                    print("✅ Found canvas element")
            except:
                pass
        
        # Method 3: Look for game container div
        if not game_found:
            try:
                game_container = self.driver.find_element(By.XPATH, "//*[contains(@class, 'game') or contains(@id, 'game')]")
                self.game_element = game_container
                game_found = True
                print("✅ Found game container")
            except:
                pass
        
        if not game_found:
            print("❌ Could not find game element")
            return False
        
        # Focus on the game element
        try:
            self.driver.execute_script("arguments[0].focus();", self.game_element)
            self.driver.execute_script("arguments[0].click();", self.game_element)
        except:
            pass
        
        return True
    
    def start_game_robust(self):
        """Try multiple methods to start the game"""
        print("🚀 Attempting to start the game...")
        
        methods_tried = []
        
        # Method 1: Look for Start button
        try:
            start_buttons = self.driver.find_elements(By.XPATH, "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'start')]")
            if start_buttons:
                for btn in start_buttons:
                    try:
                        if btn.is_displayed() and btn.is_enabled():
                            btn.click()
                            methods_tried.append("Start button click")
                            print("✅ Clicked start button")
                            break
                    except:
                        continue
        except:
            pass
        
        # Method 2: Click on game element
        try:
            self.game_element.click()
            methods_tried.append("Game element click")
            print("✅ Clicked on game element")
        except:
            pass
        
        # Method 3: Send key combinations
        try:
            ActionChains(self.driver).move_to_element(self.game_element).click().perform()
            time.sleep(0.5)
            ActionChains(self.driver).send_keys(Keys.ENTER).perform()
            time.sleep(0.5)
            ActionChains(self.driver).send_keys(Keys.SPACE).perform()
            methods_tried.append("Key combinations")
            print("✅ Sent start key combinations")
        except:
            pass
        
        # Method 4: JavaScript click
        try:
            self.driver.execute_script("arguments[0].click();", self.game_element)
            methods_tried.append("JavaScript click")
            print("✅ JavaScript click executed")
        except:
            pass
        
        print(f"🔧 Methods tried: {', '.join(methods_tried)}")
        
        # Wait and check if game started
        time.sleep(3)
        self.game_started = True
        self.game_active = True
        
        return True
    
    def get_score_from_page(self):
        """Extract score from page with multiple patterns"""
        try:
            page_text = self.driver.page_source
            
            # Look for score patterns
            score_patterns = [
                r'Score[:\s]*(\d+)',
                r'score[:\s]*(\d+)', 
                r'SCORE[:\s]*(\d+)',
                r'Points[:\s]*(\d+)',
                r'points[:\s]*(\d+)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    try:
                        new_score = int(matches[-1])
                        if new_score >= self.score:
                            self.score = new_score
                            return True
                    except:
                        continue
            
            return False
            
        except Exception as e:
            print(f"Error getting score: {e}")
            return False
    
    def perform_game_actions(self):
        """Perform optimized game actions"""
        current_time = time.time()
        
        # Rapid fire shooting
        if current_time - self.last_shot_time >= self.shot_cooldown:
            try:
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                self.last_shot_time = current_time
            except:
                pass
        
        # Movement pattern based on time
        pattern = int(current_time * 2) % 8
        
        try:
            if pattern == 0 or pattern == 1:
                # Move left
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern == 4 or pattern == 5:
                # Move right
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
            # Other patterns: stay still for better aiming
        except:
            pass
    
    def check_game_status(self):
        """Check if game is still running"""
        try:
            page_text = self.driver.page_source.lower()
            
            # Check for game over indicators
            game_over_signs = [
                'game over', 
                'gameover',
                'you died',
                'try again',
                'restart',
                'play again'
            ]
            
            for sign in game_over_signs:
                if sign in page_text:
                    return False
            
            return True
            
        except:
            return True  # Assume game is running if we can't check
    
    def play_game(self):
        """Main game playing loop"""
        print("🤖 Starting game play...")
        
        loop_count = 0
        last_score_update = time.time()
        max_loops = 6000  # About 5 minutes of gameplay
        
        while self.game_active and loop_count < max_loops:
            try:
                loop_count += 1
                
                # Perform game actions
                self.perform_game_actions()
                
                # Update score every 50 loops
                if loop_count % 50 == 0:
                    score_updated = self.get_score_from_page()
                    
                    if score_updated:
                        last_score_update = time.time()
                        print(f"📊 Loop {loop_count}: Score = {self.score}")
                    
                    # Check if game is still running
                    if not self.check_game_status():
                        print("🎮 Game over detected!")
                        break
                
                # If no score update for too long, assume game ended
                if time.time() - last_score_update > 30:  # 30 seconds without score change
                    print("⏰ No score updates for 30 seconds, checking game status...")
                    if loop_count > 100:  # Only after we've had time to start
                        break
                
                # Control game speed
                time.sleep(0.02)  # 50 FPS equivalent
                
            except KeyboardInterrupt:
                print("🛑 Game stopped by user")
                break
            except Exception as e:
                if loop_count % 100 == 0:
                    print(f"⚠️ Error in game loop: {e}")
                continue
        
        self.game_active = False
        print(f"🏁 Game completed! Final Score: {self.score} (after {loop_count} loops)")
        
        return self.score
    
    def run(self):
        """Main execution function"""
        try:
            print("🎮 Initializing Robust Space Invaders AI...")
            
            # Setup browser
            self.setup_browser()
            
            # Navigate and find game
            if not self.navigate_and_find_game():
                print("❌ Failed to find game")
                return 0
            
            # Start game
            if not self.start_game_robust():
                print("❌ Failed to start game")
                return 0
            
            # Play the game
            final_score = self.play_game()
            
            print(f"🏆 Final Result: {final_score} points")
            return final_score
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0
        finally:
            if self.driver:
                try:
                    # Take a screenshot before closing (for debugging)
                    self.driver.save_screenshot("final_game_state.png")
                    print("📸 Screenshot saved as final_game_state.png")
                except:
                    pass
                
                time.sleep(2)
                self.driver.quit()

def main():
    """Run the AI bot"""
    print("🚀 ROBUST SPACE INVADERS AI BOT")
    print("🎯 Target: https://jordancota.site/")
    print("🏆 Goal: Beat 25,940 points")
    print("="*50)
    
    # Run multiple attempts
    best_score = 0
    attempts = 3
    
    for attempt in range(1, attempts + 1):
        print(f"\n🎮 ATTEMPT {attempt}/{attempts}")
        print("-" * 30)
        
        bot = RobustSpaceInvadersAI()
        score = bot.run()
        
        if score > best_score:
            best_score = score
            print(f"🏆 NEW BEST SCORE: {best_score}")
        
        if best_score > 25940:
            print("🎉 HIGH SCORE BEATEN! MISSION ACCOMPLISHED!")
            break
        
        if attempt < attempts:
            print(f"⏳ Waiting 5 seconds before next attempt...")
            time.sleep(5)
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"🏆 Best Score Achieved: {best_score}")
    print(f"🎯 Target Score: 25,940")
    
    if best_score > 25940:
        print("🎉 SUCCESS! High score beaten!")
    else:
        print(f"📈 Need {25940 - best_score} more points to beat high score")

if __name__ == "__main__":
    main()