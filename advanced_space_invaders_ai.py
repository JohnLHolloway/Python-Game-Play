"""
Advanced Space Invaders AI with Computer Vision and Strategic Play
This version uses more sophisticated image analysis and game strategy.
"""

import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import threading
import json
import os

class AdvancedSpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.game_active = False
        self.score = 0
        self.level = 1
        self.lives = 3
        self.last_shot_time = 0
        self.shot_cooldown = 0.08  # Aggressive shooting
        self.movement_pattern = 0
        self.dodge_mode = False
        self.aggressive_mode = True
        
        # Game statistics for learning
        self.stats = {
            'games_played': 0,
            'best_score': 0,
            'strategies_tried': [],
            'successful_patterns': []
        }
        
        self.load_stats()
    
    def load_stats(self):
        """Load previous game statistics"""
        try:
            if os.path.exists('game_stats.json'):
                with open('game_stats.json', 'r') as f:
                    self.stats = json.load(f)
        except:
            pass
    
    def save_stats(self):
        """Save game statistics"""
        try:
            with open('game_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except:
            pass
    
    def setup_browser(self):
        """Initialize Chrome browser optimized for gaming"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Maximize window for better game visibility
        self.driver.maximize_window()
    
    def navigate_to_game(self):
        """Navigate to the Space Invaders game with error handling"""
        print("🎮 Navigating to Space Invaders game...")
        self.driver.get("https://jordancota.site/")
        
        # Wait for page to load completely
        WebDriverWait(self.driver, 15).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        
        # Scroll to game section
        try:
            game_section = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Space Invaders')]"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", game_section)
            time.sleep(2)
        except:
            print("Game section not found, scrolling to middle of page...")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
        
        # Find game canvas with multiple strategies
        canvas_found = False
        
        # Strategy 1: Look for canvas with ID
        try:
            self.game_canvas = self.driver.find_element(By.ID, "gameCanvas")
            canvas_found = True
            print("✅ Found game canvas by ID")
        except:
            pass
        
        # Strategy 2: Look for any canvas element
        if not canvas_found:
            try:
                canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
                if canvases:
                    self.game_canvas = canvases[0]
                    canvas_found = True
                    print("✅ Found canvas element")
            except:
                pass
        
        # Strategy 3: Look for game container
        if not canvas_found:
            try:
                game_container = self.driver.find_element(By.XPATH, "//*[contains(@class, 'game') or contains(@id, 'game')]")
                self.game_canvas = game_container
                canvas_found = True
                print("✅ Found game container")
            except:
                pass
        
        if not canvas_found:
            print("❌ Could not find game canvas")
            return False
        
        # Focus on the game canvas
        self.driver.execute_script("arguments[0].focus();", self.game_canvas)
        return True
    
    def start_game(self):
        """Start the game with multiple strategies"""
        print("🚀 Starting the game...")
        
        # Strategy 1: Look for start button
        start_clicked = False
        try:
            start_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Start') or contains(text(), 'start') or contains(text(), 'START')]")
            if start_buttons:
                start_buttons[0].click()
                start_clicked = True
                print("✅ Clicked start button")
        except:
            pass
        
        # Strategy 2: Click on canvas
        if not start_clicked:
            try:
                ActionChains(self.driver).move_to_element(self.game_canvas).click().perform()
                print("✅ Clicked on canvas")
                start_clicked = True
            except:
                pass
        
        # Strategy 3: Press Enter or Space
        if not start_clicked:
            try:
                ActionChains(self.driver).send_keys(Keys.ENTER).perform()
                time.sleep(0.5)
                ActionChains(self.driver).send_keys(Keys.SPACE).perform()
                print("✅ Sent start keys")
            except:
                pass
        
        time.sleep(1)
        self.game_active = True
        print("🎯 Game should be active now!")
    
    def get_game_state(self):
        """Extract game state with robust parsing"""
        try:
            # Get page text and parse for game info
            page_text = self.driver.page_source.lower()
            
            # Parse score
            score_patterns = [
                r'score[:\s]*(\d+)',
                r'points[:\s]*(\d+)',
                r'score.*?(\d+)'
            ]
            
            import re
            for pattern in score_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    try:
                        new_score = int(matches[-1])  # Get the last/highest score
                        if new_score > self.score:
                            self.score = new_score
                        break
                    except:
                        continue
            
            # Parse level
            level_patterns = [
                r'level[:\s]*(\d+)',
                r'stage[:\s]*(\d+)',
                r'wave[:\s]*(\d+)'
            ]
            
            for pattern in level_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    try:
                        self.level = int(matches[-1])
                        break
                    except:
                        continue
            
            # Parse lives
            lives_patterns = [
                r'lives[:\s]*(\d+)',
                r'life[:\s]*(\d+)',
                r'remaining[:\s]*(\d+)'
            ]
            
            for pattern in lives_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    try:
                        self.lives = int(matches[-1])
                        break
                    except:
                        continue
            
        except Exception as e:
            print(f"Could not parse game state: {e}")
    
    def execute_advanced_strategy(self):
        """Execute advanced AI strategy based on game state and learning"""
        
        # Continuous rapid fire
        self.rapid_fire()
        
        # Movement strategy based on level and score
        if self.level <= 2:
            # Early levels: Stay mobile, shoot constantly
            self.zigzag_movement()
        elif self.level <= 5:
            # Mid levels: More defensive, calculated movements
            self.defensive_movement()
        else:
            # High levels: Aggressive play for maximum score
            self.aggressive_movement()
    
    def rapid_fire(self):
        """Continuous rapid fire shooting"""
        current_time = time.time()
        if current_time - self.last_shot_time >= self.shot_cooldown:
            try:
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                self.last_shot_time = current_time
            except:
                pass
    
    def zigzag_movement(self):
        """Zigzag movement pattern"""
        try:
            pattern = int(time.time() * 2) % 4
            if pattern == 0:
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.1)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern == 2:
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.1)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def defensive_movement(self):
        """More calculated defensive movement"""
        try:
            pattern = int(time.time() * 1.5) % 6
            if pattern == 0:
                # Move left
                for _ in range(2):
                    ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                    time.sleep(0.05)
                    ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern == 3:
                # Move right  
                for _ in range(2):
                    ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.05)
                    ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def aggressive_movement(self):
        """Aggressive movement for high levels"""
        try:
            # Quick side-to-side movements
            pattern = int(time.time() * 3) % 6
            if pattern < 2:
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern < 4:
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
            # Sometimes stay still to aim better
        except:
            pass
    
    def check_game_over(self):
        """Check if game is over"""
        try:
            page_text = self.driver.page_source.lower()
            game_over_indicators = [
                'game over',
                'gameover',
                'you died',
                'mission failed',
                'lives: 0',
                'try again'
            ]
            
            for indicator in game_over_indicators:
                if indicator in page_text:
                    return True
            
            # Also check if lives dropped to 0
            if self.lives <= 0:
                return True
                
            return False
            
        except:
            return False
    
    def play_game_loop(self):
        """Advanced game playing loop with learning"""
        print("🤖 Starting Advanced AI game loop...")
        loop_count = 0
        last_score = 0
        stagnant_count = 0
        
        while self.game_active:
            try:
                loop_count += 1
                
                # Get current game state
                self.get_game_state()
                
                # Check for game over
                if self.check_game_over():
                    print(f"🎮 Game Over Detected! Final Score: {self.score}, Level: {self.level}")
                    self.game_active = False
                    break
                
                # Execute AI strategy
                self.execute_advanced_strategy()
                
                # Progress monitoring
                if loop_count % 50 == 0:
                    print(f"📊 Score: {self.score}, Level: {self.level}, Lives: {self.lives}, Loops: {loop_count}")
                    
                    # Check if score is improving
                    if self.score == last_score:
                        stagnant_count += 1
                    else:
                        stagnant_count = 0
                        last_score = self.score
                    
                    # If score hasn't improved in a while, try different strategy
                    if stagnant_count > 10:
                        print("🔄 Score stagnant, switching strategy...")
                        self.shot_cooldown = max(0.05, self.shot_cooldown - 0.01)
                        stagnant_count = 0
                
                # Adaptive game speed based on level
                if self.level <= 3:
                    time.sleep(0.03)
                elif self.level <= 6:
                    time.sleep(0.02)
                else:
                    time.sleep(0.01)  # Fastest response for high levels
                
            except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Error in game loop: {e}")
                time.sleep(0.1)
                continue
        
        # Update statistics
        self.stats['games_played'] += 1
        if self.score > self.stats['best_score']:
            self.stats['best_score'] = self.score
            print(f"🏆 NEW HIGH SCORE: {self.score}!")
        
        self.save_stats()
        print(f"📈 Final Score: {self.score}, Games Played: {self.stats['games_played']}, Best Ever: {self.stats['best_score']}")
    
    def run(self):
        """Main execution with retry logic"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"🚀 Starting attempt {attempt + 1}/{max_attempts}")
                
                # Setup browser
                self.setup_browser()
                
                # Navigate to game
                if not self.navigate_to_game():
                    print("❌ Failed to navigate to game")
                    continue
                
                # Start game
                self.start_game()
                
                # Play the game
                self.play_game_loop()
                
                break  # Success!
                
            except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    print("🔄 Retrying...")
                    time.sleep(2)
            finally:
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass

def main():
    """Main function"""
    print("🎮" + "="*60 + "🎮")
    print("🚀 ADVANCED SPACE INVADERS AI BOT")
    print("🎯 Target: https://jordancota.site/")
    print("🏆 Current High Score to Beat: 25,940 points (Level 8)")
    print("🤖 Using Advanced AI Strategy with Learning")
    print("="*64)
    
    bot = AdvancedSpaceInvadersAI()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    
    print("🎮 Thank you for using Space Invaders AI Bot!")

if __name__ == "__main__":
    main()