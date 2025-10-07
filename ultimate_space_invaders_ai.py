"""
Ultimate Space Invaders AI - Optimized for Maximum Score
This version focuses on achieving the highest possible score with advanced strategies
"""

import time
import re
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class UltimateSpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_element = None
        self.score = 0
        self.high_score = 0
        self.level = 1
        self.lives = 3
        self.last_shot_time = 0
        self.shot_cooldown = 0.05  # Ultra-fast shooting
        self.movement_strategy = "aggressive"
        self.survival_mode = False
        
    def setup_browser(self):
        """Setup optimized browser for maximum performance"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--window-size=1600,1000")
        chrome_options.add_argument("--aggressive-cache-discard")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Maximize for better visibility
        self.driver.maximize_window()
        
    def navigate_to_game(self):
        """Navigate with maximum efficiency"""
        print("🚀 Loading Jordan Cota's Space Invaders...")
        self.driver.get("https://jordancota.site/")
        
        # Wait for complete load
        time.sleep(4)
        
        # Smart scroll to game section
        try:
            # Look for the exact game section
            game_section = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Fun Time: Play Space Invaders!')]")
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", game_section)
            time.sleep(2)
            print("✅ Found game section")
        except:
            # Fallback scroll
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
            time.sleep(2)
            print("✅ Scrolled to game area")
        
        # Find game canvas with priority methods
        canvas_found = False
        
        # Priority 1: gameCanvas ID
        try:
            self.game_element = self.driver.find_element(By.ID, "gameCanvas")
            canvas_found = True
            print("✅ Found gameCanvas by ID")
        except:
            pass
        
        # Priority 2: Any canvas near game section
        if not canvas_found:
            try:
                canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
                if canvases:
                    # Choose the largest canvas (likely the game)
                    largest_canvas = None
                    max_area = 0
                    
                    for canvas in canvases:
                        try:
                            size = canvas.size
                            area = size['width'] * size['height']
                            if area > max_area:
                                max_area = area
                                largest_canvas = canvas
                        except:
                            continue
                    
                    if largest_canvas:
                        self.game_element = largest_canvas
                        canvas_found = True
                        print(f"✅ Found canvas ({max_area} pixels)")
            except:
                pass
        
        if not canvas_found:
            print("❌ Could not locate game canvas")
            return False
        
        # Prepare game element for interaction
        try:
            self.driver.execute_script("arguments[0].focus();", self.game_element)
            time.sleep(0.5)
        except:
            pass
        
        return True
    
    def start_game_ultimate(self):
        """Ultimate game start sequence"""
        print("🎮 Initiating game start sequence...")
        
        successful_methods = []
        
        # Method 1: Direct start button click
        try:
            start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            if start_btn.is_displayed():
                start_btn.click()
                successful_methods.append("Start Button")
                time.sleep(1)
        except:
            pass
        
        # Method 2: Canvas activation sequence
        try:
            # Click to focus
            self.game_element.click()
            time.sleep(0.5)
            
            # Send activation keys
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER).perform()
            time.sleep(0.3)
            actions.send_keys(Keys.SPACE).perform()
            time.sleep(0.3)
            
            successful_methods.append("Canvas + Keys")
        except:
            pass
        
        # Method 3: JavaScript force start
        try:
            # Try to trigger any game start events
            self.driver.execute_script("""
                var canvas = arguments[0];
                canvas.focus();
                canvas.click();
                
                // Simulate keypress events
                var events = ['keydown', 'keyup'];
                var keys = [13, 32]; // Enter and Space
                
                keys.forEach(function(keyCode) {
                    events.forEach(function(eventType) {
                        var event = new KeyboardEvent(eventType, {
                            bubbles: true,
                            cancelable: true,
                            keyCode: keyCode,
                            which: keyCode
                        });
                        canvas.dispatchEvent(event);
                    });
                });
            """, self.game_element)
            
            successful_methods.append("JavaScript Events")
        except:
            pass
        
        print(f"✅ Start methods executed: {', '.join(successful_methods)}")
        
        # Give game time to initialize
        time.sleep(2)
        return True
    
    def extract_game_state(self):
        """Advanced game state extraction"""
        try:
            page_source = self.driver.page_source
            
            # Score extraction with multiple patterns
            score_patterns = [
                r'Score[:\s]*(\d+)',
                r'score[:\s]*(\d+)',
                r'SCORE[:\s]*(\d+)',
                r'Points[:\s]*(\d+)',
                r'>(\d+)</.*?score',
                r'score.*?>(\d+)<'
            ]
            
            highest_score = 0
            for pattern in score_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    try:
                        score_val = int(match)
                        if score_val > highest_score and score_val < 1000000:  # Reasonable score range
                            highest_score = score_val
                    except:
                        continue
            
            if highest_score > self.score:
                self.score = highest_score
            
            # Level extraction
            level_patterns = [r'Level[:\s]*(\d+)', r'level[:\s]*(\d+)', r'LEVEL[:\s]*(\d+)']
            for pattern in level_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    try:
                        self.level = int(matches[-1])
                        break
                    except:
                        continue
            
            # Lives extraction
            lives_patterns = [r'Lives[:\s]*(\d+)', r'lives[:\s]*(\d+)', r'LIVES[:\s]*(\d+)']
            for pattern in lives_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    try:
                        self.lives = int(matches[-1])
                        break
                    except:
                        continue
            
            return True
            
        except Exception as e:
            return False
    
    def execute_ultimate_strategy(self):
        """Execute the ultimate gaming strategy"""
        current_time = time.time()
        
        # Ultra-rapid fire (primary strategy)
        if current_time - self.last_shot_time >= self.shot_cooldown:
            try:
                # Double-tap for maximum fire rate
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                self.last_shot_time = current_time
            except:
                pass
        
        # Advanced movement patterns based on game state
        self.execute_movement_pattern()
    
    def execute_movement_pattern(self):
        """Execute sophisticated movement patterns"""
        current_time = time.time()
        
        # Choose pattern based on level and score
        if self.level <= 2:
            self.aggressive_sweep_pattern(current_time)
        elif self.level <= 5:
            self.tactical_weave_pattern(current_time)
        else:
            self.survival_expert_pattern(current_time)
    
    def aggressive_sweep_pattern(self, current_time):
        """Aggressive sweeping for early levels"""
        pattern = int(current_time * 3) % 6
        
        try:
            if pattern == 0:
                # Quick left sweep
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.12)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern == 3:
                # Quick right sweep
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.12)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def tactical_weave_pattern(self, current_time):
        """Tactical weaving for mid levels"""
        pattern = int(current_time * 2.5) % 8
        
        try:
            if pattern in [0, 1]:
                # Left weave
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.1)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern in [4, 5]:
                # Right weave
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.1)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
            # else: hold position for better accuracy
        except:
            pass
    
    def survival_expert_pattern(self, current_time):
        """Expert survival pattern for high levels"""
        pattern = int(current_time * 4) % 10
        
        try:
            if pattern < 2:
                # Quick left dodge
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern >= 5 and pattern < 7:
                # Quick right dodge
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
            # else: minimal movement for precision
        except:
            pass
    
    def check_game_active(self):
        """Advanced game state checking"""
        try:
            page_text = self.driver.page_source.lower()
            
            # Game over detection
            game_over_indicators = [
                'game over',
                'gameover', 
                'game_over',
                'you died',
                'mission failed',
                'try again',
                'play again',
                'restart',
                'game ended'
            ]
            
            for indicator in game_over_indicators:
                if indicator in page_text:
                    return False
            
            # Additional checks
            if self.lives <= 0:
                return False
            
            return True
            
        except:
            return True
    
    def play_ultimate_game(self):
        """Ultimate game playing loop"""
        print("🚀 ULTIMATE AI GAMEPLAY INITIATED")
        print("🎯 Target: Maximum score possible")
        
        loop_count = 0
        last_score_time = time.time()
        max_loops = 12000  # Extended gameplay time
        score_milestones = [100, 500, 1000, 2500, 5000, 10000, 25000]
        
        while loop_count < max_loops:
            try:
                loop_count += 1
                
                # Execute ultimate strategy
                self.execute_ultimate_strategy()
                
                # Monitor progress every 25 loops
                if loop_count % 25 == 0:
                    self.extract_game_state()
                    
                    # Check for score milestones
                    for milestone in score_milestones:
                        if self.score >= milestone and milestone not in [m for m in score_milestones if self.score >= m and self.score != milestone]:
                            print(f"🏆 MILESTONE REACHED: {self.score} points! (Level {self.level})")
                            last_score_time = time.time()
                    
                    # Regular progress updates
                    if loop_count % 100 == 0:
                        print(f"📊 Score: {self.score} | Level: {self.level} | Lives: {self.lives} | Loop: {loop_count}")
                        
                        # Update high score
                        if self.score > self.high_score:
                            self.high_score = self.score
                    
                    # Check if game is still active
                    if not self.check_game_active():
                        print("🎮 Game over detected!")
                        break
                
                # Adaptive speed based on performance
                if self.level <= 3:
                    time.sleep(0.02)
                elif self.level <= 6:
                    time.sleep(0.015)
                else:
                    time.sleep(0.01)  # Maximum speed for high levels
                
            except KeyboardInterrupt:
                print("🛑 Ultimate AI stopped by user")
                break
            except Exception as e:
                continue
        
        print(f"🏁 ULTIMATE GAME COMPLETED!")
        print(f"🏆 Final Score: {self.score}")
        print(f"📊 Reached Level: {self.level}")
        print(f"🎮 Total Loops: {loop_count}")
        
        return self.score
    
    def run_ultimate_session(self):
        """Run the ultimate gaming session"""
        try:
            print("🚀 ULTIMATE SPACE INVADERS AI INITIALIZING...")
            
            # Setup
            self.setup_browser()
            
            # Navigate
            if not self.navigate_to_game():
                return 0
            
            # Start
            if not self.start_game_ultimate():
                return 0
            
            # Play
            final_score = self.play_ultimate_game()
            
            return final_score
            
        except Exception as e:
            print(f"💥 Ultimate AI Error: {e}")
            return 0
        finally:
            if self.driver:
                try:
                    self.driver.save_screenshot("ultimate_final_state.png")
                    print("📸 Ultimate session screenshot saved")
                except:
                    pass
                time.sleep(2)
                self.driver.quit()

def run_ultimate_challenge():
    """Run the ultimate high-score challenge"""
    print("🎮" + "="*60 + "🎮")
    print("🚀 ULTIMATE SPACE INVADERS AI CHALLENGE")
    print("🎯 TARGET: BEAT 25,940 POINTS")
    print("🏆 CURRENT RECORD HOLDER: John H (Level 8)")
    print("🤖 STRATEGY: MAXIMUM AGGRESSION + PRECISION")
    print("="*64)
    
    max_attempts = 5
    best_score = 0
    best_level = 0
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n🎮 ULTIMATE ATTEMPT {attempt}/{max_attempts}")
        print("🚀 " + "-" * 40 + " 🚀")
        
        ai = UltimateSpaceInvadersAI()
        score = ai.run_ultimate_session()
        
        print(f"📊 Attempt {attempt} Result: {score} points")
        
        if score > best_score:
            best_score = score
            best_level = ai.level
            print(f"🏆 NEW PERSONAL BEST: {best_score} points (Level {best_level})!")
        
        # Check if we beat the target
        if best_score > 25940:
            print("🎉🎉🎉 TARGET ACHIEVED! HIGH SCORE BEATEN! 🎉🎉🎉")
            break
        
        # Brief pause between attempts
        if attempt < max_attempts:
            print("⏳ Preparing for next attempt...")
            time.sleep(3)
    
    # Final results
    print("\n🏁 ULTIMATE CHALLENGE RESULTS:")
    print("="*50)
    print(f"🏆 Best Score Achieved: {best_score} points")
    print(f"📊 Best Level Reached: {best_level}")
    print(f"🎯 Target Score: 25,940 points")
    print(f"📈 Gap to Target: {max(0, 25940 - best_score)} points")
    
    if best_score > 25940:
        print("🎉 MISSION ACCOMPLISHED! HIGH SCORE BEATEN!")
        print("🏆 You are the new Space Invaders champion!")
    elif best_score > 20000:
        print("🔥 EXCELLENT PERFORMANCE! Very close to the target!")
    elif best_score > 10000:
        print("💪 GREAT SCORE! Keep pushing for the record!")
    else:
        print("📈 Good start! More optimization needed for the record.")
    
    print("🤖 Thank you for using Ultimate Space Invaders AI!")

if __name__ == "__main__":
    run_ultimate_challenge()