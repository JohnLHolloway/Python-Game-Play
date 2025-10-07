"""
Advanced Bullet-Dodging Space Invaders AI
Focus on bullet detection, evasive maneuvers, and high score name entry
"""

import time
import re
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
import pyautogui

class BulletDodgingAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.score = 0
        self.level = 1
        self.lives = 3
        self.last_shot_time = 0
        self.shot_cooldown = 0.05
        self.player_position = 0.5  # Normalized position (0.0 = left, 1.0 = right)
        self.dodge_direction = 1  # 1 = right, -1 = left
        self.last_dodge_time = 0
        self.high_score_achieved = False
        
    def setup_browser(self):
        """Setup browser optimized for computer vision and long sessions"""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1400,1000")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Keep window open for potential high score entry
        self.driver.implicitly_wait(30)
        
    def navigate_and_start_game(self):
        """Navigate to game with enhanced detection"""
        print("🎮 Loading Space Invaders for bullet-dodging AI...")
        self.driver.get("https://jordancota.site/")
        time.sleep(5)
        
        # Scroll to game with precision
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
        time.sleep(3)
        
        # Find and focus on game canvas
        try:
            self.game_canvas = self.driver.find_element(By.ID, "gameCanvas")
            print("✅ Game canvas located for bullet detection")
        except:
            canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                self.game_canvas = canvases[0]
                print("✅ Canvas found for bullet detection")
            else:
                return False
        
        # Start game with multiple methods
        self.start_game_sequence()
        return True
    
    def start_game_sequence(self):
        """Enhanced game start with focus on keeping session alive"""
        print("🚀 Starting enhanced game session...")
        
        # Method 1: Start button
        try:
            start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            start_btn.click()
            print("✅ Start button clicked")
        except:
            pass
        
        # Method 2: Canvas focus and interaction
        try:
            self.game_canvas.click()
            ActionChains(self.driver).move_to_element(self.game_canvas).click().perform()
            print("✅ Canvas focused for bullet dodging")
        except:
            pass
        
        # Method 3: Activation keys
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER).perform()
            time.sleep(0.5)
            actions.send_keys(Keys.SPACE).perform()
            print("✅ Activation keys sent")
        except:
            pass
        
        time.sleep(3)  # Allow game to fully initialize
    
    def capture_game_screen(self):
        """Capture game screen for bullet detection"""
        try:
            # Get canvas position and size
            canvas_location = self.game_canvas.location
            canvas_size = self.game_canvas.size
            
            # Capture screenshot of game area
            screenshot = pyautogui.screenshot(region=(
                canvas_location['x'],
                canvas_location['y'], 
                canvas_size['width'],
                canvas_size['height']
            ))
            
            # Convert to numpy array for processing
            screen_array = np.array(screenshot)
            return screen_array
            
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def detect_bullets_and_threats(self, screen_array):
        """Detect enemy bullets and calculate threat zones"""
        if screen_array is None:
            return {'bullets': [], 'safe_zones': [0.3, 0.7], 'immediate_threat': False}
        
        try:
            # Convert to grayscale for easier processing
            gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Define player area (bottom 20% of screen)
            player_area_start = int(height * 0.8)
            player_area = gray[player_area_start:, :]
            
            # Detect bullets using edge detection and contour finding
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bullets = []
            for contour in contours:
                # Filter for bullet-like objects (small, fast-moving)
                area = cv2.contourArea(contour)
                if 5 < area < 100:  # Bullet size range
                    x, y, w, h = cv2.boundingRect(contour)
                    # Normalize coordinates
                    norm_x = x / width
                    norm_y = y / height
                    
                    # Only consider objects in the middle/lower part of screen
                    if norm_y > 0.3:
                        bullets.append({
                            'x': norm_x,
                            'y': norm_y,
                            'width': w / width,
                            'height': h / height
                        })
            
            # Calculate safe zones (areas with fewer bullets)
            safe_zones = self.calculate_safe_zones(bullets, width)
            
            # Determine immediate threat (bullets near player area)
            immediate_threat = any(bullet['y'] > 0.7 for bullet in bullets)
            
            return {
                'bullets': bullets,
                'safe_zones': safe_zones,
                'immediate_threat': immediate_threat,
                'bullet_count': len(bullets)
            }
            
        except Exception as e:
            print(f"Bullet detection error: {e}")
            return {'bullets': [], 'safe_zones': [0.3, 0.7], 'immediate_threat': False}
    
    def calculate_safe_zones(self, bullets, screen_width):
        """Calculate safe zones to move to"""
        # Divide screen into zones
        zones = [0.1, 0.3, 0.5, 0.7, 0.9]
        zone_safety = {}
        
        for zone in zones:
            # Count nearby bullets
            nearby_bullets = 0
            for bullet in bullets:
                distance = abs(bullet['x'] - zone)
                if distance < 0.15:  # Within danger radius
                    nearby_bullets += 1
            
            zone_safety[zone] = nearby_bullets
        
        # Return zones sorted by safety (fewer bullets = safer)
        safe_zones = sorted(zones, key=lambda z: zone_safety[z])
        return safe_zones[:3]  # Return top 3 safest zones
    
    def execute_bullet_dodging_strategy(self, threat_analysis):
        """Execute advanced bullet dodging and shooting strategy"""
        current_time = time.time()
        
        # Priority 1: Continuous rapid fire
        if current_time - self.last_shot_time >= self.shot_cooldown:
            try:
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                self.last_shot_time = current_time
            except:
                pass
        
        # Priority 2: Bullet dodging
        if threat_analysis['immediate_threat'] or len(threat_analysis['bullets']) > 3:
            self.execute_evasive_maneuvers(threat_analysis)
        else:
            # Normal movement pattern when safe
            self.execute_standard_movement()
    
    def execute_evasive_maneuvers(self, threat_analysis):
        """Execute evasive maneuvers to dodge bullets"""
        current_time = time.time()
        
        # Don't dodge too frequently (allow time for movement)
        if current_time - self.last_dodge_time < 0.2:
            return
        
        try:
            safe_zones = threat_analysis['safe_zones']
            
            if safe_zones:
                # Move to the safest zone
                target_zone = safe_zones[0]
                
                if target_zone < self.player_position - 0.1:
                    # Move left to safe zone
                    ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                    time.sleep(0.08)
                    ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
                    self.player_position = max(0.0, self.player_position - 0.1)
                    self.dodge_direction = -1
                    print(f"🔄 Dodging LEFT to safe zone {target_zone:.1f}")
                    
                elif target_zone > self.player_position + 0.1:
                    # Move right to safe zone
                    ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.08)
                    ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
                    self.player_position = min(1.0, self.player_position + 0.1)
                    self.dodge_direction = 1
                    print(f"🔄 Dodging RIGHT to safe zone {target_zone:.1f}")
            
            else:
                # Emergency dodge - quick side-to-side
                if self.dodge_direction == 1 and self.player_position < 0.9:
                    ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.06)
                    ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
                    self.player_position += 0.1
                else:
                    ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                    time.sleep(0.06)
                    ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
                    self.player_position -= 0.1
                    self.dodge_direction = -1
                
                print("⚡ Emergency dodge maneuver!")
            
            self.last_dodge_time = current_time
            
        except Exception as e:
            print(f"Dodge error: {e}")
    
    def execute_standard_movement(self):
        """Standard movement when no immediate threats"""
        try:
            movement_pattern = int(time.time() * 1.5) % 10
            
            if movement_pattern < 2 and self.player_position > 0.2:
                # Move left
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.05)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
                self.player_position = max(0.0, self.player_position - 0.05)
                
            elif movement_pattern > 7 and self.player_position < 0.8:
                # Move right
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.05)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
                self.player_position = min(1.0, self.player_position + 0.05)
            
            # Otherwise hold position for better accuracy
            
        except:
            pass
    
    def update_game_state(self):
        """Update game state and check for high score opportunity"""
        try:
            page_source = self.driver.page_source
            
            # Extract score
            score_matches = re.findall(r'Score[:\s]*(\d+)', page_source, re.IGNORECASE)
            if score_matches:
                new_score = int(score_matches[-1])
                if new_score > self.score:
                    self.score = new_score
                    
                    # Check if we're approaching high score territory
                    if self.score > 1000 and not self.high_score_achieved:
                        print(f"🎯 Great progress! Score: {self.score} - Getting closer to high score!")
                    
                    if self.score > 25940:
                        print(f"🏆 HIGH SCORE ACHIEVED! {self.score} points!")
                        self.high_score_achieved = True
            
            # Extract level
            level_matches = re.findall(r'Level[:\s]*(\d+)', page_source, re.IGNORECASE)
            if level_matches:
                self.level = int(level_matches[-1])
            
            # Extract lives
            lives_matches = re.findall(r'Lives[:\s]*(\d+)', page_source, re.IGNORECASE)
            if lives_matches:
                self.lives = int(lives_matches[-1])
                
        except Exception as e:
            pass
    
    def check_for_name_entry(self):
        """Check if we need to enter our name for high score"""
        try:
            page_text = self.driver.page_source.lower()
            
            # Look for name entry prompts
            name_prompts = [
                'enter your name',
                'enter name',
                'your name',
                'high score',
                'new record',
                'congratulations'
            ]
            
            for prompt in name_prompts:
                if prompt in page_text:
                    print(f"🎉 HIGH SCORE NAME ENTRY DETECTED!")
                    return True
            
            # Look for input fields that might be for name entry
            try:
                input_fields = self.driver.find_elements(By.TAG_NAME, "input")
                if input_fields:
                    print("🎯 Input field found - might be for name entry")
                    return True
            except:
                pass
            
            return False
            
        except:
            return False
    
    def enter_high_score_name(self):
        """Enter name for high score"""
        try:
            print("🏆 Entering name for high score...")
            
            # Try to find input field
            input_field = None
            try:
                input_field = self.driver.find_element(By.TAG_NAME, "input")
            except:
                try:
                    input_field = self.driver.find_element(By.XPATH, "//input[@type='text']")
                except:
                    pass
            
            if input_field:
                input_field.clear()
                input_field.send_keys("AI_BOT_JLH")
                input_field.send_keys(Keys.ENTER)
                print("✅ Name entered: AI_BOT_JLH")
                
                # Wait to see if it was accepted
                time.sleep(5)
            else:
                # Try typing directly
                ActionChains(self.driver).send_keys("AI_BOT_JLH").perform()
                time.sleep(1)
                ActionChains(self.driver).send_keys(Keys.ENTER).perform()
                print("✅ Name typed directly: AI_BOT_JLH")
                
        except Exception as e:
            print(f"Name entry error: {e}")
    
    def is_game_over(self):
        """Conservative game over detection"""
        try:
            page_text = self.driver.page_source.lower()
            
            # Check for definitive game over indicators
            game_over_phrases = ['game over', 'you died', 'mission failed']
            
            indicators_found = 0
            for phrase in game_over_phrases:
                if phrase in page_text:
                    indicators_found += 1
            
            # Only declare game over if we have strong evidence
            if indicators_found >= 1 and self.lives == 0:
                return True
            elif indicators_found >= 2:
                return True
            
            return False
            
        except:
            return False
    
    def play_enhanced_game(self):
        """Main enhanced gameplay loop with bullet dodging"""
        print("🤖 Starting BULLET-DODGING AI gameplay...")
        print("🎯 Focus: Dodge bullets, survive longer, beat 25,940 points!")
        
        loop_count = 0
        last_progress_report = time.time()
        last_screen_capture = 0
        max_loops = 50000  # Extended play for high scores
        
        while loop_count < max_loops:
            try:
                loop_count += 1
                current_time = time.time()
                
                # Capture screen for bullet detection every few loops
                threat_analysis = {'bullets': [], 'safe_zones': [0.3, 0.7], 'immediate_threat': False}
                
                if current_time - last_screen_capture > 0.1:  # 10 FPS screen analysis
                    screen = self.capture_game_screen()
                    threat_analysis = self.detect_bullets_and_threats(screen)
                    last_screen_capture = current_time
                
                # Execute bullet dodging strategy
                self.execute_bullet_dodging_strategy(threat_analysis)
                
                # Update game state every 100 loops
                if loop_count % 100 == 0:
                    self.update_game_state()
                
                # Progress reports
                if loop_count % 500 == 0:
                    elapsed = current_time - last_progress_report
                    bullets_detected = len(threat_analysis['bullets'])
                    
                    print(f"📊 Loop {loop_count}: Score={self.score}, Level={self.level}, Lives={self.lives}")
                    print(f"🎯 Bullets detected: {bullets_detected}, Position: {self.player_position:.2f}")
                    
                    last_progress_report = current_time
                
                # Check for high score name entry opportunity
                if loop_count % 200 == 0:
                    if self.check_for_name_entry():
                        self.enter_high_score_name()
                        # Don't end game, keep playing for even higher score!
                
                # Conservative game over check
                if loop_count % 300 == 0:
                    if self.is_game_over():
                        print("🎮 Game over confirmed")
                        break
                
                # Adaptive timing based on threat level
                if threat_analysis['immediate_threat']:
                    time.sleep(0.01)  # Faster response under threat
                elif len(threat_analysis['bullets']) > 5:
                    time.sleep(0.015)  # Medium response
                else:
                    time.sleep(0.02)  # Standard response
                
            except KeyboardInterrupt:
                print("🛑 Enhanced AI stopped by user")
                break
            except Exception as e:
                continue
        
        print(f"🏁 Enhanced bullet-dodging game completed!")
        print(f"🏆 Final Score: {self.score}")
        print(f"📊 Final Level: {self.level}")
        print(f"🔄 Total Loops: {loop_count}")
        print(f"🎯 High Score Achieved: {self.high_score_achieved}")
        
        # Keep window open for potential name entry
        if self.score > 100:  # Any decent score
            print("🔄 Keeping window open for potential high score entry...")
            time.sleep(10)
            
            # Final check for name entry
            if self.check_for_name_entry():
                self.enter_high_score_name()
                time.sleep(5)
        
        return self.score
    
    def run_enhanced_session(self):
        """Run enhanced bullet-dodging session"""
        try:
            self.setup_browser()
            
            if not self.navigate_and_start_game():
                return 0
            
            final_score = self.play_enhanced_game()
            
            # Save screenshot of final state
            self.driver.save_screenshot(f"bullet_dodging_final_{final_score}.png")
            print(f"📸 Final screenshot saved")
            
            return final_score
            
        except Exception as e:
            print(f"❌ Enhanced session error: {e}")
            return 0
        finally:
            # Keep browser open longer for high score entry
            if hasattr(self, 'score') and self.score > 200:
                print("🏆 Keeping browser open for high score opportunities...")
                time.sleep(30)
            
            if self.driver:
                self.driver.quit()

def main():
    """Main function for bullet-dodging AI"""
    print("🎮 BULLET-DODGING SPACE INVADERS AI")
    print("🎯 Mission: Beat 25,940 points with advanced bullet dodging")
    print("🤖 Features: Computer vision, evasive maneuvers, high score name entry")
    print("="*70)
    
    max_sessions = 3
    best_score = 0
    target_score = 25940
    
    for session in range(1, max_sessions + 1):
        print(f"\n🎮 ENHANCED SESSION {session}/{max_sessions}")
        print("🚀 " + "-" * 50 + " 🚀")
        
        ai = BulletDodgingAI()
        score = ai.run_enhanced_session()
        
        print(f"📊 Session {session} Result: {score} points")
        
        if score > best_score:
            best_score = score
            print(f"🏆 NEW PERSONAL BEST: {best_score} points!")
        
        if best_score > target_score:
            print("🎉🎉🎉 HIGH SCORE BEATEN! MISSION ACCOMPLISHED! 🎉🎉🎉")
            break
        
        if session < max_sessions:
            improvement_needed = target_score - best_score
            print(f"📈 Need {improvement_needed} more points to beat record")
            print("⏳ 20 second break before next enhanced session...")
            time.sleep(20)
    
    # Final results
    print("\n" + "="*70)
    print("🏁 BULLET-DODGING AI FINAL RESULTS")
    print("="*70)
    print(f"🏆 Best Score: {best_score} points")
    print(f"🎯 Target: {target_score} points")
    print(f"📊 Sessions: {max_sessions}")
    
    if best_score > target_score:
        print("🎉 SUCCESS! High score beaten with bullet-dodging AI!")
    elif best_score > target_score * 0.8:
        print("🔥 Excellent! Very close to beating the record!")
    elif best_score > target_score * 0.5:
        print("💪 Great progress! Bullet dodging is working!")
    else:
        print("📈 Good foundation! Continue optimizing bullet detection!")
    
    print("\n🤖 Thank you for using the Bullet-Dodging Space Invaders AI!")

if __name__ == "__main__":
    main()