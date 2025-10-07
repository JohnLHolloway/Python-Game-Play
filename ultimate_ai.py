"""
Ultimate Space Invaders AI - Final Optimized Version
Only saves scores when beating the current leaderboard high score
Uses "John H" as the champion name
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

class UltimateSpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.score = 0
        self.level = 1
        self.lives = 3
        self.current_high_score = 25940  # John H's current record
        self.leaderboard_beaten = False
        self.name_entered = False
        self.session_start_time = 0
        
    def setup_browser(self):
        """Setup optimized browser for high performance gaming"""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1400,1000")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-hang-monitor")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(30)
        
    def get_current_leaderboard_high_score(self):
        """Extract the current high score from the leaderboard"""
        try:
            # Look for high scores section
            page_source = self.driver.page_source
            
            # Find the highest score in the high scores section
            high_score_patterns = [
                r'(\d+,?\d*)\s*\(Level\s*\d+\)',  # Format: "25940 (Level 8)"
                r'(\d+,?\d*)\s*points?',          # Format: "25940 points"
                r'(\d+,?\d*)\s*-\s*\d+',         # Format: "25940 - Level 8"
            ]
            
            highest_found = 0
            for pattern in high_score_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    try:
                        score_str = match.replace(',', '')  # Remove comma separators
                        score_val = int(score_str)
                        if score_val > highest_found and score_val < 1000000:  # Reasonable range
                            highest_found = score_val
                    except:
                        continue
            
            if highest_found > 0:
                self.current_high_score = highest_found
                print(f"📊 Current leaderboard high score: {self.current_high_score}")
            
            return self.current_high_score
            
        except Exception as e:
            print(f"Error reading leaderboard: {e}")
            return self.current_high_score
        
    def navigate_and_start(self):
        """Navigate to game and check leaderboard"""
        print("🎮 Ultimate Space Invaders AI - Loading game...")
        self.driver.get("https://jordancota.site/")
        time.sleep(5)
        
        # Check current leaderboard high score
        self.get_current_leaderboard_high_score()
        
        # Scroll to game
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
        time.sleep(3)
        
        # Find game canvas
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
        self.start_game()
        return True
    
    def start_game(self):
        """Start the game with multiple methods"""
        print("🚀 Starting ultimate gaming session...")
        
        # Start button
        try:
            start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            start_btn.click()
            print("✅ Start button clicked")
        except:
            pass
        
        # Canvas interaction
        try:
            self.game_canvas.click()
            ActionChains(self.driver).move_to_element(self.game_canvas).click().perform()
        except:
            pass
        
        # Activation keys
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER).perform()
            time.sleep(0.5)
            actions.send_keys(Keys.SPACE).perform()
        except:
            pass
        
        self.session_start_time = time.time()
        time.sleep(3)
    
    def execute_optimal_strategy(self):
        """Execute the most effective strategy based on all our testing"""
        current_time = time.time()
        
        # Continuous rapid fire (most important for score)
        ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        
        # Optimized movement based on level
        if self.level <= 2:
            self.early_game_movement(current_time)
        elif self.level <= 5:
            self.mid_game_movement(current_time)
        else:
            self.expert_movement(current_time)
    
    def early_game_movement(self, current_time):
        """Aggressive movement for early levels"""
        pattern = int(current_time * 2.5) % 10
        
        try:
            if pattern < 2:
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern > 7:
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.08)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def mid_game_movement(self, current_time):
        """Balanced movement for mid levels"""
        pattern = int(current_time * 2) % 12
        
        try:
            if pattern < 3:
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.07)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern > 8:
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.07)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def expert_movement(self, current_time):
        """Precise movement for high levels"""
        pattern = int(current_time * 3) % 16
        
        try:
            if pattern < 2:
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.06)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif pattern > 13:
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.06)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
        except:
            pass
    
    def update_game_state(self):
        """Update game state and check for leaderboard-beating score"""
        try:
            page_source = self.driver.page_source
            
            # Extract score
            score_matches = re.findall(r'Score[:\s]*(\d+)', page_source, re.IGNORECASE)
            if score_matches:
                new_score = int(score_matches[-1])
                if new_score > self.score:
                    self.score = new_score
                    
                    # Check if we beat the leaderboard
                    if self.score > self.current_high_score and not self.leaderboard_beaten:
                        print(f"🏆🏆🏆 LEADERBOARD BEATEN! {self.score} > {self.current_high_score}! 🏆🏆🏆")
                        self.leaderboard_beaten = True
                    
                    # Progress milestones
                    milestones = [1000, 5000, 10000, 15000, 20000, 25000]
                    for milestone in milestones:
                        if self.score >= milestone and (self.score - new_score + self.score) < milestone:
                            print(f"🎯 Milestone: {milestone} points!")
                            if milestone >= 20000:
                                remaining = self.current_high_score - self.score
                                print(f"🔥 Only {remaining} points from beating leaderboard!")
            
            # Extract level
            level_matches = re.findall(r'Level[:\s]*(\d+)', page_source, re.IGNORECASE)
            if level_matches:
                new_level = int(level_matches[-1])
                if new_level > self.level:
                    self.level = new_level
                    print(f"📈 Level {self.level} reached!")
            
            # Extract lives
            lives_matches = re.findall(r'Lives[:\s]*(\d+)', page_source, re.IGNORECASE)
            if lives_matches:
                self.lives = int(lives_matches[-1])
                
        except Exception as e:
            pass
    
    def check_for_high_score_entry(self):
        """Check if we need to enter name for beating leaderboard"""
        if not self.leaderboard_beaten or self.name_entered:
            return False
        
        try:
            page_text = self.driver.page_source.lower()
            
            # Look for high score name entry indicators
            entry_indicators = [
                'enter your name',
                'enter name',
                'your name',
                'new high score',
                'new record',
                'congratulations',
                'well done',
                'high score'
            ]
            
            for indicator in entry_indicators:
                if indicator in page_text:
                    print(f"🎉 High score name entry detected: '{indicator}'")
                    return True
            
            # Look for input fields
            try:
                input_elements = self.driver.find_elements(By.TAG_NAME, "input")
                if input_elements:
                    for element in input_elements:
                        if element.is_displayed() and element.is_enabled():
                            print("📝 Input field found for high score entry")
                            return True
            except:
                pass
            
            return False
            
        except:
            return False
    
    def enter_leaderboard_name(self):
        """Enter 'John H' for the new leaderboard high score"""
        if self.name_entered:
            return
        
        try:
            print("🏆 ENTERING 'John H' FOR NEW LEADERBOARD HIGH SCORE!")
            
            # Find input field
            input_field = None
            selectors = [
                "input[type='text']",
                "input",
                "textarea",
                "*[contenteditable='true']"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            input_field = element
                            break
                    if input_field:
                        break
                except:
                    continue
            
            champion_name = "John H"
            
            if input_field:
                input_field.clear()
                time.sleep(0.5)
                input_field.send_keys(champion_name)
                time.sleep(1)
                input_field.send_keys(Keys.ENTER)
                print(f"✅ Leaderboard name entered: {champion_name}")
            else:
                # Direct typing fallback
                ActionChains(self.driver).send_keys(champion_name).perform()
                time.sleep(1)
                ActionChains(self.driver).send_keys(Keys.ENTER).perform()
                print(f"✅ Leaderboard name typed: {champion_name}")
            
            self.name_entered = True
            
            # Take victory screenshot
            self.driver.save_screenshot(f"LEADERBOARD_CHAMPION_{self.score}.png")
            print(f"📸 Champion screenshot saved!")
            
            # Wait to confirm
            time.sleep(5)
            
        except Exception as e:
            print(f"Name entry error: {e}")
    
    def is_game_over(self):
        """Conservative game over detection"""
        try:
            page_text = self.driver.page_source.lower()
            game_over_indicators = ['game over', 'you died', 'mission failed']
            
            indicators_found = sum(1 for indicator in game_over_indicators if indicator in page_text)
            
            # Only declare game over with strong evidence
            return indicators_found >= 1 and self.lives == 0
            
        except:
            return False
    
    def play_ultimate_game(self):
        """Ultimate game loop optimized for leaderboard beating"""
        print("🎮 ULTIMATE AI GAMING SESSION STARTED!")
        print(f"🎯 Target: Beat current high score of {self.current_high_score}")
        print("🏆 Will only save score if it beats the leaderboard!")
        
        loop_count = 0
        last_report = time.time()
        max_loops = 75000  # Extended for high score attempts
        
        while loop_count < max_loops:
            try:
                loop_count += 1
                current_time = time.time()
                
                # Execute optimal strategy
                self.execute_optimal_strategy()
                
                # Update game state every 50 loops
                if loop_count % 50 == 0:
                    self.update_game_state()
                
                # Progress reports
                if loop_count % 1000 == 0:
                    session_time = current_time - self.session_start_time
                    points_per_second = self.score / max(1, session_time)
                    
                    print(f"📊 Loop {loop_count}: Score={self.score}, Level={self.level}, Lives={self.lives}")
                    print(f"⏱️ Time: {session_time:.1f}s, Rate: {points_per_second:.1f} pts/sec")
                    
                    if self.score > 0:
                        progress = (self.score / self.current_high_score) * 100
                        print(f"🎯 Progress: {progress:.1f}% toward leaderboard")
                    
                    last_report = current_time
                
                # Check for high score name entry (only if we beat leaderboard)
                if loop_count % 200 == 0:
                    if self.check_for_high_score_entry():
                        self.enter_leaderboard_name()
                
                # Game over check
                if loop_count % 300 == 0:
                    if self.is_game_over():
                        print("🎮 Game over detected")
                        break
                
                # Adaptive timing
                if self.score > 20000:
                    time.sleep(0.008)  # Fastest for high scores
                elif self.score > 10000:
                    time.sleep(0.012)
                elif self.score > 5000:
                    time.sleep(0.015)
                else:
                    time.sleep(0.02)
                
            except KeyboardInterrupt:
                print("🛑 Ultimate AI stopped by user")
                break
            except:
                continue
        
        # Final results
        session_time = time.time() - self.session_start_time
        
        print(f"\n🏁 ULTIMATE SESSION COMPLETED!")
        print(f"🏆 Final Score: {self.score}")
        print(f"📊 Final Level: {self.level}")
        print(f"⏱️ Session Time: {session_time:.1f} seconds")
        print(f"🔄 Total Loops: {loop_count}")
        print(f"🎯 Leaderboard Beaten: {self.leaderboard_beaten}")
        print(f"📝 Name Entered: {self.name_entered}")
        
        if self.leaderboard_beaten:
            print(f"🎉 CONGRATULATIONS! New leaderboard record: {self.score} points!")
        else:
            gap = self.current_high_score - self.score
            print(f"📈 {gap} points needed to beat leaderboard")
        
        return self.score
    
    def run_ultimate_session(self):
        """Run the ultimate high score session"""
        try:
            print("🚀 ULTIMATE SPACE INVADERS AI - INITIALIZING...")
            
            self.setup_browser()
            
            if not self.navigate_and_start():
                return 0
            
            final_score = self.play_ultimate_game()
            
            # Only take screenshot if we beat the leaderboard
            if self.leaderboard_beaten:
                self.driver.save_screenshot(f"NEW_LEADERBOARD_RECORD_{final_score}.png")
                print("📸 New record screenshot saved!")
            
            return final_score
            
        except Exception as e:
            print(f"❌ Ultimate session error: {e}")
            return 0
        finally:
            # Extended monitoring only if we beat leaderboard
            if self.leaderboard_beaten and not self.name_entered:
                print("🏆 Monitoring for leaderboard name entry...")
                
                for i in range(6):  # 60 seconds
                    time.sleep(10)
                    try:
                        if self.check_for_high_score_entry():
                            self.enter_leaderboard_name()
                            break
                    except:
                        pass
                    print(f"🔄 Monitoring... {(i+1)*10}s")
            
            if self.driver:
                if self.leaderboard_beaten:
                    print("🎉 New leaderboard record! Closing in 30 seconds...")
                    time.sleep(30)
                self.driver.quit()

def main():
    """Main function for ultimate AI"""
    print("🏆" + "="*60 + "🏆")
    print("🎮 ULTIMATE SPACE INVADERS AI")
    print("🎯 Mission: Beat the leaderboard and save as 'John H'")
    print("📋 Only saves scores that beat the current high score")
    print("="*64)
    
    # Run single ultimate session
    print("🚀 Starting ultimate leaderboard challenge...")
    
    ai = UltimateSpaceInvadersAI()
    final_score = ai.run_ultimate_session()
    
    print("\n" + "🏆" + "="*60 + "🏆")
    print("🏁 ULTIMATE AI RESULTS")
    print("🏆" + "="*60 + "🏆")
    print(f"🏆 Final Score: {final_score}")
    print(f"🎯 Leaderboard Beaten: {ai.leaderboard_beaten}")
    print(f"📝 Name Saved: {'John H' if ai.name_entered else 'No (score did not beat leaderboard)'}")
    
    if ai.leaderboard_beaten:
        print("🎉 SUCCESS! New leaderboard champion!")
        print("👑 'John H' is now the high score holder!")
    else:
        print("📈 Score not high enough to beat leaderboard")
        print("🎯 No name entry needed - leaderboard unchanged")
    
    print("\n🤖 Ultimate Space Invaders AI - Mission Complete!")

if __name__ == "__main__":
    main()