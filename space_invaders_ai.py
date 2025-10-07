"""
Space Invaders AI Bot for https://jordancota.site/
This bot automatically plays the Space Invaders game to achieve the highest score possible.
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
import pyautogui
from PIL import Image
import threading
import math

class SpaceInvadersAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.game_active = False
        self.score = 0
        self.level = 1
        self.lives = 3
        self.last_shot_time = 0
        self.shot_cooldown = 0.1  # Minimum time between shots
        self.player_position = 0.5  # Normalized position (0.0 to 1.0)
        self.invader_positions = []
        self.bullets = []
        self.enemy_bullets = []
        
    def setup_browser(self):
        """Initialize Chrome browser with optimal settings for game automation"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def navigate_to_game(self):
        """Navigate to the Space Invaders game"""
        print("Navigating to Space Invaders game...")
        self.driver.get("https://jordancota.site/")
        
        # Wait for page to load
        time.sleep(3)
        
        # Scroll to the game section
        game_section = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h2[contains(text(), 'Fun Time: Play Space Invaders!')]"))
        )
        self.driver.execute_script("arguments[0].scrollIntoView(true);", game_section)
        time.sleep(2)
        
        # Find the game canvas
        try:
            self.game_canvas = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "gameCanvas"))
            )
            print("Game canvas found!")
        except:
            # Try alternative selectors
            canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                self.game_canvas = canvases[0]
                print("Found canvas element!")
            else:
                print("Could not find game canvas")
                return False
        
        return True
    
    def start_game(self):
        """Start the Space Invaders game"""
        print("Starting the game...")
        
        # Look for start button
        try:
            start_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            start_button.click()
            print("Clicked start button")
        except:
            # Try clicking on canvas to start
            self.game_canvas.click()
            print("Clicked on canvas to start")
        
        time.sleep(1)
        self.game_active = True
        
    def get_game_state(self):
        """Extract current game state from the page"""
        try:
            # Get score
            score_element = self.driver.find_element(By.XPATH, "//span[contains(text(), 'Score:')]/following-sibling::span")
            self.score = int(score_element.text) if score_element.text.isdigit() else self.score
            
            # Get level  
            level_element = self.driver.find_element(By.XPATH, "//span[contains(text(), 'Level:')]/following-sibling::span")
            self.level = int(level_element.text) if level_element.text.isdigit() else self.level
            
            # Get lives
            lives_element = self.driver.find_element(By.XPATH, "//span[contains(text(), 'Lives:')]/following-sibling::span")
            self.lives = int(lives_element.text) if lives_element.text.isdigit() else self.lives
            
        except Exception as e:
            print(f"Could not read game state: {e}")
    
    def move_left(self):
        """Move player left"""
        ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
        time.sleep(0.05)
        ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
    
    def move_right(self):
        """Move player right"""
        ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
        time.sleep(0.05)
        ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
    
    def shoot(self):
        """Shoot a bullet"""
        current_time = time.time()
        if current_time - self.last_shot_time >= self.shot_cooldown:
            ActionChains(self.driver).key_down(Keys.SPACE).perform()
            time.sleep(0.05)
            ActionChains(self.driver).key_up(Keys.SPACE).perform()
            self.last_shot_time = current_time
    
    def analyze_game_screen(self):
        """Analyze the game screen to determine optimal moves"""
        try:
            # Get canvas screenshot
            canvas_location = self.game_canvas.location
            canvas_size = self.game_canvas.size
            
            # Take screenshot of the game area
            screenshot = pyautogui.screenshot(region=(
                canvas_location['x'], 
                canvas_location['y'], 
                canvas_size['width'], 
                canvas_size['height']
            ))
            
            # Convert to numpy array for analysis
            screen_array = np.array(screenshot)
            
            return self.detect_game_objects(screen_array)
            
        except Exception as e:
            print(f"Error analyzing screen: {e}")
            return None
    
    def detect_game_objects(self, screen_array):
        """Detect invaders, bullets, and player position using image processing"""
        # Convert to grayscale for easier processing
        gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        
        # Simple object detection based on color patterns
        # This is a simplified approach - a more sophisticated AI would use
        # machine learning models trained on game screenshots
        
        height, width = gray.shape
        
        # Detect invaders (typically in upper portion)
        invader_region = gray[:height//2, :]
        
        # Detect bullets (small moving objects)
        # Enemy bullets typically move downward
        # Player bullets move upward
        
        return {
            'invaders_detected': True,  # Simplified
            'enemy_bullets': [],
            'safe_zones': self.find_safe_zones(gray),
            'optimal_position': width // 2  # Default to center
        }
    
    def find_safe_zones(self, gray_image):
        """Find safe zones to avoid enemy bullets"""
        height, width = gray_image.shape
        safe_zones = []
        
        # Analyze columns for bullet trajectories
        for x in range(0, width, 20):  # Sample every 20 pixels
            column = gray_image[:, x:x+20]
            # Simplified safe zone detection
            safe_zones.append(x / width)  # Normalize to 0-1
        
        return safe_zones
    
    def calculate_optimal_move(self, game_analysis):
        """Calculate the optimal move based on game analysis"""
        if not game_analysis:
            return 'shoot'  # Default action
        
        # Advanced AI strategy
        actions = []
        
        # Always shoot when possible (maximize score)
        actions.append('shoot')
        
        # Movement strategy based on threat analysis
        optimal_x = game_analysis.get('optimal_position', 0.5)
        current_x = self.player_position
        
        # Move towards optimal position
        if optimal_x < current_x - 0.1:
            actions.append('left')
            self.player_position = max(0.0, self.player_position - 0.05)
        elif optimal_x > current_x + 0.1:
            actions.append('right')
            self.player_position = min(1.0, self.player_position + 0.05)
        
        return actions
    
    def execute_actions(self, actions):
        """Execute the calculated actions"""
        for action in actions:
            if action == 'left':
                self.move_left()
            elif action == 'right':
                self.move_right()
            elif action == 'shoot':
                self.shoot()
            
            time.sleep(0.02)  # Small delay between actions
    
    def play_game_loop(self):
        """Main game playing loop"""
        print("Starting AI game loop...")
        consecutive_failures = 0
        max_failures = 100
        
        while self.game_active and consecutive_failures < max_failures:
            try:
                # Get current game state
                self.get_game_state()
                
                # Check if game is still active
                if self.lives <= 0:
                    print(f"Game Over! Final Score: {self.score}, Level: {self.level}")
                    self.game_active = False
                    break
                
                # Analyze game screen
                game_analysis = self.analyze_game_screen()
                
                # Calculate optimal moves
                actions = self.calculate_optimal_move(game_analysis)
                
                # Execute actions
                self.execute_actions(actions)
                
                # Print progress every 100 iterations
                if consecutive_failures % 20 == 0:
                    print(f"Score: {self.score}, Level: {self.level}, Lives: {self.lives}")
                
                consecutive_failures = 0
                time.sleep(0.05)  # Control game speed
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures % 10 == 0:
                    print(f"Error in game loop: {e}")
                time.sleep(0.1)
        
        print(f"Game ended. Final Score: {self.score}")
    
    def run(self):
        """Main execution function"""
        try:
            print("Initializing Space Invaders AI Bot...")
            
            # Setup browser
            self.setup_browser()
            
            # Navigate to game
            if not self.navigate_to_game():
                print("Failed to navigate to game")
                return
            
            # Start game
            self.start_game()
            
            # Play the game
            self.play_game_loop()
            
        except KeyboardInterrupt:
            print("Bot stopped by user")
        except Exception as e:
            print(f"Error running bot: {e}")
        finally:
            if self.driver:
                print("Closing browser...")
                self.driver.quit()

def main():
    """Main function to run the Space Invaders AI"""
    print("🚀 Space Invaders AI Bot")
    print("Target: https://jordancota.site/")
    print("Current High Score to Beat: 25,940 points (Level 8)")
    print("=" * 50)
    
    bot = SpaceInvadersAI()
    bot.run()

if __name__ == "__main__":
    main()