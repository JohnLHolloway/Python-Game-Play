#!/usr/bin/env python3
"""
🚀 FIXED JORDANCOTA SPACE INVADERS AI
🎯 MISSION: Proper keyboard input handling for https://jordancota.site/
🏆 Goal: Beat 25,940 points with WORKING controls
"""

import tensorflow as tf
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import time
import threading
import random
from collections import deque
import json
import os

class FixedJordanCotaSpaceInvadersAI:
    def __init__(self):
        print("🎮 FIXED JORDANCOTA SPACE INVADERS AI")
        print("🎯 Mission: Working controls for https://jordancota.site/")
        print("🏆 Target: 25,940+ points")
        
        # Game configuration - CORRECT URL
        self.game_url = "https://jordancota.site/"
        self.target_score = 25940
        self.best_total_score = 0
        self.session_count = 0
        
        # Neural network configuration
        self.state_size = 15  # Streamlined for better performance
        self.action_size = 5  # Simplified actions: left, right, shoot, left+shoot, right+shoot
        self.memory = deque(maxlen=100000)  # Experience replay buffer
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Build neural network
        self.model = self._build_model()
        
        # Selenium setup
        self.driver = None
        self.game_canvas = None
        self.action_chains = None
        
        # Monitoring
        self.running = True
        self.current_score = 0
        self.current_lives = 3
        self.game_started = False
        
    def _build_model(self):
        """Build optimized neural network with working input handling"""
        print("🧠 Building FIXED neural network...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256, activation='relu'),  
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        param_count = model.count_params()
        print(f"✅ FIXED Neural Network: {param_count:,} parameters")
        print("🎯 Optimized for: Working keyboard input")
        
        return model
    
    def setup_browser(self):
        """Setup Chrome browser with proper focus handling"""
        print("🚀 Setting up FIXED browser for jordancota.site...")
        
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--window-size=1400,900")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            self.action_chains = ActionChains(self.driver)
            print("✅ FIXED Browser ready with ActionChains")
            return True
        except Exception as e:
            print(f"❌ Browser setup failed: {e}")
            return False
    
    def navigate_to_jordancota_game(self):
        """Navigate to jordancota.site and ensure proper focus"""
        print("🎮 Navigating to jordancota.site...")
        
        try:
            # Navigate to the correct URL
            self.driver.get(self.game_url)
            print(f"✅ Successfully loaded {self.game_url}")
            
            # Wait for page to fully load
            time.sleep(5)
            
            # Look for the game canvas
            try:
                canvas = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "canvas"))
                )
                self.game_canvas = canvas
                print("✅ Game canvas found!")
                
                # Ensure canvas is clickable and focused
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable(canvas)
                )
                
                # Click canvas to focus it
                canvas.click()
                time.sleep(1)
                print("✅ Canvas focused and ready")
                
                return True
                
            except Exception as e:
                print(f"⚠️ Canvas not found: {e}")
                print("🔍 Looking for alternative game elements...")
                
                # Try to find game by looking for common patterns
                game_found = False
                possible_selectors = [
                    "canvas", "#game", ".game", "#gameCanvas", 
                    ".gameCanvas", "#game-canvas", ".game-canvas"
                ]
                
                for selector in possible_selectors:
                    try:
                        if selector.startswith("#") or selector.startswith("."):
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        else:
                            element = self.driver.find_element(By.TAG_NAME, selector)
                        
                        element.click()
                        self.game_canvas = element
                        print(f"✅ Found game element: {selector}")
                        game_found = True
                        break
                    except:
                        continue
                
                return game_found
                
        except Exception as e:
            print(f"❌ Failed to load jordancota.site: {e}")
            return False
    
    def start_jordancota_game(self):
        """Start the game with PROPER input focus"""
        print("▶️ Starting game with FIXED controls...")
        
        try:
            # Multiple methods to ensure game focus and start
            
            # Method 1: Focus on canvas first
            if self.game_canvas:
                self.game_canvas.click()
                time.sleep(0.5)
                print("✅ Canvas clicked for focus")
            
            # Method 2: Focus on body as fallback  
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.click()
            time.sleep(0.5)
            
            # Method 3: Use ActionChains for proper key dispatch
            print("🎹 Sending start commands with ActionChains...")
            
            # Send start keys using ActionChains (more reliable)
            start_keys = [Keys.SPACE, Keys.ENTER, "s", "1"]
            for key in start_keys:
                try:
                    if self.game_canvas:
                        self.action_chains.click(self.game_canvas).send_keys(key).perform()
                    else:
                        self.action_chains.send_keys(key).perform()
                    time.sleep(0.3)
                    print(f"✅ ActionChains sent: {key}")
                except Exception as e:
                    print(f"⚠️ ActionChains failed for {key}: {e}")
            
            # Method 4: Direct JavaScript execution for more reliable input
            print("🔧 Using JavaScript for direct game interaction...")
            try:
                # Dispatch keyboard events directly to the canvas
                js_script = """
                var canvas = document.querySelector('canvas');
                if (canvas) {
                    canvas.focus();
                    
                    // Dispatch space key event
                    var spaceEvent = new KeyboardEvent('keydown', {
                        key: ' ',
                        keyCode: 32,
                        which: 32,
                        bubbles: true
                    });
                    canvas.dispatchEvent(spaceEvent);
                    
                    // Dispatch enter key event
                    var enterEvent = new KeyboardEvent('keydown', {
                        key: 'Enter',
                        keyCode: 13,
                        which: 13,
                        bubbles: true
                    });
                    canvas.dispatchEvent(enterEvent);
                    
                    console.log('Game start events dispatched');
                    return true;
                } else {
                    console.log('Canvas not found');
                    return false;
                }
                """
                result = self.driver.execute_script(js_script)
                if result:
                    print("✅ JavaScript game start events dispatched")
                else:
                    print("⚠️ JavaScript canvas not found")
            except Exception as e:
                print(f"⚠️ JavaScript execution failed: {e}")
            
            # Method 5: Look for and click start buttons
            start_button_texts = ["START", "PLAY", "BEGIN", "Start Game", "New Game"]
            for button_text in start_button_texts:
                try:
                    button = self.driver.find_element(By.XPATH, f"//*[contains(text(), '{button_text}')]")
                    self.action_chains.click(button).perform()
                    time.sleep(0.5)
                    print(f"✅ Clicked {button_text} button")
                    break
                except:
                    continue
            
            self.game_started = True
            print("✅ FIXED game start sequence completed")
            return True
            
        except Exception as e:
            print(f"⚠️ Game start issue: {e}")
            self.game_started = True  # Proceed anyway
            return True
    
    def perform_jordancota_action_FIXED(self, action):
        """FIXED action performance with proper input dispatch"""
        try:
            # Ensure game focus before each action
            if self.game_canvas:
                target = self.game_canvas
            else:
                target = self.driver.find_element(By.TAG_NAME, "body")
            
            # Use ActionChains for reliable input dispatch
            if action == 0:  # Move left
                self.action_chains.click(target).send_keys(Keys.ARROW_LEFT).perform()
                print("⬅️ LEFT", end="", flush=True)
            elif action == 1:  # Move right  
                self.action_chains.click(target).send_keys(Keys.ARROW_RIGHT).perform()
                print("➡️ RIGHT", end="", flush=True)
            elif action == 2:  # Shoot
                self.action_chains.click(target).send_keys(Keys.SPACE).perform()
                print("🔫 SHOOT", end="", flush=True)
            elif action == 3:  # Move left and shoot
                self.action_chains.click(target).send_keys(Keys.ARROW_LEFT).pause(0.05).send_keys(Keys.SPACE).perform()
                print("⬅️🔫 LEFT+SHOOT", end="", flush=True)
            elif action == 4:  # Move right and shoot
                self.action_chains.click(target).send_keys(Keys.ARROW_RIGHT).pause(0.05).send_keys(Keys.SPACE).perform()
                print("➡️🔫 RIGHT+SHOOT", end="", flush=True)
            
            # Also try JavaScript dispatch as backup
            if action <= 4:
                try:
                    js_action_map = {
                        0: "ArrowLeft",
                        1: "ArrowRight", 
                        2: " ",
                        3: ["ArrowLeft", " "],
                        4: ["ArrowRight", " "]
                    }
                    
                    keys_to_send = js_action_map[action]
                    if not isinstance(keys_to_send, list):
                        keys_to_send = [keys_to_send]
                    
                    for key in keys_to_send:
                        keycode = 32 if key == " " else (37 if key == "ArrowLeft" else 39)
                        js_script = f"""
                        var canvas = document.querySelector('canvas') || document.body;
                        var event = new KeyboardEvent('keydown', {{
                            key: '{key}',
                            keyCode: {keycode},
                            which: {keycode},
                            bubbles: true
                        }});
                        canvas.dispatchEvent(event);
                        """
                        self.driver.execute_script(js_script)
                        if len(keys_to_send) > 1:
                            time.sleep(0.02)
                            
                except Exception as e:
                    pass  # Continue if JS fails
            
        except Exception as e:
            print(f"\n⚠️ Action failed: {e}")
    
    def get_jordancota_game_state(self):
        """Enhanced game state extraction"""
        try:
            # Try to extract score and lives from the page
            score = 0
            lives = 3
            
            # Enhanced score detection
            try:
                # JavaScript-based score extraction
                js_score_script = """
                var scoreElements = document.querySelectorAll('*');
                var score = 0;
                for (var i = 0; i < scoreElements.length; i++) {
                    var text = scoreElements[i].textContent || scoreElements[i].innerText || '';
                    var matches = text.match(/(?:score|points?)\\s*:?\\s*(\\d+)/i);
                    if (matches) {
                        score = Math.max(score, parseInt(matches[1]));
                    }
                }
                return score;
                """
                js_score = self.driver.execute_script(js_score_script)
                if js_score and js_score > 0:
                    score = js_score
                    
                # Also try text-based extraction
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                import re
                score_patterns = [
                    r'Score[:\s]*(\d+)',
                    r'SCORE[:\s]*(\d+)', 
                    r'Points[:\s]*(\d+)',
                    r'(\d+)\s*points?',
                    r'(\d{2,})'  # Any number with 2+ digits
                ]
                
                for pattern in score_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        potential_score = max([int(x) for x in matches])
                        if potential_score > score and potential_score < 1000000:  # Reasonable score range
                            score = potential_score
                            break
                        
            except Exception as e:
                pass
            
            # Similar enhanced detection for lives
            try:
                js_lives_script = """
                var lifeElements = document.querySelectorAll('*');
                var lives = 3;
                for (var i = 0; i < lifeElements.length; i++) {
                    var text = lifeElements[i].textContent || lifeElements[i].innerText || '';
                    var matches = text.match(/(?:lives?|life)\\s*:?\\s*(\\d+)/i);
                    if (matches) {
                        lives = parseInt(matches[1]);
                        break;
                    }
                }
                return lives;
                """
                js_lives = self.driver.execute_script(js_lives_script)
                if js_lives is not None:
                    lives = js_lives
                    
            except Exception as e:
                pass
            
            # Create simplified but effective state vector
            state = np.array([
                min(score / 1000.0, 50.0),  # Normalized score
                lives / 3.0,  # Normalized lives
                self.session_count / 50.0,  # Session progress  
                time.time() % 30 / 30.0,  # Time cycle
                self.epsilon,  # Current exploration rate
                (score / max(self.target_score, 1)) * 100,  # Progress toward target
                1.0 if score > 0 else 0.0,  # Scoring indicator
                1.0 if lives == 3 else 0.0,  # Full lives
                1.0 if lives == 2 else 0.0,  # Two lives  
                1.0 if lives == 1 else 0.0,  # One life
                1.0 if score > self.best_total_score else 0.0,  # New high score
                score / max(self.best_total_score, 1),  # Relative performance
                random.uniform(-0.1, 0.1),  # Small exploration noise
                1.0 if self.game_started else 0.0,  # Game state
                min(len(self.memory) / 10000.0, 1.0)  # Memory fullness
            ])
            
            self.current_score = score
            self.current_lives = lives
            
            return state
        
        except Exception as e:
            # Return default state if extraction fails
            return np.array([0.0] * self.state_size)
    
    def calculate_reward(self, old_state, new_state, action):
        """Reward function optimized for working controls"""
        old_score = old_state[0] * 1000
        new_score = new_state[0] * 1000  
        old_lives = old_state[1] * 3
        new_lives = new_state[1] * 3
        
        reward = 0
        
        # Strong score increase reward
        score_diff = new_score - old_score
        if score_diff > 0:
            reward += score_diff * 50  # High reward for any scoring
            print(f" +{score_diff} pts! ", end="", flush=True)
        
        # Life loss penalty
        if new_lives < old_lives:
            reward -= 2000  # Penalty but not too harsh
            print(" LIFE LOST! ", end="", flush=True)
        
        # Survival bonus
        reward += 5  # Reward for staying alive
        
        # Action bonuses for aggressive play
        if action == 2 or action == 3 or action == 4:  # Shooting actions
            reward += 10  # Bonus for shooting
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action with epsilon-greedy strategy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the neural network"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        target_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def monitor_progress(self):
        """Monitor with action feedback"""
        while self.running:
            try:
                progress = min((self.current_score / self.target_score) * 100, 100)
                strategy = "🎯 LEARNING" if self.epsilon > 0.5 else "🎮 PLAYING" if self.epsilon > 0.1 else "🏆 EXPERT"
                
                print(f"\r🎮 LIVE: {self.current_score} pts | {self.current_lives} lives | {progress:.1f}% | {strategy} | S{self.session_count} ", end="", flush=True)
                
                if self.current_score >= self.target_score:
                    print(f"\n🏆 CHAMPIONSHIP ACHIEVED! Score: {self.current_score}")
                    print("🎊 BEAT THE LEADERBOARD!")
                    
                    name = input("🏆 Enter your champion name: ").strip()
                    if name:
                        print(f"🏆 CHAMPION: {name} - Score: {self.current_score}")
                        achievement = {
                            'name': name,
                            'score': self.current_score,
                            'timestamp': time.time(),
                            'sessions': self.session_count,
                            'site': 'jordancota.site'
                        }
                        with open('jordancota_champion.json', 'w') as f:
                            json.dump(achievement, f)
                
                time.sleep(0.5)  # Faster updates to see action feedback
                
            except Exception as e:
                time.sleep(1)
    
    def play_session(self):
        """Play session with working controls"""
        print(f"\n🎮 FIXED SESSION {self.session_count + 1}")
        print(f"🏆 Best Score: {self.best_total_score}")
        print(f"🎯 Target: {self.target_score} points")
        
        session_start_time = time.time()
        steps = 0
        session_total_score = 0
        
        # Get initial state
        state = self.get_jordancota_game_state()
        
        # Play with working controls
        while self.running and steps < 1000:  # Shorter sessions for faster feedback
            try:
                # Choose action
                action = self.act(state)
                
                # Perform action with FIXED controls
                self.perform_jordancota_action_FIXED(action)
                
                # Appropriate delay
                time.sleep(0.1)
                
                # Get new state
                next_state = self.get_jordancota_game_state()
                
                # Calculate reward
                reward = self.calculate_reward(state, next_state, action)
                
                # Check if session should end
                done = (self.current_lives <= 0 or 
                       time.time() - session_start_time > 120)  # 2 minute sessions
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                steps += 1
                session_total_score = max(session_total_score, self.current_score)
                
                # Train network
                if len(self.memory) > 32 and steps % 5 == 0:
                    self.replay()
                
                # End session if done
                if done:
                    break
                    
            except Exception as e:
                print(f"\n⚠️ Step error: {e}")
                time.sleep(0.5)
                continue
        
        # Update best score
        if session_total_score > self.best_total_score:
            self.best_total_score = session_total_score
            print(f"\n🏆 NEW BEST SCORE: {self.best_total_score}")
        
        print(f"\n📊 Session Complete: {session_total_score} points in {steps} steps")
        return session_total_score
    
    def run_infinite_training(self):
        """Run with FIXED controls"""
        print("\n" + "="*80)
        print("🎮 FIXED JORDANCOTA SPACE INVADERS AI")
        print("🎯 MISSION: Working keyboard controls for https://jordancota.site/")
        print("🏆 Strategy: ActionChains + JavaScript input dispatch")
        print("🧠 AI: Neural networks with proper game interaction")
        print("🎊 Goal: Beat 25,940 points with WORKING controls")
        print("="*80)
        
        # Setup
        if not self.setup_browser():
            print("❌ Failed to setup browser")
            return
        
        if not self.navigate_to_jordancota_game():
            print("❌ Failed to navigate to game")
            return
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self.monitor_progress)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            while self.running:
                self.session_count += 1
                
                # Start game with FIXED controls
                self.start_jordancota_game()
                
                # Play session
                score = self.play_session()
                
                # Check for championship
                if score >= self.target_score:
                    print(f"\n🏆 CHAMPIONSHIP ACHIEVED!")
                    break
                
                # Brief pause between sessions
                time.sleep(2)
                
                # Refresh for next session
                try:
                    self.driver.refresh()
                    time.sleep(3)
                    # Re-establish canvas focus
                    if self.navigate_to_jordancota_game():
                        print("🔄 Game refreshed and focused")
                except Exception as e:
                    print(f"⚠️ Refresh failed: {e}")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Training error: {e}")
        finally:
            self.running = False
            if self.driver:
                self.driver.quit()
            
            print(f"\n🏆 TRAINING COMPLETE")
            print(f"📊 Sessions: {self.session_count}")
            print(f"🎯 Best Score: {self.best_total_score}")
            
            if self.best_total_score >= self.target_score:
                print("🏆 CHAMPIONSHIP STATUS: ACHIEVED! 🎊")
            else:
                print(f"📈 Progress: {(self.best_total_score/self.target_score)*100:.1f}%")

def main():
    """Main execution with FIXED controls"""
    print("🚀 Initializing FIXED JORDANCOTA AI...")
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Create and run FIXED AI
    ai = FixedJordanCotaSpaceInvadersAI()
    ai.run_infinite_training()

if __name__ == "__main__":
    main()