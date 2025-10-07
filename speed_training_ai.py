"""
SPEED TRAINING AI - Ultra-Fast Space Invaders Training
Optimized for maximum training speed with minimal delays
Based on Ultimate Training AI but with speed optimizations
"""

import time
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import mss
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class SpeedTrainingAI:
    def __init__(self):
        self.driver = None
        self.game_canvas = None
        self.score = 0
        self.best_score = 0
        self.level = 1
        self.lives = 3
        self.current_high_score = 25940
        self.leaderboard_beaten = False
        self.name_entered = False
        
        # Performance tracking
        self.game_sessions = []
        self.score_history = []
        self.strategy_performance = {
            'aggressive': [],
            'defensive': [],
            'balanced': [],
            'rapid_fire': []
        }
        
        # Strategy management
        self.current_strategy = 'balanced'
        self.episodes_with_strategy = 0
        self.strategy_switch_threshold = 3  # Reduced from 5 for faster adaptation
        
        # Neural networks
        self.decision_network = None
        self.cnn_network = None
        self.action_history = deque(maxlen=50)  # Reduced from 100
        self.score_predictor = None
        
        # Speed optimizations
        self.ultra_fast_mode = True
        self.minimal_delays = True
        self.reduced_checks = True

    def setup_ultra_fast_browser(self):
        """Setup browser with maximum speed optimizations"""
        print("🚀 Ultra-speed browser initialization...")
        
        options = Options()
        # Speed optimizations
        options.add_argument('--headless=new')  # Run headless for maximum speed
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=VizDisplayCompositor')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')  # Disable image loading for speed
        options.add_argument('--disable-javascript-harmony-shipping')
        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-ipc-flooding-protection')
        options.add_argument('--max_old_space_size=4096')
        
        # Ultra-fast page loading
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2
        }
        options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        # Minimal wait times for speed
        self.driver.implicitly_wait(5)  # Reduced from 30
        self.driver.set_page_load_timeout(15)  # Reduced from 30
        
        # Navigate to game
        self.driver.get("https://jordancota.site/")
        
        # Minimal load wait
        if not self.ultra_fast_mode:
            time.sleep(2)  # Reduced from 5
        
        self.game_canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        
        # Minimal setup wait
        if not self.ultra_fast_mode:
            time.sleep(1)  # Reduced from 3
        
        print("⚡ Ultra-fast browser ready!")

    def ultra_fast_game_activation(self):
        """Speed-optimized game activation with minimal delays"""
        methods_used = []
        
        try:
            # Method 1: Start button (fastest)
            start_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            if start_button.is_displayed():
                start_button.click()
                methods_used.append("StartButton")
                if not self.minimal_delays:
                    time.sleep(0.1)  # Minimal delay
        except:
            pass
            
        try:
            # Method 2: Canvas click (fast)
            self.game_canvas.click()
            methods_used.append("Canvas")
            if not self.minimal_delays:
                time.sleep(0.1)
        except:
            pass
            
        try:
            # Method 3: Key activation (fastest)
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.SPACE).perform()
            methods_used.append("Keys")
            if not self.minimal_delays:
                time.sleep(0.05)
        except:
            pass
        
        return methods_used

    def speed_optimized_get_game_state(self):
        """Ultra-fast game state detection with minimal processing"""
        try:
            # Speed-optimized score extraction
            score_text = self.driver.execute_script("""
                return document.body.innerText.match(/Score: (\\d+)/);
            """)
            
            if score_text:
                self.score = int(score_text[1])
            
            # Fast lives detection (reduced frequency)
            if hasattr(self, 'step_count') and self.step_count % 100 == 0:  # Check every 100 steps
                lives_text = self.driver.execute_script("""
                    return document.body.innerText.match(/Lives: (\\d+)/);
                """)
                if lives_text:
                    self.lives = int(lives_text[1])
            
            return True
        except:
            return False

    def ultra_fast_strategy_execution(self, strategy):
        """Speed-optimized strategy execution with minimal delays"""
        actions = []
        
        if strategy == 'rapid_fire':
            actions = [
                ('space', 0),      # No delay for ultra-speed
                ('left', 0),
                ('space', 0),
                ('right', 0),
                ('space', 0)
            ]
        elif strategy == 'aggressive':
            actions = [
                ('space', 0),
                ('left', 0.001),   # Minimal delay
                ('space', 0),
                ('right', 0.001),
                ('space', 0)
            ]
        elif strategy == 'defensive':
            actions = [
                ('left', 0.002),
                ('right', 0.002),
                ('space', 0.001)
            ]
        else:  # balanced
            actions = [
                ('space', 0.001),
                ('left', 0.001),
                ('space', 0.001),
                ('right', 0.001)
            ]
        
        # Execute actions with minimal delays
        try:
            for action, delay in actions:
                if action == 'space':
                    ActionChains(self.driver).send_keys(Keys.SPACE).perform()
                elif action == 'left':
                    ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                elif action == 'right':
                    ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
                
                if delay > 0 and not self.minimal_delays:
                    time.sleep(delay)
                    
        except Exception as e:
            pass

    def speed_optimized_game_over_check(self):
        """Ultra-fast game over detection"""
        try:
            # Simplified game over detection
            page_text = self.driver.execute_script("return document.body.innerText.toLowerCase();")
            
            # Quick checks
            game_over_indicators = ['game over', 'restart', 'play again']
            return any(indicator in page_text for indicator in game_over_indicators)
            
        except:
            return False

    def speed_training_session(self, max_episodes=50):  # Increased episodes for speed training
        """Ultra-fast training session with speed optimizations"""
        print("⚡🎯 SPEED TRAINING SESSION INITIATED!")
        print("🚀 Ultra-Fast Mode: Enabled")
        print("⏱️ Minimal Delays: Enabled") 
        print("🎯 Reduced Checks: Enabled")
        print(f"📊 Max Episodes: {max_episodes}")
        
        for episode in range(max_episodes):
            print(f"\n🚀 Episode {episode + 1}/{max_episodes}")
            
            # Speed-optimized episode start
            episode_start_score = self.score
            step_count = 0
            max_steps = 300  # Reduced from 15000 for faster episodes
            
            # Strategy selection (faster adaptation)
            if episode % self.strategy_switch_threshold == 0:
                strategies = ['balanced', 'aggressive', 'rapid_fire', 'defensive']
                strategy = random.choice(strategies)
                if episode > 0:
                    print(f"🔄 Strategy switched to: {strategy}")
            else:
                strategy = self.current_strategy
            
            print(f"🧠 Using strategy: {strategy}")
            
            # Ultra-fast episode execution
            while step_count < max_steps:
                step_count += 1
                
                # Speed-optimized game state
                if step_count % 20 == 0:  # Reduced frequency
                    self.speed_optimized_get_game_state()
                
                # Ultra-fast strategy execution
                self.ultra_fast_strategy_execution(strategy)
                
                # Reduced game over checks for speed
                if step_count % 100 == 0 and self.reduced_checks:  # Check every 100 steps
                    if self.speed_optimized_game_over_check():
                        print(f"⚡ Game over detected at step {step_count}")
                        break
                
                # Success check
                if self.score > self.current_high_score:
                    print("🎉 SPEED TRAINING SUCCESS!")
                    break
                
                # Ultra-minimal delays for maximum speed
                if strategy == 'rapid_fire':
                    # No delay for maximum speed
                    pass
                elif strategy == 'aggressive':
                    time.sleep(0.001)  # 1ms delay
                else:
                    time.sleep(0.002)  # 2ms delay
            
            # Episode results
            episode_score = self.score - episode_start_score
            self.score_history.append(self.score)
            self.strategy_performance[strategy].append(episode_score)
            
            # Update best score
            if self.score > self.best_score:
                self.best_score = self.score
                print(f"🎯 New Speed Record: {self.best_score}")
            
            print(f"⚡ Episode {episode + 1} Complete:")
            print(f"   📊 Score: {episode_score} (Total: {self.score})")
            print(f"   🏆 Best: {self.best_score}")
            print(f"   📈 Progress: {(self.score / self.current_high_score) * 100:.1f}%")
            print(f"   🎯 Strategy: {strategy}")
            
            # Success check
            if self.score > self.current_high_score:
                print("🎉 SPEED TRAINING VICTORY!")
                break
            
            # Fast restart
            if episode < max_episodes - 1:
                print("🔄 Ultra-fast restart...")
                self.ultra_fast_game_activation()
        
        # Session complete
        print(f"\n⚡🎯 SPEED TRAINING SESSION COMPLETE!")
        print(f"🏆 Speed Training Best: {self.best_score}")
        print(f"🎯 Target: {self.current_high_score}")
        print(f"📈 Speed Success Rate: {(self.best_score / self.current_high_score) * 100:.1f}%")
        
        # Strategy performance analysis
        print(f"\n📊 Speed Strategy Performance:")
        for strategy, scores in self.strategy_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   {strategy}: {avg_score:.1f} avg ({len(scores)} episodes)")

def main():
    """Main speed training execution"""
    print("⚡🎯=============================================================================⚡🎯")
    print("🚀 SPEED TRAINING AI - ULTRA-FAST SPACE INVADERS TRAINING")
    print("⚡ Maximum Speed Optimizations + Minimal Delays")
    print("🎯 Mission: Achieve high scores with ultra-fast training")
    print("🏆 Features: Speed Mode, Reduced Delays, Fast Activation")
    print("================================================================================⚡🎯")
    
    # Initialize speed training AI
    ai = SpeedTrainingAI()
    
    try:
        # Setup ultra-fast browser
        ai.setup_ultra_fast_browser()
        
        # Ultra-fast game activation
        print("⚡ Ultra-fast game activation...")
        methods = ai.ultra_fast_game_activation()
        print(f"✅ Game activated via: {', '.join(methods) if methods else 'Auto'}")
        
        # Speed training session
        ai.speed_training_session(max_episodes=100)  # More episodes with faster execution
        
    except KeyboardInterrupt:
        print("\n⚡ Speed training interrupted by user")
    except Exception as e:
        print(f"⚡ Speed training error: {e}")
    finally:
        if ai.driver:
            ai.driver.quit()
        print("⚡ Speed training AI session ended")

if __name__ == "__main__":
    main()