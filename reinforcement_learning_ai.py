"""
Advanced Reinforcement Learning Space Invaders AI
Deep Q-Network (DQN) with Experience Replay and Target Networks
Advanced training algorithms for maximum high score achievement
"""

import time
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import mss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network with advanced architecture"""
    def __init__(self, state_size=50, action_size=4, hidden_size=512):
        super(DQNNetwork, self).__init__()
        
        # Advanced architecture with residual connections
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        # Residual connection
        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x + residual))  # Residual connection
        
        x = F.relu(self.bn3(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for better learning"""
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities.append(priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority

class ReinforcementLearningAI:
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
        
        # RL Components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Using device: {self.device}")
        
        self.state_size = 50
        self.action_size = 4  # [shoot, left, right, stay]
        
        # DQN Networks
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)
        
        # Training parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.batch_size = 64
        self.update_frequency = 4
        self.target_update_frequency = 1000
        
        # Training metrics
        self.training_scores = []
        self.training_losses = []
        self.episode_rewards = []
        self.training_step = 0
        
        # Screen capture
        self.sct = mss.mss()
        self.game_region = None
        self.state_scaler = StandardScaler()
        
        # Initialize target network
        self.update_target_network()
        
        print("🧠 Reinforcement Learning AI initialized with DQN")
    
    def setup_advanced_browser(self):
        """Setup browser with optimal gaming settings"""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1200")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--force-gpu-mem-available-mb=4096")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(30)
        
        print("🌐 Advanced RL browser initialized")
    
    def navigate_and_analyze(self):
        """Navigate to game and set up environment"""
        print("🎮 RL AI - Loading and analyzing game environment...")
        self.driver.get("https://jordancota.site/")
        time.sleep(5)
        
        # Get leaderboard target
        self.get_current_leaderboard_high_score()
        
        # Navigate to game
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
        time.sleep(3)
        
        # Find game canvas
        try:
            self.game_canvas = self.driver.find_element(By.ID, "gameCanvas")
            location = self.game_canvas.location
            size = self.game_canvas.size
            self.game_region = {
                'top': location['y'],
                'left': location['x'],
                'width': size['width'],
                'height': size['height']
            }
            print("✅ Game environment analyzed")
        except:
            canvases = self.driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                self.game_canvas = canvases[0]
                location = self.game_canvas.location
                size = self.game_canvas.size
                self.game_region = {
                    'top': location['y'],
                    'left': location['x'],
                    'width': size['width'],
                    'height': size['height']
                }
                print("✅ Canvas found and analyzed")
            else:
                return False
        
        self.start_rl_game()
        return True
    
    def get_current_leaderboard_high_score(self):
        """Extract current high score to beat"""
        try:
            page_source = self.driver.page_source
            patterns = [
                r'(\d{1,2},?\d{3,4})\s*\(Level\s*\d+\)',
                r'John\s*H.*?(\d{1,2},?\d{3,4})',
                r'(\d{1,2},?\d{3,4}).*?Level\s*8'
            ]
            
            highest_score = 0
            for pattern in patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    try:
                        score_str = match.replace(',', '')
                        score_val = int(score_str)
                        if 10000 <= score_val <= 100000:
                            highest_score = max(highest_score, score_val)
                    except:
                        continue
            
            if highest_score > 0:
                self.current_high_score = highest_score
            
            print(f"🎯 RL Target: {self.current_high_score} points to beat")
            return self.current_high_score
            
        except Exception as e:
            print(f"Leaderboard analysis error: {e}")
            return self.current_high_score
    
    def start_rl_game(self):
        """Start game with multiple activation methods"""
        print("🚀 Starting Reinforcement Learning session...")
        
        try:
            start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start')]")
            start_btn.click()
            print("✅ Game started via button")
        except:
            pass
        
        try:
            self.game_canvas.click()
            ActionChains(self.driver).move_to_element(self.game_canvas).click().perform()
            print("✅ Game activated via canvas")
        except:
            pass
        
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER).perform()
            time.sleep(0.5)
            actions.send_keys(Keys.SPACE).perform()
            print("✅ Game started via keys")
        except:
            pass
        
        time.sleep(3)
    
    def get_advanced_game_state(self):
        """Get comprehensive game state for RL training"""
        try:
            # Screen capture
            screen_img = None
            if self.game_region:
                screenshot = self.sct.grab(self.game_region)
                screen_img = np.array(screenshot)
                screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2RGB)
            
            # Basic game state
            page_source = self.driver.page_source
            
            # Extract score
            old_score = self.score
            score_patterns = [r'Score[:\s]*(\d+)', r'score[:\s]*(\d+)']
            for pattern in score_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    try:
                        new_score = int(matches[-1])
                        if new_score >= self.score:
                            self.score = new_score
                            if self.score > self.best_score:
                                self.best_score = self.score
                        break
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
            
            # Calculate reward
            reward = self.calculate_reward(old_score, self.score)
            
            # Create state vector
            state_vector = self.create_state_vector(screen_img)
            
            # Check if game over
            game_over = self.is_game_over()
            
            return {
                'state': state_vector,
                'reward': reward,
                'done': game_over,
                'score': self.score,
                'level': self.level,
                'lives': self.lives
            }
            
        except Exception as e:
            return {
                'state': np.zeros(self.state_size),
                'reward': 0,
                'done': False,
                'score': self.score,
                'level': self.level,
                'lives': self.lives
            }
    
    def create_state_vector(self, screen_img):
        """Create advanced state vector from game screen"""
        try:
            if screen_img is None:
                return np.zeros(self.state_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(screen_img, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            features = []
            
            # Image-based features
            # 1. Screen density (activity level)
            features.append(np.mean(gray) / 255.0)
            
            # 2. Edge density (movement/activity)
            edges = cv2.Canny(gray, 50, 150)
            features.append(np.sum(edges > 0) / (height * width))
            
            # 3. Zone analysis (divide screen into zones)
            zones = [
                gray[:height//3, :],  # Top
                gray[height//3:2*height//3, :],  # Middle
                gray[2*height//3:, :]  # Bottom
            ]
            
            for zone in zones:
                zone_activity = np.mean(zone) / 255.0
                zone_edges = cv2.Canny(zone, 50, 150)
                zone_edge_density = np.sum(zone_edges > 0) / (zone.shape[0] * zone.shape[1])
                features.extend([zone_activity, zone_edge_density])
            
            # 4. Horizontal analysis (left, center, right)
            left_zone = gray[:, :width//3]
            center_zone = gray[:, width//3:2*width//3]
            right_zone = gray[:, 2*width//3:]
            
            for zone in [left_zone, center_zone, right_zone]:
                features.append(np.mean(zone) / 255.0)
            
            # 5. Vertical gradients (detect falling objects)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features.append(np.mean(np.abs(grad_y)) / 255.0)
            
            # 6. Horizontal gradients (detect horizontal movement)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            features.append(np.mean(np.abs(grad_x)) / 255.0)
            
            # Game state features
            features.extend([
                self.score / max(self.current_high_score, 1),  # Normalized score
                self.level / 10.0,  # Normalized level
                self.lives / 3.0,  # Normalized lives
                self.epsilon,  # Current exploration rate
                len(self.replay_buffer.buffer) / self.replay_buffer.capacity  # Buffer fullness
            ])
            
            # Historical features (simple momentum)
            if len(self.training_scores) > 0:
                recent_scores = self.training_scores[-10:]
                features.extend([
                    np.mean(recent_scores) / max(self.current_high_score, 1),
                    np.std(recent_scores) / max(self.current_high_score, 1)
                ])
            else:
                features.extend([0.0, 0.0])
            
            # Pad or truncate to exact size
            while len(features) < self.state_size:
                features.append(0.0)
            features = features[:self.state_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"State vector creation error: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def calculate_reward(self, old_score, new_score):
        """Advanced reward function for RL training"""
        reward = 0.0
        
        # Score-based reward
        score_increase = new_score - old_score
        if score_increase > 0:
            reward += score_increase * 0.1  # Base reward for score
            
            # Bonus for larger score increases
            if score_increase >= 50:
                reward += 10  # Bonus for good performance
            elif score_increase >= 100:
                reward += 25  # Larger bonus
            elif score_increase >= 200:
                reward += 50  # Even larger bonus
        
        # Level progression reward
        if hasattr(self, 'prev_level') and self.level > self.prev_level:
            reward += 100  # Big reward for level advancement
        self.prev_level = self.level
        
        # Survival reward (staying alive)
        if self.lives > 0:
            reward += 0.1  # Small reward for staying alive
        
        # Life loss penalty
        if hasattr(self, 'prev_lives') and self.lives < self.prev_lives:
            reward -= 50  # Penalty for losing life
        self.prev_lives = self.lives
        
        # High score bonus
        if new_score > self.best_score:
            reward += 20  # Reward for personal best
        
        # Leaderboard approach bonus
        progress_ratio = new_score / self.current_high_score
        if progress_ratio > 0.1:
            reward += progress_ratio * 50  # Reward for approaching leaderboard
        
        return reward
    
    def select_action(self, state):
        """Epsilon-greedy action selection with DQN"""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def execute_action(self, action):
        """Execute the selected action"""
        try:
            if action == 0:  # Shoot
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
            elif action == 1:  # Move left
                ActionChains(self.driver).key_down(Keys.ARROW_LEFT).perform()
                time.sleep(0.05)
                ActionChains(self.driver).key_up(Keys.ARROW_LEFT).perform()
            elif action == 2:  # Move right
                ActionChains(self.driver).key_down(Keys.ARROW_RIGHT).perform()
                time.sleep(0.05)
                ActionChains(self.driver).key_up(Keys.ARROW_RIGHT).perform()
            # action == 3: Stay (do nothing)
            
            # Always try to shoot for maximum score
            if action != 0:
                ActionChains(self.driver).key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
                
        except Exception as e:
            pass
    
    def train_dqn(self):
        """Train the DQN using prioritized experience replay"""
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        batch_data = self.replay_buffer.sample(self.batch_size)
        if batch_data is None:
            return
            
        experiences, indices, weights = batch_data
        
        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Calculate TD errors for priority update
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Weighted loss
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss.squeeze() * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.squeeze())
        
        # Store loss for monitoring
        self.training_losses.append(weighted_loss.item())
        
        return weighted_loss.item()
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def is_game_over(self):
        """Check if game is over"""
        try:
            page_text = self.driver.page_source.lower()
            game_over_indicators = ['game over', 'try again', 'restart']
            
            indicator_count = sum(1 for indicator in game_over_indicators if indicator in page_text)
            
            # Game over if multiple indicators or lives are 0
            return indicator_count >= 2 or self.lives == 0
            
        except:
            return False
    
    def check_for_high_score_entry(self):
        """Check for high score entry opportunity"""
        if not self.leaderboard_beaten or self.name_entered:
            return False
        
        try:
            page_text = self.driver.page_source.lower()
            entry_patterns = ['enter.*?name', 'your.*?name', 'new.*?high.*?score', 'congratulations']
            
            for pattern in entry_patterns:
                if re.search(pattern, page_text):
                    return True
            
            # Check for input fields
            input_elements = self.driver.find_elements(By.TAG_NAME, "input")
            for element in input_elements:
                if element.is_displayed() and element.is_enabled():
                    return True
            
            return False
            
        except:
            return False
    
    def enter_champion_name(self):
        """Enter 'John H' as champion name"""
        if self.name_entered:
            return
        
        try:
            print("🏆 RL AI CHAMPION - ENTERING 'John H'!")
            
            # Find input field
            input_field = None
            selectors = ["input[type='text']", "input", "textarea"]
            
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
                print(f"✅ RL CHAMPION: {champion_name} saved!")
            else:
                ActionChains(self.driver).send_keys(champion_name).perform()
                time.sleep(1)
                ActionChains(self.driver).send_keys(Keys.ENTER).perform()
                print(f"✅ RL CHAMPION: {champion_name} entered!")
            
            self.name_entered = True
            
        except Exception as e:
            print(f"Champion name entry error: {e}")
    
    def train_rl_agent(self, episodes=100):
        """Main training loop for RL agent"""
        print("🧠 REINFORCEMENT LEARNING TRAINING INITIATED!")
        print(f"🎯 Target: Beat {self.current_high_score} points")
        print(f"📊 Training for {episodes} episodes")
        
        for episode in range(episodes):
            print(f"\n🧠 Episode {episode + 1}/{episodes}")
            
            # Reset game for new episode
            if episode > 0:
                self.restart_game()
            
            # Initialize episode
            current_state_info = self.get_advanced_game_state()
            current_state = current_state_info['state']
            episode_reward = 0
            episode_score = 0
            step_count = 0
            max_steps = 10000  # Prevent infinite episodes
            
            while not current_state_info['done'] and step_count < max_steps:
                step_count += 1
                
                # Select and execute action
                action = self.select_action(current_state)
                self.execute_action(action)
                
                # Get next state
                time.sleep(0.01)  # Small delay for state transition
                next_state_info = self.get_advanced_game_state()
                next_state = next_state_info['state']
                reward = next_state_info['reward']
                done = next_state_info['done']
                
                # Store experience
                experience = Experience(current_state, action, reward, next_state, done)
                # Calculate priority based on reward magnitude
                priority = abs(reward) + 1.0
                self.replay_buffer.add(experience, priority)
                
                # Train every few steps
                if self.training_step % self.update_frequency == 0:
                    loss = self.train_dqn()
                
                # Update target network
                if self.training_step % self.target_update_frequency == 0:
                    self.update_target_network()
                    print(f"🧠 Target network updated at step {self.training_step}")
                
                # Update state and metrics
                current_state = next_state
                current_state_info = next_state_info
                episode_reward += reward
                episode_score = next_state_info['score']
                self.training_step += 1
                
                # Check for high score entry
                if step_count % 100 == 0:
                    if self.score > self.current_high_score and not self.leaderboard_beaten:
                        print(f"🏆 RL AI BEATS LEADERBOARD! {self.score} > {self.current_high_score}!")
                        self.leaderboard_beaten = True
                    
                    if self.check_for_high_score_entry():
                        self.enter_champion_name()
                
                # Progress updates
                if step_count % 500 == 0:
                    progress = (self.score / self.current_high_score) * 100
                    print(f"📊 Step {step_count}: Score={self.score}, Reward={reward:.2f}, "
                          f"Progress={progress:.1f}%, ε={self.epsilon:.3f}")
            
            # Episode summary
            self.training_scores.append(episode_score)
            self.episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Episode results
            avg_score = np.mean(self.training_scores[-10:]) if len(self.training_scores) >= 10 else episode_score
            avg_loss = np.mean(self.training_losses[-100:]) if len(self.training_losses) >= 100 else 0
            
            print(f"🏁 Episode {episode + 1} Complete:")
            print(f"   📊 Score: {episode_score} (Best: {self.best_score})")
            print(f"   🎯 Progress: {(episode_score / self.current_high_score) * 100:.1f}%")
            print(f"   💰 Total Reward: {episode_reward:.2f}")
            print(f"   📈 Avg Score (10): {avg_score:.1f}")
            print(f"   🧠 Avg Loss: {avg_loss:.4f}")
            print(f"   🔍 Epsilon: {self.epsilon:.3f}")
            print(f"   🗃️ Buffer Size: {len(self.replay_buffer.buffer)}")
            
            # Check for success
            if episode_score > self.current_high_score:
                print("🎉 LEADERBOARD BEATEN! RL TRAINING SUCCESSFUL!")
                break
            
            # Early stopping if performance is good
            if len(self.training_scores) >= 10:
                recent_avg = np.mean(self.training_scores[-10:])
                if recent_avg > self.current_high_score * 0.8:  # 80% of target
                    print("🚀 Strong performance detected, continuing training...")
        
        # Training complete
        print("\n🧠 REINFORCEMENT LEARNING TRAINING COMPLETE!")
        print(f"🏆 Best Score Achieved: {self.best_score}")
        print(f"🎯 Leaderboard Beaten: {self.leaderboard_beaten}")
        
        # Save model
        self.save_model()
        
        # Create training visualization
        self.create_training_visualization()
        
        return self.best_score
    
    def restart_game(self):
        """Restart the game for new episode"""
        try:
            # Refresh page
            self.driver.refresh()
            time.sleep(3)
            
            # Navigate back to game
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.65);")
            time.sleep(2)
            
            # Reset game state
            self.score = 0
            self.level = 1
            self.lives = 3
            
            # Start game
            self.start_rl_game()
            
        except Exception as e:
            print(f"Game restart error: {e}")
    
    def save_model(self):
        """Save the trained model"""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_scores': self.training_scores,
                'best_score': self.best_score,
                'epsilon': self.epsilon
            }, 'rl_space_invaders_model.pth')
            print("💾 RL model saved successfully!")
        except Exception as e:
            print(f"Model save error: {e}")
    
    def load_model(self, filepath='rl_space_invaders_model.pth'):
        """Load a pre-trained model"""
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_scores = checkpoint.get('training_scores', [])
            self.best_score = checkpoint.get('best_score', 0)
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print("📂 RL model loaded successfully!")
        except Exception as e:
            print(f"Model load error: {e}")
    
    def create_training_visualization(self):
        """Create comprehensive training visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Score progression
            axes[0, 0].plot(self.training_scores)
            axes[0, 0].axhline(y=self.current_high_score, color='r', linestyle='--', label='Target Score')
            axes[0, 0].set_title('RL Training Score Progression')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Episode rewards
            if self.episode_rewards:
                axes[0, 1].plot(self.episode_rewards)
                axes[0, 1].set_title('Episode Rewards')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Total Reward')
                axes[0, 1].grid(True)
            
            # Training loss
            if self.training_losses:
                axes[1, 0].plot(self.training_losses)
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True)
            
            # Performance summary
            axes[1, 1].text(0.1, 0.8, f"Best Score: {self.best_score}", fontsize=14)
            axes[1, 1].text(0.1, 0.6, f"Target Score: {self.current_high_score}", fontsize=14)
            axes[1, 1].text(0.1, 0.4, f"Success Rate: {(self.best_score/self.current_high_score)*100:.1f}%", fontsize=14)
            axes[1, 1].text(0.1, 0.2, f"Episodes Trained: {len(self.training_scores)}", fontsize=14)
            axes[1, 1].set_title('RL Training Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'RL_Training_Analysis_{self.best_score}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Training visualization saved!")
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def run_rl_session(self):
        """Run the complete RL training session"""
        try:
            print("🧠 REINFORCEMENT LEARNING SPACE INVADERS AI")
            print("🎯 Advanced DQN with Prioritized Experience Replay")
            
            self.setup_advanced_browser()
            
            if not self.navigate_and_analyze():
                return 0
            
            # Train the agent
            final_score = self.train_rl_agent(episodes=50)  # Start with 50 episodes
            
            return final_score
            
        except Exception as e:
            print(f"❌ RL session error: {e}")
            return 0
        finally:
            if self.leaderboard_beaten and not self.name_entered:
                print("🧠 RL AI: Extended monitoring for champion name entry...")
                
                for i in range(10):
                    time.sleep(10)
                    try:
                        if self.check_for_high_score_entry():
                            self.enter_champion_name()
                            break
                    except:
                        pass
                    print(f"🧠 RL monitoring... {(i+1)*10}s")
            
            if self.driver:
                if self.leaderboard_beaten:
                    print("🎉 RL AI Champion! Closing in 60 seconds...")
                    time.sleep(60)
                self.driver.quit()

def main():
    """Main function for Reinforcement Learning Space Invaders AI"""
    print("🧠" + "="*90 + "🧠")
    print("🎮 REINFORCEMENT LEARNING SPACE INVADERS AI")
    print("🤖 Deep Q-Network (DQN) with Prioritized Experience Replay")
    print("🎯 Mission: Achieve high score through advanced RL training")
    print("🧠 Features: Target Networks, Experience Replay, Advanced Rewards")
    print("="*94)
    
    rl_ai = ReinforcementLearningAI()
    final_score = rl_ai.run_rl_session()
    
    print("\n" + "🧠" + "="*90 + "🧠")
    print("🏁 REINFORCEMENT LEARNING RESULTS")
    print("🧠" + "="*90 + "🧠")
    print(f"🏆 RL AI Best Score: {rl_ai.best_score}")
    print(f"🎯 Target Score: {rl_ai.current_high_score}")
    print(f"📈 Success Rate: {(rl_ai.best_score/rl_ai.current_high_score)*100:.1f}%")
    print(f"🧠 Episodes Trained: {len(rl_ai.training_scores)}")
    print(f"🎯 Leaderboard Beaten: {rl_ai.leaderboard_beaten}")
    print(f"📝 Champion Name Saved: {'John H' if rl_ai.name_entered else 'No'}")
    
    if rl_ai.leaderboard_beaten:
        print("🎉 REINFORCEMENT LEARNING SUCCESS!")
        print("👑 RL AI has mastered Space Invaders!")
    else:
        print("📈 RL AI is learning and improving with each episode")
        print("🧠 Advanced training algorithms are optimizing performance")
    
    print("\n🤖 Reinforcement Learning Space Invaders AI - Mission Complete!")

if __name__ == "__main__":
    main()