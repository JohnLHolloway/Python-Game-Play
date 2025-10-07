#!/usr/bin/env python3
"""
Advanced Space Invaders RL Agent for https://jordancota.site/
=================================================================
Goal:
  - Optimize for maximum total score and survival (3 lives)
  - Surpass on‑page high score dynamically (no hardcoded target)
  - Prompt user for champion name when beating current high score

Technique:
  - Vision-based Deep Reinforcement Learning (Dueling Noisy DQN subset of Rainbow)
  - Prioritized Experience Replay (rank-based simplified)
  - Frame stacking (4 grayscale 84x84 frames)
  - Action set: [NOOP, LEFT, RIGHT, SHOOT, LEFT+SHOOT, RIGHT+SHOOT]

Notes / Caveats:
  - Training via live browser is inherently slow (real time). Expect many hours for strong policies.
  - For experimentation: start with evaluation_mode=True to verify interaction & state capture.
  - Consider later migrating to a local JS canvas extraction or custom headless game port for speed.
  - This script focuses on correctness & extensibility rather than raw training throughput.

Run Example:
  python advanced_jordancota_rl_ai.py --headless --episodes 5 --evaluation

Requirements (add to requirements.txt if missing):
  tensorflow (optional if you keep earlier scripts)
  torch
  torchvision
  pillow
  selenium
  numpy

"""
from __future__ import annotations
import os
import io
import time
import math
import random
import argparse
import threading
import base64
import re
import sys
import pickle
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from collections import deque

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

# =============================
# Configuration & Hyperparameters
# =============================
ACTION_MEANING = ["NOOP", "LEFT", "RIGHT", "SHOOT", "LEFT_SHOOT", "RIGHT_SHOOT"]
N_ACTIONS = len(ACTION_MEANING)
FRAME_SIZE = 84
STACK_SIZE = 4

DEFAULT_CONFIG = dict(
    discount=0.99,
    learning_rate=1e-4,
    batch_size=32,
    replay_capacity=50000,
    min_replay_size=5000,
    update_target_every=5000,
    train_freq=4,
    prioritized=True,
    alpha=0.6,              # PER exponent
    beta_start=0.4,
    beta_frames=200000,
    eps_priority=1e-6,
    noisy=True,
    dueling=True,
    double_dqn=True,
    pixel_diff=True,
    max_steps_per_episode=1500,
    life_loss_penalty=-500.0,
    survival_reward=1.0,
    score_scale=1.0,
    replay_pickle_path='replay_buffer.pkl',
    replay_save_every=2000,
)


# =============================
# Utility: Noisy Linear Layer
# =============================
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init: float):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# =============================
# Dueling Noisy DQN Network
# =============================
class DQN(nn.Module):
    def __init__(self, action_dim: int, noisy: bool = True, dueling: bool = True):
        super().__init__()
        self.noisy = noisy
        self.dueling = dueling
        # (STACK_SIZE, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, 8, stride=4),  # -> 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),          # -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),          # -> 64x7x7
            nn.ReLU(),
        )
        conv_out = 64 * 7 * 7

        def linear(in_f, out_f):
            if noisy:
                return NoisyLinear(in_f, out_f)
            return nn.Linear(in_f, out_f)

        if dueling:
            self.advantage = nn.Sequential(
                linear(conv_out, 512), nn.ReLU(), linear(512, action_dim)
            )
            self.value = nn.Sequential(
                linear(conv_out, 512), nn.ReLU(), linear(512, 1)
            )
        else:
            self.head = nn.Sequential(
                linear(conv_out, 512), nn.ReLU(), linear(512, action_dim)
            )

    def forward(self, x):
        x = x / 255.0  # normalize pixel values
        feats = self.conv(x).view(x.size(0), -1)
        if self.dueling:
            adv = self.advantage(feats)
            val = self.value(feats)
            q = val + adv - adv.mean(dim=1, keepdim=True)
            return q
        else:
            return self.head(feats)

    def reset_noise(self):
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# =============================
# Prioritized Replay Buffer (rank-based simplified)
# =============================
class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float, eps: float):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.pos = 0

    def add(self, transition, priority: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == 0:
            raise ValueError("Empty buffer")
        # rank-based: sort priorities descending
        priorities = np.array(self.priorities)
        ranks = priorities.argsort()[::-1]
        probs = 1.0 / (np.arange(1, len(ranks) + 1))
        probs = probs ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(ranks, batch_size, p=probs)
        # Importance sampling weights
        weights = (len(self.buffer) * probs[np.searchsorted(ranks, indices)]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, new_priorities):
        for idx, p in zip(indices, new_priorities):
            self.priorities[idx] = float(p)

    def __len__(self):
        return len(self.buffer)


class BasicReplay:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def add(self, transition, *_):
        self.buffer.append(transition)

    def sample(self, batch_size: int, *_):
        batch = random.sample(self.buffer, batch_size)
        weights = torch.ones(batch_size, dtype=torch.float32)
        indices = None
        return batch, indices, weights

    def update_priorities(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.buffer)


# =============================
# Environment Wrapper
# =============================
class JordanCotaEnv:
    def __init__(self, headless: bool = False, verbose: bool = True, config: dict | None = None):
        self.url = "https://jordancota.site/"
        self.verbose = verbose
        self.lives = 3
        self.prev_lives = 3
        self.score = 0
        self.prev_score = 0
        self.high_score = 0
        self.done = False
        self.action_delay = 0.085  # seconds between actions
        self.canvas_selector = 'canvas'
        self.frame_stack = deque(maxlen=STACK_SIZE)
        self.prev_raw_frame = None
        self.has_canvas = False
        self.game_started = False
        self.driver = self._init_driver(headless)
        self.config = config or DEFAULT_CONFIG
        self._load_game()
        self._detect_high_score()

    # ---------- Browser ----------
    def _init_driver(self, headless: bool):
        opts = Options()
        if headless:
            opts.add_argument('--headless=new')
        opts.add_argument('--disable-gpu')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('--window-size=1200,900')
        # Use webdriver-manager for robust driver provisioning
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        except Exception:
            # Fallback to default resolution (expects chromedriver in PATH)
            driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(40)
        return driver

    def _load_game(self):
        if self.verbose:
            print("🌐 Loading game page...")
        self.driver.get(self.url)
        time.sleep(4)
        self._scroll_to_game_section()
        # Detect canvas if present
        try:
            self.driver.find_element(By.CSS_SELECTOR, self.canvas_selector)
            self.has_canvas = True
            if self.verbose:
                print("🟢 Canvas element detected.")
        except Exception:
            self.has_canvas = False
            if self.verbose:
                print("🟡 No canvas detected; using body screenshots.")
        self._try_start_game(max_attempts=5)

    def _focus_canvas(self):
        try:
            if self.has_canvas:
                canvas = self.driver.find_element(By.CSS_SELECTOR, self.canvas_selector)
                canvas.click()
            else:
                self.driver.find_element(By.TAG_NAME, 'body').click()
        except Exception:
            pass

    def _start_game_sequence(self):
        keys = [Keys.SPACE, Keys.ENTER, '1', 's']
        body = self.driver.find_element(By.TAG_NAME, 'body')
        for k in keys:
            try:
                body.send_keys(k)
                time.sleep(0.3)
            except Exception:
                pass
        self._focus_canvas()
        # Also dispatch JS keyboard events which some games rely on
        try:
            for code in [32, 13, 37, 39]:  # space, enter, left, right
                self.driver.execute_script(
                    "var e=new KeyboardEvent('keydown',{keyCode:arguments[0],which:arguments[0]});document.dispatchEvent(e);",
                    code
                )
        except Exception:
            pass

    # -------- New helpers for game start -------
    def _scroll_to_game_section(self):
        phrases = ["Fun Time", "Play Space Invaders", "Score:"]
        for i in range(10):
            try:
                body_text = self.driver.find_element(By.TAG_NAME, 'body').text
                if any(p.lower() in body_text.lower() for p in phrases):
                    for p in phrases:
                        try:
                            el = self.driver.find_element(By.XPATH, f"//*[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), '{p.lower()}')]")
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior:'instant', block:'center'});", el)
                            if self.verbose:
                                print("🔎 Game section located and scrolled into view.")
                            return
                        except Exception:
                            continue
                self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight/9);")
                time.sleep(0.3)
            except Exception:
                break
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.7);")
        except Exception:
            pass

    def _click_start_button(self) -> bool:
        candidate_texts = ["Start Game", "START GAME", "Start"]
        for txt in candidate_texts:
            try:
                el = self.driver.find_element(By.XPATH, f"//*[contains(text(), '{txt}')]")
                self.driver.execute_script("arguments[0].scrollIntoView({behavior:'instant', block:'center'});", el)
                el.click()
                if self.verbose:
                    print(f"🖱️ Clicked start element: {txt}")
                time.sleep(0.5)
                return True
            except Exception:
                continue
        return False

    def _score_line_text(self) -> str:
        try:
            body_text = self.driver.find_element(By.TAG_NAME, 'body').text
            # Attempt to isolate line containing Score:
            for line in body_text.splitlines():
                if 'Score:' in line and 'Lives:' in line:
                    return line.strip()
            m = re.search(r"Score:.*Lives:.*", body_text)
            if m:
                return m.group(0)[:120]
            return body_text[:300]
        except Exception:
            return ""

    def _try_start_game(self, max_attempts: int = 5):
        # New approach: click start then poll scoreboard regex
        start_clicked = self._click_start_button()
        if not start_clicked and self.verbose:
            print("ℹ️ Could not find explicit Start Game element; will attempt key spam only.")
        self._start_game_sequence()
        poll_start = time.time()
        while time.time() - poll_start < 8:  # up to 8 seconds polling
            body_text = ''
            try:
                body_text = self.driver.find_element(By.TAG_NAME, 'body').text
            except Exception:
                pass
            if body_text:
                if re.search(r"Score:\s*\d+.*?Lives:\s*\d+", body_text, re.IGNORECASE | re.DOTALL):
                    self.game_started = True
                    if self.verbose:
                        print("✅ Game appears to have started (score & lives detected).")
                    break
            # send a space every 0.5s to ensure game un-pauses
            try:
                self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)
            except Exception:
                pass
            time.sleep(0.5)
        if not self.game_started and self.verbose:
            print("⚠️ Could not confirm game start after polling; proceeding anyway.")

    def _detect_high_score(self):
        try:
            body_text = self.driver.find_element(By.TAG_NAME, 'body').text
            # Try to find patterns like "High Score: 25940"
            matches = re.findall(r"High\s*Score\s*:?\s*(\d+)", body_text, re.IGNORECASE)
            if matches:
                self.high_score = max(int(m) for m in matches)
            else:
                # fallback: largest number on page below some threshold
                nums = re.findall(r"\b(\d{2,6})\b", body_text)
                if nums:
                    self.high_score = max(int(n) for n in nums)
            if self.verbose:
                print(f"🏁 Detected high score on page: {self.high_score}")
        except Exception:
            if self.verbose:
                print("⚠️ Could not detect high score; defaulting to 0")
            self.high_score = 0

    # ---------- Frame Processing ----------
    def _grab_frame(self) -> np.ndarray:
        # Use canvas if present; otherwise capture viewport region around score line
        if self.has_canvas:
            try:
                canvas = self.driver.find_element(By.CSS_SELECTOR, self.canvas_selector)
                png_bytes = canvas.screenshot_as_png
                img = Image.open(io.BytesIO(png_bytes)).convert('L')
                img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
                return np.array(img, dtype=np.uint8)
            except Exception:
                pass
        # Fallback full screenshot cropping
        try:
            full_png = self.driver.get_screenshot_as_png()
            img_full = Image.open(io.BytesIO(full_png)).convert('L')
            w, h = img_full.size
            crop_box = (int(w*0.1), int(h*0.35), int(w*0.9), int(h*0.95))
            img_cropped = img_full.crop(crop_box)
            img_cropped = img_cropped.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
            return np.array(img_cropped, dtype=np.uint8)
        except Exception:
            # last resort blank frame
            return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

    def _update_score_and_lives(self):
        # Regex scoreboard parse: Score: <n> Level: <n> Lives: <n>
        try:
            body_text = self.driver.find_element(By.TAGNAME, 'body').text
        except Exception:
            try:
                body_text = self.driver.find_element(By.TAG_NAME, 'body').text
            except Exception:
                return
        try:
            m = re.search(r"Score:\s*(\d+).*?Lives:\s*(\d+)", body_text, re.IGNORECASE | re.DOTALL)
            if m:
                new_score = int(m.group(1))
                new_lives = int(m.group(2))
                # Only advance score forward
                if new_score >= self.score:
                    self.score = new_score
                self.lives = min(3, max(0, new_lives))
        except Exception:
            pass

    def _handle_alert(self):
        # Capture Game Over alert if present to prevent exceptions
        try:
            alert = self.driver.switch_to.alert
            text = alert.text
            # Extract final score
            m = re.search(r"Final Score:\s*(\d+)", text)
            if m:
                final_score = int(m.group(1))
                if final_score > self.score:
                    self.score = final_score
            alert.dismiss()
            self.lives = 0
            self.done = True
            if self.verbose:
                print(f"🛑 Game Over alert handled. Final Score {self.score}")
            return True
        except Exception:
            return False

    # ---------- Public Gym-like API ----------
    def reset(self):
        self.driver.refresh()
        time.sleep(4)
        self.score = 0
        self.prev_score = 0
        self.lives = 3
        self.prev_lives = 3
        self.done = False
        self._scroll_to_game_section()
        # re-detect canvas
        try:
            self.driver.find_element(By.CSS_SELECTOR, self.canvas_selector)
            self.has_canvas = True
        except Exception:
            self.has_canvas = False
        self._try_start_game(max_attempts=3)
        self.frame_stack.clear()
        frame = self._grab_frame()
        self.prev_raw_frame = frame.copy()
        for _ in range(STACK_SIZE):
            self.frame_stack.append(frame)
        return np.stack(self.frame_stack, axis=0)

    def step(self, action: int):
        assert 0 <= action < N_ACTIONS
        self._perform_action(action)
        time.sleep(self.action_delay)
        frame = self._grab_frame()
        # Check alert first
        if self._handle_alert():
            # treat as terminal, frame remains
            self._update_score_and_lives()
            self.frame_stack.append(frame if not self.config.get('pixel_diff', True) else np.zeros_like(frame))
            stacked = np.stack(self.frame_stack, axis=0)
            info = {"score": self.score, "lives": self.lives, "high_score": self.high_score}
            return stacked, 0.0, True, info
        self._update_score_and_lives()

        reward = 0.0
        # Score delta
        if self.score > self.prev_score:
            reward += (self.score - self.prev_score) * DEFAULT_CONFIG['score_scale']
        # Survival reward
        reward += DEFAULT_CONFIG['survival_reward']
        # Life lost
        if self.lives < self.prev_lives:
            reward += DEFAULT_CONFIG['life_loss_penalty']

        self.prev_score = self.score
        self.prev_lives = self.lives

        if self.config.get('pixel_diff', True) and self.prev_raw_frame is not None:
            diff = np.clip(np.abs(frame.astype(int) - self.prev_raw_frame.astype(int)), 0, 255).astype(np.uint8)
            composite = frame  # could merge info later
            # Replace last frame with diff for motion emphasis
            self.frame_stack.append(diff)
        else:
            self.frame_stack.append(frame)
        self.prev_raw_frame = frame.copy()
        stacked = np.stack(self.frame_stack, axis=0)

        # Terminal condition: lives depleted OR max steps externally enforced
        done = self.lives <= 0
        self.done = done
        info = {"score": self.score, "lives": self.lives, "high_score": self.high_score}
        return stacked, reward, done, info

    def _perform_action(self, action: int):
        # Choose target element for key dispatch
        if self.has_canvas:
            try:
                target = self.driver.find_element(By.CSS_SELECTOR, self.canvas_selector)
            except Exception:
                target = self.driver.find_element(By.TAG_NAME, 'body')
        else:
            target = self.driver.find_element(By.TAG_NAME, 'body')
        if action == 0:
            return  # NOOP
        keys = []
        if action in (1, 4):  # LEFT or LEFT+SHOOT
            keys.append(Keys.ARROW_LEFT)
        if action in (2, 5):  # RIGHT or RIGHT+SHOOT
            keys.append(Keys.ARROW_RIGHT)
        if action in (3, 4, 5):  # SHOOT or combos
            keys.append(Keys.SPACE)
        for k in keys:
            try:
                target.send_keys(k)
            except Exception:
                # if alert popped mid-send, handle it
                self._handle_alert()
        # Also try JS key events for reliability
        try:
            for k in keys:
                code_map = {Keys.ARROW_LEFT:37, Keys.ARROW_RIGHT:39, Keys.SPACE:32}
                key_code = code_map.get(k, 0)
                if key_code:
                    self.driver.execute_script(
                        "var e=new KeyboardEvent('keydown',{keyCode:arguments[0],which:arguments[0]});document.dispatchEvent(e);",
                        key_code
                    )
        except Exception:
            pass

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


# =============================
# Agent Wrapper
# =============================
class Agent:
    def __init__(self, config: dict, device: str = 'cpu'):
        self.cfg = config
        self.device = device
        self.online = DQN(N_ACTIONS, noisy=config['noisy'], dueling=config['dueling']).to(device)
        self.target = DQN(N_ACTIONS, noisy=config['noisy'], dueling=config['dueling']).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=config['learning_rate'])
        self.learn_step = 0
        if config['prioritized']:
            self.replay = PrioritizedReplay(config['replay_capacity'], config['alpha'], config['eps_priority'])
        else:
            self.replay = BasicReplay(config['replay_capacity'])

    def select_action(self, state: np.ndarray, epsilon: float = 0.0):
        # If using noisy nets, epsilon-greedy not required; still allow evaluation epsilon override
        if (not self.cfg['noisy']) and (random.random() < epsilon):
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    def push(self, transition, priority: Optional[float] = None):
        if priority is None:
            priority = 1.0
        self.replay.add(transition, priority)

    def beta_by_frame(self, frame_idx):
        b0, bf = self.cfg['beta_start'], 1.0
        return b0 + (bf - b0) * frame_idx / self.cfg['beta_frames']

    def train_step(self, frame_idx: int):
        if len(self.replay) < self.cfg['min_replay_size']:
            return None
        batch, indices, weights = self.replay.sample(self.cfg['batch_size'], self.beta_by_frame(frame_idx))
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = weights.to(self.device).unsqueeze(1)

        # Q estimates
        q_values = self.online(states).gather(1, actions)
        with torch.no_grad():
            if self.cfg.get('double_dqn', True):
                next_actions = self.online(next_states).argmax(1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
            else:
                next_q = self.target(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.cfg['discount'] * next_q

        td_error = target_q - q_values
        loss = (weights * td_error.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        if self.cfg['prioritized'] and indices is not None:
            new_p = td_error.detach().cpu().abs().numpy().flatten() + self.cfg['eps_priority']
            self.replay.update_priorities(indices, new_p)

        # Target network sync
        if self.learn_step % self.cfg['update_target_every'] == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.learn_step += 1
        if self.online.noisy:
            self.online.reset_noise()
            self.target.reset_noise()
        return float(loss.item())

    def save(self, path: str):
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'cfg': self.cfg
        }, path)

    def load(self, path: str, map_location=None):
        data = torch.load(path, map_location=map_location)
        self.online.load_state_dict(data['online'])
        self.target.load_state_dict(data['target'])


# =============================
# Training / Evaluation Loop
# =============================
def run(config: dict, episodes: int, headless: bool, evaluation_mode: bool, model_path: str, prompt_on_highscore: bool):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    env = JordanCotaEnv(headless=headless, verbose=True, config=config)
    agent = Agent(config, device=device)
    frame_idx = 0
    global_high_score = env.high_score
    champion_recorded = False

    if evaluation_mode and os.path.isfile(model_path):
        agent.load(model_path, map_location=device)
        print(f"✅ Loaded model from {model_path}")

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0.0
            lives_at_start = env.lives
            for step in range(config['max_steps_per_episode']):
                epsilon = 0.0 if (config['noisy'] or evaluation_mode) else max(0.01, 1.0 - frame_idx / 200000)
                action = agent.select_action(state, epsilon)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                transition = (state, action, reward, next_state, done)
                priority = abs(reward) + 1e-4
                agent.push(transition, priority)
                state = next_state
                frame_idx += 1

                if (not evaluation_mode) and frame_idx % config['train_freq'] == 0:
                    loss = agent.train_step(frame_idx)
                else:
                    loss = None

                if frame_idx % 1000 == 0 and (not evaluation_mode):
                    agent.save(model_path)
                    print(f"💾 Saved checkpoint at frame {frame_idx}")

                # Periodic replay buffer persistence
                if (not evaluation_mode) and config.get('replay_pickle_path') and frame_idx % config.get('replay_save_every', 2000) == 0 and len(agent.replay) > config['min_replay_size']:
                    try:
                        if hasattr(agent.replay, 'buffer'):
                            with open(config['replay_pickle_path'], 'wb') as pf:
                                pickle.dump(agent.replay, pf)
                            print(f"🧪 Saved replay buffer ({len(agent.replay)} transitions)")
                    except Exception as e:
                        print(f"⚠️ Replay save failed: {e}")

                # Dynamic high score surpass detection
                if info['score'] > global_high_score and prompt_on_highscore and not champion_recorded:
                    print(f"\n🏆 New High Score: {info['score']} (previous {global_high_score})")
                    if sys.stdin is not None and sys.stdin.isatty():
                        champion_name = input("Enter champion name: ").strip() or "ANONYMOUS"
                    else:
                        champion_name = "AUTO_AGENT"
                    with open('champion_record.json', 'w') as f:
                        f.write(repr({'name': champion_name, 'score': info['score'], 'frame': frame_idx, 'episode': ep}))
                    champion_recorded = True
                    global_high_score = info['score']

                if done:
                    break

            print(f"EP {ep}/{episodes} | Score: {info['score']} | Reward: {episode_reward:.1f} | Lives Used: {lives_at_start - info['lives']} | Frames: {frame_idx}")
    finally:
        if not evaluation_mode:
            agent.save(model_path)
            print("💾 Final model saved.")
        env.close()


# =============================
# CLI Interface
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="Advanced RL Agent for jordancota Space Invaders")
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes (browser sessions)')
    parser.add_argument('--headless', action='store_true', help='Run Chrome headless')
    parser.add_argument('--evaluation', action='store_true', help='Evaluation only (no training)')
    parser.add_argument('--model-path', type=str, default='jordancota_dqn.pt', help='Checkpoint path')
    parser.add_argument('--no-double', action='store_true', help='Disable Double DQN')
    parser.add_argument('--no-pixel-diff', action='store_true', help='Disable pixel difference channel substitution')
    parser.add_argument('--no-prompt', action='store_true', help='Disable high score prompt')
    return parser.parse_args()


def main():
    args = parse_args()
    print("🚀 Starting Advanced JordanCota Space Invaders RL Agent")
    cfg = DEFAULT_CONFIG.copy()
    if args.no_double:
        cfg['double_dqn'] = False
    if args.no_pixel_diff:
        cfg['pixel_diff'] = False
    run(cfg, args.episodes, args.headless, args.evaluation, args.model_path, not args.no_prompt)


if __name__ == '__main__':
    main()
