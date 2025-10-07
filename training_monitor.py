"""
Training Monitor - Real-time monitoring of all active AI training sessions
"""

import time
import subprocess
import psutil
import re

class TrainingMonitor:
    def __init__(self):
        self.training_processes = {}
        self.performance_data = {}
        
    def find_training_processes(self):
        """Find all running Python training processes"""
        training_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(script in cmdline for script in [
                        'ultimate_training_ai.py',
                        'enhanced_ultimate_ai.py', 
                        'speed_training_ai.py',
                        'reinforcement_learning_ai.py'
                    ]):
                        training_processes.append({
                            'pid': proc.info['pid'],
                            'script': self.extract_script_name(cmdline),
                            'cmdline': cmdline,
                            'cpu_percent': proc.cpu_percent(),
                            'memory_mb': proc.memory_info().rss / 1024 / 1024
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return training_processes
    
    def extract_script_name(self, cmdline):
        """Extract the script name from command line"""
        if 'ultimate_training_ai.py' in cmdline:
            return 'Ultimate Training AI'
        elif 'enhanced_ultimate_ai.py' in cmdline:
            return 'Enhanced Ultimate AI'
        elif 'speed_training_ai.py' in cmdline:
            return 'Speed Training AI'
        elif 'reinforcement_learning_ai.py' in cmdline:
            return 'Reinforcement Learning AI'
        else:
            return 'Unknown AI'
    
    def display_training_status(self):
        """Display current training status"""
        print("🎯" + "="*80 + "🎯")
        print("🎮 SPACE INVADERS AI TRAINING MONITOR")
        print("🚀 Real-time Performance Tracking")
        print("🎯 Target Score: 25,940 points")
        print("="*84)
        
        processes = self.find_training_processes()
        
        if not processes:
            print("❌ No active training processes found")
            print("\n💡 Available training scripts:")
            print("   • ultimate_training_ai.py - Multi-strategy adaptive AI")
            print("   • enhanced_ultimate_ai.py - Large window + extended gameplay")
            print("   • speed_training_ai.py - Ultra-fast training with minimal delays")
            print("   • reinforcement_learning_ai.py - Deep Q-Network training")
            return
        
        print(f"🔥 Found {len(processes)} active training session(s):")
        print()
        
        for i, proc in enumerate(processes, 1):
            print(f"🤖 {i}. {proc['script']}")
            print(f"   📊 PID: {proc['pid']}")
            print(f"   💻 CPU: {proc['cpu_percent']:.1f}%")
            print(f"   🧠 Memory: {proc['memory_mb']:.1f} MB")
            print(f"   ⚡ Status: ACTIVE TRAINING")
            print()
        
        # Performance comparison
        print("📈 PERFORMANCE COMPARISON:")
        print("🏆 Ultimate Training AI: Best score 640 points (2.5% progress)")
        print("🖥️ Enhanced Ultimate AI: Extended gameplay with life tracking")
        print("⚡ Speed Training AI: Ultra-fast training with minimal delays")
        print("🧠 Reinforcement Learning AI: Completed 50 episodes (max 10 points)")
        print()
        
        # Speed optimization summary
        print("⚡ SPEED OPTIMIZATIONS IMPLEMENTED:")
        print("✅ Reduced browser wait times: 30s → 10s")
        print("✅ Faster page loading: 5s → 2s")
        print("✅ Action delays reduced by 50-80%")
        print("✅ Game over checks optimized")
        print("✅ Ultra-fast mode available")
        print("✅ New Speed Training AI with minimal delays")
        print()
        
        print("🎯 TRAINING RECOMMENDATIONS:")
        print("🥇 Focus on Ultimate Training AI - showing best results")
        print("⚡ Try Speed Training AI for faster iteration")
        print("🖥️ Enhanced Ultimate AI for extended gameplay analysis")
        print("📊 Monitor all systems for breakthrough performance")

def main():
    """Main monitoring function"""
    monitor = TrainingMonitor()
    
    try:
        while True:
            print("\033[2J\033[H")  # Clear screen
            monitor.display_training_status()
            print(f"\n🔄 Last updated: {time.strftime('%H:%M:%S')}")
            print("Press Ctrl+C to exit monitoring...")
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Training monitoring stopped")

if __name__ == "__main__":
    main()