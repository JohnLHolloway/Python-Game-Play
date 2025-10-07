"""
Test script to verify the Space Invaders AI setup
"""

import sys
import time

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing package imports...")
    
    try:
        import selenium
        print("✅ Selenium imported successfully")
    except ImportError as e:
        print(f"❌ Selenium import failed: {e}")
        return False
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ WebDriver Manager imported successfully")
    except ImportError as e:
        print(f"❌ WebDriver Manager import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import pyautogui
        print("✅ PyAutoGUI imported successfully")
    except ImportError as e:
        print(f"❌ PyAutoGUI import failed: {e}")
        return False
    
    return True

def test_webdriver():
    """Test if Chrome WebDriver can be initialized"""
    print("\n🧪 Testing Chrome WebDriver...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background for test
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test basic functionality
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print("✅ Chrome WebDriver test successful")
        return True
        
    except Exception as e:
        print(f"❌ Chrome WebDriver test failed: {e}")
        return False

def test_website_access():
    """Test if the target website is accessible"""
    print("\n🧪 Testing website access...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test target website
        driver.get("https://jordancota.site/")
        time.sleep(3)
        
        page_source = driver.page_source.lower()
        
        if "space invaders" in page_source or "invaders" in page_source:
            print("✅ Target website accessible - Space Invaders game found")
            result = True
        else:
            print("⚠️ Target website accessible but game not clearly detected")
            result = True  # Website is still accessible
        
        driver.quit()
        return result
        
    except Exception as e:
        print(f"❌ Website access test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Space Invaders AI Bot - System Test")
    print("="*50)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: WebDriver functionality
    if not test_webdriver():
        all_tests_passed = False
    
    # Test 3: Website access
    if not test_website_access():
        all_tests_passed = False
    
    print("\n" + "="*50)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The AI bot is ready to run.")
        print("\n🚀 To start the bot, run:")
        print("   python advanced_space_invaders_ai.py")
        print("   or")
        print("   python run_bot.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🛠️ Try installing missing packages:")
        print("   pip install -r requirements.txt")
    
    print("\n🎯 Target: Beat 25,940 points on https://jordancota.site/")

if __name__ == "__main__":
    main()