"""
Simple test to verify we can access and interact with the Space Invaders game
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def test_game_interaction():
    """Simple test to verify game access and interaction"""
    print("🧪 Testing Space Invaders game interaction...")
    
    # Setup browser
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1400,900")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Navigate to site
        print("🌐 Navigating to https://jordancota.site/...")
        driver.get("https://jordancota.site/")
        time.sleep(5)
        
        # Scroll to find game
        print("🔍 Scrolling to find game...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
        time.sleep(3)
        
        # Take a screenshot to see what we have
        driver.save_screenshot("game_page.png")
        print("📸 Screenshot saved as game_page.png")
        
        # Try to find any interactive elements
        print("🔍 Looking for interactive elements...")
        
        # Look for canvas
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        print(f"📊 Found {len(canvases)} canvas elements")
        
        # Look for buttons
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"📊 Found {len(buttons)} button elements")
        
        for i, button in enumerate(buttons):
            try:
                text = button.text.strip()
                if text:
                    print(f"  Button {i+1}: '{text}'")
            except:
                pass
        
        # Look for game-related text
        page_text = driver.page_source.lower()
        game_keywords = ['space invaders', 'score', 'level', 'lives', 'start', 'game']
        found_keywords = [kw for kw in game_keywords if kw in page_text]
        print(f"📊 Found game keywords: {found_keywords}")
        
        # Try to interact with canvas if found
        if canvases:
            print("🎮 Attempting to interact with first canvas...")
            canvas = canvases[0]
            
            # Click on canvas
            ActionChains(driver).move_to_element(canvas).click().perform()
            time.sleep(1)
            
            # Try sending some game keys
            print("⌨️ Sending test keys...")
            ActionChains(driver).send_keys(Keys.SPACE).perform()
            time.sleep(0.5)
            ActionChains(driver).send_keys(Keys.ARROW_LEFT).perform()
            time.sleep(0.5)
            ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
            time.sleep(0.5)
            ActionChains(driver).send_keys(Keys.SPACE).perform()
            
            # Take another screenshot after interaction
            driver.save_screenshot("game_after_interaction.png")
            print("📸 Screenshot after interaction saved")
        
        # Look for score or game state changes
        print("📊 Checking for game state...")
        current_page = driver.page_source
        
        # Look for score patterns
        import re
        score_matches = re.findall(r'score[:\s]*(\d+)', current_page, re.IGNORECASE)
        if score_matches:
            print(f"🎯 Found score values: {score_matches}")
        else:
            print("🎯 No score values found")
        
        # Wait a bit to see if anything changes
        print("⏳ Waiting 10 seconds to observe changes...")
        time.sleep(10)
        
        # Final screenshot
        driver.save_screenshot("game_final_state.png")
        print("📸 Final screenshot saved")
        
        print("✅ Test completed! Check the screenshots to see the game state.")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        input("Press Enter to close browser...")
        driver.quit()

if __name__ == "__main__":
    test_game_interaction()