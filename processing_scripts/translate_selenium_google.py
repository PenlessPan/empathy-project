# Input directory containing dialog files
INPUT_DIR = "dialog_text"

# Output directory for translated files
OUTPUT_DIR = "translated_text_trial1"

# Delay between translation requests (seconds)
# Increase this value if you see translation failures
DELAY = 3

import os
import re
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

class GoogleTranslateWeb:
    def __init__(self, source_lang="iw", target_lang="en", headless=True):
        """
        Initialize Google Translate web interface automation.
        
        Args:
            source_lang: Source language code (iw for Hebrew)
            target_lang: Target language code (en for English)
            headless: Run browser in headless mode (no GUI)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialize browser
        print("Initializing Chrome browser...")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # Navigate to Google Translate with the specified languages
        self.url = f"https://translate.google.com/?sl={source_lang}&tl={target_lang}&op=translate"
        self.driver.get(self.url)
        time.sleep(2)  # Wait for page to load
        print("Browser initialized and ready")
        
    def translate(self, text):
        """
        Translate text using Google Translate website.
        
        Args:
            text: Text to translate
        
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
        
        try:
            # Clear previous input if any
            clear_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//button[@aria-label="Clear source text"]'))
            )
            if clear_button.is_displayed():
                clear_button.click()
                time.sleep(0.5)
            
            # Find the source textarea and input the text
            source_textarea = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//textarea[@aria-label="Source text"]'))
            )
            source_textarea.clear()
            source_textarea.send_keys(text)
            
            # Wait for translation to appear
            time.sleep(DELAY)
            
            # Get the translated text
            target_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'ryNqvb'))
            )
            
            return target_element.text
        except Exception as e:
            print(f"Translation error: {e}")
            # Refresh page if there was an error
            self.driver.get(self.url)
            time.sleep(2)
            return f"[TRANSLATION ERROR: {text}]"
            
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

def translate_dialog_file(translator, input_file, output_file):
    """
    Translate a dialog file from Hebrew to English.
    Preserves speaker labels and dialog structure.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If file is empty, write empty file and return
        if not content.strip():
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("")
            return True
        
        # Split into lines while preserving empty lines
        lines = content.split('\n')
        translated_lines = []
        
        for line in lines:
            if not line.strip():
                # Preserve empty lines
                translated_lines.append("")
                continue
            
            # Check if line starts with a speaker label (like "NURSE:" or "PATIENT:")
            speaker_match = re.match(r'^([A-Za-z]+):(.*)', line)
            
            if speaker_match:
                # Extract speaker and dialog
                speaker = speaker_match.group(1)
                dialog = speaker_match.group(2).strip()
                
                # Translate only the dialog part
                translated_dialog = translator.translate(dialog)
                
                # Reconstruct the line with original speaker label
                translated_lines.append(f"{speaker}: {translated_dialog}")
            else:
                # Translate the whole line if no speaker label is found
                translated_lines.append(translator.translate(line))
        
        # Join lines back into content
        translated_content = '\n'.join(translated_lines)
        
        # Write translated content to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        return True
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return False

def process_directory(input_dir, output_dir):
    """
    Process all dialog files in the input directory and its subdirectories.
    Maintains the same directory structure in the output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all session directories
    session_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    total_files = 0
    for session_dir in session_dirs:
        dialog_files = list(session_dir.glob('*_dialog.txt'))
        total_files += len(dialog_files)
    
    print(f"Found {total_files} dialog files in {len(session_dirs)} session directories")
    
    # Initialize translator
    translator = GoogleTranslateWeb(source_lang="iw", target_lang="en")
    
    try:
        # Process each session directory
        for session_dir in session_dirs:
            session_id = session_dir.name
            output_session_dir = output_path / session_id
            output_session_dir.mkdir(exist_ok=True)
            
            dialog_files = list(session_dir.glob('*_dialog.txt'))
            
            print(f"Processing session {session_id} with {len(dialog_files)} dialog files")
            
            for dialog_file in tqdm(dialog_files, desc=f"Session {session_id}"):
                file_name = dialog_file.name
                output_file = output_session_dir / file_name
                
                success = translate_dialog_file(translator, dialog_file, output_file)
                if not success:
                    print(f"Failed to translate {dialog_file}")
    finally:
        # Make sure to close the browser
        translator.close()

if __name__ == "__main__":
    print("Starting dialog translation process using Google Translate website...")
    process_directory(INPUT_DIR, OUTPUT_DIR)
    print("Translation completed!")
