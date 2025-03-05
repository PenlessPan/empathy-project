import os
import pandas as pd
import numpy as np
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_empathy_judge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMEmpathyJudge:
    def __init__(self):
        """Initialize the ChatGPT API client for empathy judgment"""
        logger.info("Initializing ChatGPT API client")
        self.client = OpenAI(api_key=API_KEY)

    def get_empathy_score(self, dialog_text):
        """
        Use ChatGPT to judge nurse empathy on a 1-5 scale
        
        Args:
            dialog_text: A string containing the dialog transcript
        
        Returns:
            Integer score between 1-5 representing empathy level
        """
        if not dialog_text or len(dialog_text.strip()) < 10:
            logger.warning("Dialog text too short for evaluation")
            return None
        
        # Construct prompt
        prompt = f"""The following is a dialog transcript between a nurse and a patient. Rate the nurse's empathy on a scale from 1 to 5. Respond only with: \"The nurse's empathy score is: {{1-5}}\".
        
        Dialog:
        {dialog_text}
        """
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            result = response.choices[0].message.content
            
            # Extract the score using regex
            match = re.search(r"The nurse's empathy score is: (\d+)", result)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    logger.info(f"Extracted empathy score: {score}")
                    return score
            
            logger.warning(f"Could not extract empathy score from response: {result}")
            return None
        except Exception as e:
            logger.error(f"Error getting ChatGPT response: {str(e)}")
            return None

def batch_process_dialogs(dialog_dir="dialog_text", output_dir="LLM_empathy_judge"):
    """
    Process all dialog files and compute empathy scores using ChatGPT API
    
    Args:
        dialog_dir: Directory containing dialog text files
        output_dir: Directory to save results
    
    Returns:
        DataFrame with session IDs and empathy scores
    """
    logger.info(f"Starting batch processing of dialog files from {dialog_dir}")
    os.makedirs(output_dir, exist_ok=True)
    judge = LLMEmpathyJudge()
    
    dialog_files = [os.path.join(root, file) for root, _, files in os.walk(dialog_dir) for file in files if file.endswith("_dialog.txt")]
    logger.info(f"Found {len(dialog_files)} dialog files to process")
    
    results = []
    for file_path in dialog_files:
        session_id = os.path.basename(os.path.dirname(file_path))
        angle = os.path.basename(file_path).replace("_dialog.txt", "").split("_")[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dialog_text = f.read()
        
        empathy_score = judge.get_empathy_score(dialog_text)
        if empathy_score is not None:
            results.append({"session_id": session_id, "angle": angle, "empathy_score": empathy_score, "file_path": file_path})
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No valid dialog files found!")
        return pd.DataFrame()
    
    session_scores = results_df.groupby("session_id")["empathy_score"].mean().reset_index().sort_values("session_id")
    results_df.to_csv(os.path.join(output_dir, "llm_empathy_detailed.csv"), index=False)
    session_scores.to_csv(os.path.join(output_dir, "llm_empathy_by_session.csv"), index=False)
    
    return results_df, session_scores

if __name__ == "__main__":
    logger.info("Starting LLM empathy judgment experiment")
    detailed_results, session_scores = batch_process_dialogs()
    if not session_scores.empty:
        logger.info(f"Number of sessions analyzed: {len(session_scores)}")
        logger.info(f"Average empathy score: {session_scores['empathy_score'].mean():.2f}")
        logger.info("LLM empathy judgment experiment completed")
