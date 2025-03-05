import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multilingual_hebrew_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultilingualEmpathyScorer:
    def __init__(self):
        """Initialize the multilingual model and tokenizer for empathy scoring"""
        logger.info("Initializing multilingual model and tokenizer")
        
        # Load XLM-RoBERTa model and tokenizer (more memory efficient than BLOOM)
        # This model performs well on Hebrew despite being multilingual
        self.model_name = "xlm-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {self.model_name}")
        self.model.to(self.device)
        
        # Define empathetic expressions in Hebrew (same as AlephBERT experiment)
        self.empathetic_phrases = [
            "אני מבין איך אתה מרגיש",  # I understand how you feel
            "אני מבינה איך את מרגישה",  # I understand how you feel (female)
            "זה נשמע קשה",  # That sounds difficult
            "אני מקשיב לך",  # I'm listening to you
            "אני מקשיבה לך",  # I'm listening to you (female)
            "אני כאן בשבילך",  # I'm here for you
            "אני מבין",  # I understand
            "אני מבינה",  # I understand (female)
            "תודה שסיפרת לי",  # Thank you for telling me
            "תודה ששיתפת אותי",  # Thank you for sharing with me
            "איך אתה מרגיש עם זה",  # How do you feel about it
            "איך את מרגישה עם זה",  # How do you feel about it (female)
            "אני יכול להבין למה אתה מרגיש כך",  # I can understand why you feel this way
            "אני יכולה להבין למה את מרגישה כך",  # I can understand why you feel this way (female)
            "זה נשמע מאוד מאתגר",  # That sounds very challenging
            "אני מצטער לשמוע",  # I'm sorry to hear that
            "אני מצטערת לשמוע",  # I'm sorry to hear that (female)
            "אני מעריך את זה שאתה משתף אותי",  # I appreciate you sharing with me
            "אני מעריכה את זה שאת משתפת אותי",  # I appreciate you sharing with me (female)
        ]
        
        # Cache embeddings of empathetic phrases for efficiency
        self.empathetic_embeddings = self._get_phrases_embeddings()
        
        logger.info("Multilingual empathy scorer initialized successfully")

    def _get_embedding(self, text):
        """
        Get embedding for a text string using multilingual model
        """
        # Prepare inputs for the model
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the CLS token embedding as the sentence representation
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _get_phrases_embeddings(self):
        """
        Compute embeddings for the list of empathetic phrases
        """
        embeddings = []
        for phrase in self.empathetic_phrases:
            embedding = self._get_embedding(phrase)
            embeddings.append(embedding.flatten())
        return embeddings

    def score_utterance_empathy(self, utterance):
        """
        Calculate empathy score for a single utterance
        
        Args:
            utterance: A string containing the dialog utterance
            
        Returns:
            Float value between 0 and 1 representing empathy score
        """
        if not utterance or len(utterance.strip()) < 2:
            return 0.0
        
        # Get embedding for utterance
        utterance_embedding = self._get_embedding(utterance).flatten()
        
        # Calculate similarities with empathetic phrases
        similarities = []
        for emb in self.empathetic_embeddings:
            sim = cosine_similarity([utterance_embedding], [emb])[0][0]
            similarities.append(sim)
        
        # Base score: average of top 3 similarities
        similarities.sort(reverse=True)
        base_score = np.mean(similarities[:3]) if similarities else 0.0
        
        # Additional features that indicate empathy
        
        # 1. Question mark presence (asking about feelings)
        has_question = 0.1 if '?' in utterance else 0.0
        
        # 2. Common empathetic words in Hebrew
        empathy_words = ['מבין', 'מרגיש', 'מקשיב', 'עוזר', 'תומך', 'מצטער', 'דואג']
        word_matches = sum(1 for word in empathy_words if re.search(fr'\b{word}\b', utterance)) / len(empathy_words)
        word_score = 0.2 * word_matches
        
        # 3. Length penalty (very short responses are less likely to be empathetic)
        length_factor = min(1.0, len(utterance.split()) / 10)
        
        # Combined score with weights
        empathy_score = (0.6 * base_score) + (0.2 * has_question) + (0.2 * word_score)
        
        # Apply length factor
        empathy_score *= length_factor
        
        # Ensure range is between 0 and 1
        return min(1.0, max(0.0, empathy_score))

    def analyze_dialog_file(self, file_path):
        """
        Analyze a dialog file and compute average empathy score for nurse utterances
        
        Args:
            file_path: Path to dialog transcript file
            
        Returns:
            Average empathy score for the session (float between 0 and 1)
        """
        logger.info(f"Analyzing dialog file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dialog_text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
        
        # Split dialog into utterances
        lines = dialog_text.strip().split('\n')
        nurse_utterances = [line.replace("NURSE:", "").strip() for line in lines if line.startswith("NURSE:")]
        
        # Skip files with no nurse utterances
        if not nurse_utterances:
            logger.warning(f"No nurse utterances found in {file_path}")
            return None
        
        # Score each nurse utterance
        empathy_scores = []
        for utterance in nurse_utterances:
            if utterance.strip():
                score = self.score_utterance_empathy(utterance)
                empathy_scores.append(score)
                logger.debug(f"Utterance: '{utterance}' - Score: {score:.4f}")
        
        # Return average empathy score for the session
        avg_score = np.mean(empathy_scores) if empathy_scores else 0.0
        logger.info(f"Average empathy score for {file_path}: {avg_score:.4f}")
        return avg_score

def batch_process_dialogs(dialog_dir="dialog_text", output_dir="multilingual_hebrew_exp"):
    """
    Process all dialog files and compute empathy scores using multilingual model
    
    Args:
        dialog_dir: Directory containing Hebrew dialog text files
        output_dir: Directory to save results
        
    Returns:
        DataFrame with session IDs and empathy scores
    """
    logger.info(f"Starting batch processing of dialog files from {dialog_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize multilingual scorer
    scorer = MultilingualEmpathyScorer()
    
    # Get all dialog files
    dialog_files = []
    for root, _, files in os.walk(dialog_dir):
        for file in files:
            if file.endswith("_dialog.txt"):
                dialog_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(dialog_files)} dialog files to process")
    
    # Process each file
    results = []
    for file_path in dialog_files:
        # Extract session ID from directory name
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        # Try to extract session ID from directory name
        if dir_name.isdigit():
            session_id = dir_name
            
            # Extract angle from filename
            filename = os.path.basename(file_path)
            parts = filename.replace("_dialog.txt", "").split("_")
            angle = parts[0] if len(parts) >= 1 else "unknown"
            
            logger.info(f"Processing session {session_id}, angle {angle}...")
            empathy_score = scorer.analyze_dialog_file(file_path)
            
            if empathy_score is not None:
                results.append({
                    "session_id": session_id,
                    "angle": angle,
                    "empathy_score": empathy_score,
                    "file_path": file_path
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Handle empty results
    if results_df.empty:
        logger.warning("No valid dialog files found after processing!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate average scores per session (across angles)
    session_scores = results_df.groupby("session_id")["empathy_score"].mean().reset_index()
    
    # Normalize scores to 0-100 range
    min_score = session_scores["empathy_score"].min()
    max_score = session_scores["empathy_score"].max()
    
    # Avoid division by zero if all scores are the same
    if max_score == min_score:
        session_scores["normalized_score"] = 50.0  # Assign middle value
    else:
        session_scores["normalized_score"] = ((session_scores["empathy_score"] - min_score) / 
                                              (max_score - min_score) * 100)
    
    # Convert to 1-5 scale
    session_scores["empathy_scale_1_5"] = (session_scores["normalized_score"] / 100 * 4) + 1
    
    # Sort by session ID
    session_scores = session_scores.sort_values("session_id")
    
    # Save results to CSV
    detailed_csv_path = os.path.join(output_dir, "multilingual_hebrew_empathy_detailed.csv")
    session_csv_path = os.path.join(output_dir, "multilingual_hebrew_empathy_by_session.csv")
    
    results_df.to_csv(detailed_csv_path, index=False)
    session_scores.to_csv(session_csv_path, index=False)
    
    logger.info(f"Saved detailed results to {detailed_csv_path}")
    logger.info(f"Saved session scores to {session_csv_path}")
    
    return results_df, session_scores

if __name__ == "__main__":
    dialog_dir = "dialog_text"
    output_dir = "multilingual_hebrew_exp"
    
    logger.info("Starting multilingual Hebrew empathy analysis experiment")
    detailed_results, session_scores = batch_process_dialogs(dialog_dir, output_dir)
    
    # Display summary of results
    if not session_scores.empty:
        logger.info("\nSummary of empathy scores:")
        logger.info(f"Number of sessions analyzed: {len(session_scores)}")
        logger.info(f"Average empathy score: {session_scores['empathy_score'].mean():.4f}")
        logger.info(f"Min empathy score: {session_scores['empathy_score'].min():.4f}")
        logger.info(f"Max empathy score: {session_scores['empathy_score'].max():.4f}")
        
        # Display top 5 and bottom 5 sessions
        top5 = session_scores.nlargest(5, 'empathy_score')
        bottom5 = session_scores.nsmallest(5, 'empathy_score')
        
        logger.info("\nTop 5 most empathetic sessions:")
        for _, row in top5.iterrows():
            logger.info(f"Session {row['session_id']}: {row['empathy_score']:.4f} (Scale 1-5: {row['empathy_scale_1_5']:.2f})")
        
        logger.info("\nBottom 5 least empathetic sessions:")
        for _, row in bottom5.iterrows():
            logger.info(f"Session {row['session_id']}: {row['empathy_score']:.4f} (Scale 1-5: {row['empathy_scale_1_5']:.2f})")
    
    logger.info("Multilingual Hebrew empathy analysis experiment completed")