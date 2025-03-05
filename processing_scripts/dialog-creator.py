import os
import json
import glob
from typing import Dict, List

def create_dialog_from_segments(segments: List[Dict]) -> str:
    """
    Create a formatted dialog from transcript segments.
    
    Args:
        segments: List of transcript segments with speaker_role and text
        
    Returns:
        Formatted dialog as a string
    """
    dialog_lines = []
    
    for segment in segments:
        if 'speaker_role' in segment and 'text' in segment:
            speaker = segment['speaker_role']
            text = segment['text'].strip()
            
            # Skip empty text
            if not text:
                continue
                
            dialog_line = f"{speaker}: {text}"
            dialog_lines.append(dialog_line)
    
    return "\n".join(dialog_lines)

def process_session_files(input_dir: str, output_dir: str) -> None:
    """
    Process all labeled transcript files and create dialog text files.
    
    Args:
        input_dir: Directory containing labeled transcripts
        output_dir: Directory to save dialog text files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all session directories
    session_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for session_id in session_dirs:
        session_path = os.path.join(input_dir, session_id)
        output_session_path = os.path.join(output_dir, session_id)
        os.makedirs(output_session_path, exist_ok=True)
        
        # Find all labeled JSON files
        json_files = glob.glob(os.path.join(session_path, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    labeled_data = json.load(f)
                
                # Extract segments from full transcript
                segments = labeled_data['segments']
                dialog_text = create_dialog_from_segments(segments)

                # Get angle name from filename
                filename = os.path.basename(json_file)
                angle = filename.split('_')[0]  # e.g., "ceiling_labeled.json" -> "ceiling"

                # Save dialog text
                output_file = os.path.join(output_session_path, f"{angle}_dialog.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(dialog_text)

                print(f"Successfully created dialog for session {session_id}, angle {angle}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

def main():
    """Main execution function."""
    input_dir = "labeled_speech"
    output_dir = "dialog_text"
    
    process_session_files(input_dir, output_dir)
    print(f"Dialog text files saved to {output_dir}")

    input_dir = "student_speech"
    output_dir = "student_dialog_text"

    process_session_files(input_dir, output_dir)
    print(f"Student dialog text files saved to {output_dir}")

if __name__ == "__main__":
    main()
