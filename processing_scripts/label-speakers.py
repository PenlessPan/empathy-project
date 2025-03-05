import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def identify_speakers(processed_data: Dict) -> Dict:
    """
    Identify which speaker is the nurse (student) and which is the patient
    based on question mark frequency.
    
    Args:
        processed_data: Dictionary with transcript and speaker information
        
    Returns:
        Updated dictionary with 'NURSE' and 'PATIENT' speaker roles
    """
    # Count question marks per speaker
    question_marks_count = defaultdict(int)

    for segment in processed_data['segments']:
        speaker = segment['speaker']
        text = segment['text']
        question_marks_count[speaker] += text.count('?')

    # If no question marks, use speaking time as fallback
    if sum(question_marks_count.values()) == 0:
        speaking_time = defaultdict(float)
        for segment in processed_data['segments']:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            speaking_time[speaker] += duration

        # The speaker with less speaking time is likely the nurse
        # (since patients typically talk more about their condition)
        speakers_by_time = sorted(speaking_time.items(), key=lambda x: x[1])
        if len(speakers_by_time) >= 2:
            nurse_speaker = speakers_by_time[0][0]
        else:
            nurse_speaker = speakers_by_time[0][0] if speakers_by_time else "unknown"
    else:
        # Identify speakers based on question mark frequency
        speakers_by_questions = sorted(question_marks_count.items(), key=lambda x: x[1], reverse=True)
        nurse_speaker = speakers_by_questions[0][0]

    # Update segments with new speaker roles
    new_segments = []
    for segment in processed_data['segments']:
        original_speaker = segment['speaker']
        if original_speaker == 'unknown':
            continue  # Skip unknown speakers

        if original_speaker == nurse_speaker:
            segment['speaker_role'] = "NURSE"
        else:
            segment['speaker_role'] = "PATIENT"

        new_segments.append(segment)

    processed_data['segments'] = new_segments
    return processed_data


def extract_student_speech(processed_data: Dict) -> dict[str, list[Any]]:
    """
    Extract only the segments spoken by the nurse (student).
    
    Args:
        processed_data: Processed data with labeled speakers
        
    Returns:
        List of segments spoken by the nurse
    """
    nurse_segments = []

    for segment in processed_data['segments']:
        if segment.get('speaker_role') == "NURSE":
            nurse_segments.append(segment)

    return {"segments": nurse_segments}


def process_session_files(input_dir: str, output_dir_all: str, output_dir_student: str) -> None:
    """
    Process all session files, identify speaker roles, and save to output directories.

    Args:
        input_dir: Directory containing processed transcripts
        output_dir_all: Directory to save labeled transcripts
        output_dir_student: Directory to save student speech segments
    """
    os.makedirs(output_dir_all, exist_ok=True)
    os.makedirs(output_dir_student, exist_ok=True)

    # Get all session directories
    session_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for session_id in session_dirs:
        session_path = os.path.join(input_dir, session_id)
        output_session_path_all = os.path.join(output_dir_all, session_id)
        output_session_path_student = os.path.join(output_dir_student, session_id)
        os.makedirs(output_session_path_all, exist_ok=True)
        os.makedirs(output_session_path_student, exist_ok=True)

        # Find all processed JSON files
        json_files = glob.glob(os.path.join(session_path, "*_processed.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)

                # Identify speakers
                labeled_data = identify_speakers(processed_data)

                # Extract student speech
                nurse_segments = extract_student_speech(labeled_data)

                # Prepare output data
                output_data = {
                    'full_transcript': labeled_data,
                    'nurse_segments': nurse_segments
                }

                # Get angle name from filename
                filename = os.path.basename(json_file)
                angle = filename.split('_')[0]  # e.g., "ceiling_processed.json" -> "ceiling"

                # Save full transcript to labeled_speech directory
                output_file_full_transcript = os.path.join(output_session_path_all, f"{angle}_labeled.json")
                with open(output_file_full_transcript, 'w', encoding='utf-8') as f:
                    json.dump(output_data['full_transcript'], f, ensure_ascii=False, indent=2)

                # Save nurse segments to student_speech directory
                output_file_nurse_segments = os.path.join(output_session_path_student, f"{angle}_nurse_segments.json")
                with open(output_file_nurse_segments, 'w', encoding='utf-8') as f:
                    json.dump(output_data['nurse_segments'], f, ensure_ascii=False, indent=2)

                print(f"Successfully labeled session {session_id}, angle {angle}")
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")


def main():
    """Main execution function."""
    input_dir = "speech"
    output_dir_all = "labeled_speech"
    output_dir_student = "student_speech"

    process_session_files(input_dir, output_dir_all, output_dir_student)
    print(f"Labeled data saved to {output_dir_all} and {output_dir_student}")


if __name__ == "__main__":
    main()
