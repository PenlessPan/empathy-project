import os
import json
from typing import Dict, List, Tuple
import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_CACHE_DIR"] = "./speechbrain_cache"
from huggingface_hub import login
login(token=HF_TOKEN)
class SpeechProcessor:
    def __init__(self, audio_dir: str, output_dir: str):
        """Initialize speech processing pipeline.
        
        Args:
            audio_dir: Directory containing WAV audio files
            output_dir: Directory to save processed outputs
        """
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.whisper_model = whisper.load_model("large", device=self.device)
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HF_TOKEN
        ).to(self.device)

    def process_session(self, session_id: str) -> Dict:
        """Process all audio files for a given session.
        
        Args:
            session_id: Session identifier (e.g., '101')
            
        Returns:
            Dictionary containing transcripts and speaker segments
        """
        session_results = {}
        angles = ['ceiling', 'face', 'top']
        
        for angle in angles:
            audio_path = os.path.join(self.audio_dir, session_id, f"{session_id}_{angle}.wav")
            if not os.path.exists(audio_path):
                continue
                
            # Get transcription with timestamps
            transcript = self.transcribe_audio(audio_path)
            
            # Get speaker diarization
            speakers = self.diarize_audio(audio_path)
            
            # Combine transcription with speaker information
            combined = self.align_transcript_speakers(transcript, speakers)
            
            # Save results
            output_path = os.path.join(self.output_dir, session_id)
            os.makedirs(output_path, exist_ok=True)
            
            with open(os.path.join(output_path, f"{angle}_processed.json"), 'w', encoding='utf-8') as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
                
            session_results[angle] = combined
            
        return session_results

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            Dictionary containing transcription segments with timestamps
        """
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(
            audio_path,
            language="he",
            task="transcribe"
        )

        return {
            'segments': [{
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            } for seg in result['segments']]
        }

    def diarize_audio(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            List of speaker segments with timestamps
        """
        diarization = self.diarization_pipeline(audio_path)
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            
        return speaker_segments

    def align_transcript_speakers(
        self, 
        transcript: Dict, 
        speaker_segments: List[Dict]
    ) -> Dict:
        """Align transcription segments with speaker information.
        
        Args:
            transcript: Whisper transcription output
            speaker_segments: Speaker diarization output
            
        Returns:
            Combined transcript with speaker labels
        """
        aligned_segments = []
        
        for trans_seg in transcript['segments']:
            # Find overlapping speaker segments
            matching_speakers = [
                spk for spk in speaker_segments
                if (spk['start'] < trans_seg['end'] and 
                    spk['end'] > trans_seg['start'])
            ]
            
            # Use the speaker with most overlap
            if matching_speakers:
                speaker = max(
                    matching_speakers,
                    key=lambda x: min(x['end'], trans_seg['end']) - 
                                max(x['start'], trans_seg['start'])
                )['speaker']
            else:
                speaker = "unknown"
                
            aligned_segments.append({
                'start': trans_seg['start'],
                'end': trans_seg['end'],
                'text': trans_seg['text'],
                'speaker': speaker
            })
            
        return {'segments': aligned_segments}

    def extract_student_speech(
        self, 
        processed_data: Dict,
        output_path: str
    ) -> None:
        """Extract audio segments containing only student speech.
        
        Args:
            processed_data: Aligned transcript and speaker data
            output_path: Path to save extracted audio
        """
        # Load full audio
        audio = AudioSegment.from_wav(output_path)
        
        # Extract and concatenate student segments
        student_segments = []
        for segment in processed_data['segments']:
            if segment['speaker'] == "SPEAKER_00":  # Adjust based on actual student speaker label
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                student_segments.append(audio[start_ms:end_ms])
                
        if student_segments:
            combined = sum(student_segments[1:], student_segments[0])
            combined.export(
                output_path.replace('.wav', '_student.wav'),
                format='wav'
            )

def main():
    """Main execution function."""
    audio_dir = "audio"
    output_dir = "speech"
    
    processor = SpeechProcessor(audio_dir, output_dir)
    
    # Process all sessions
    sessions = range(101, 141)
    for session_id in sessions:
        if session_id in {117, 118, 139}:  # Skip missing sessions
            continue
        try:
            results = processor.process_session(str(session_id))
            print(f"Successfully processed session {session_id}")
        except Exception as e:
            print(f"Error processing session {session_id}: {str(e)}")

if __name__ == "__main__":
    main()
