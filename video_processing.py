import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import os

class VideoProcessor:
    def __init__(self):
        pass
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get basic video information"""
        cap = cv2.VideoCapture(str(video_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    
    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video"""
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        
        command = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            return audio_path
        except subprocess.CalledProcessError:
            # If ffmpeg fails, create empty audio file
            return self._create_empty_audio(audio_path)
    
    def extract_frames(self, video_path: Path, num_frames: int = 5) -> list:
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        if frame_count > 0:
            # Calculate frame indices
            frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
        
        cap.release()
        return frames
    
    def _create_empty_audio(self, audio_path: Path) -> Path:
        """Create empty audio file if extraction fails"""
        # Create 1 second of silence
        sample_rate = 16000
        duration = 1.0
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
        
        # Save as WAV (simplified)
        audio_path.write_bytes(samples.tobytes())
        return audio_path
    
