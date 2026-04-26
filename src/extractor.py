import os
import json
import warnings
import numpy as np
import librosa
from faster_whisper import WhisperModel
from pathlib import Path

# Suppress minor librosa/numba warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)

class SynestheticExtractor:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        print(f"[Extractor] Booting Faster-Whisper ({model_size} model) on {device.upper()}...")
        # int8 compute type runs blazingly fast on CPUs
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
    def extract_text(self, file_path):
        """Extracts text and word-level/segment-level timestamps."""
        print("[Extractor] Running linguistic analysis (Whisper)...")
        segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
        
        transcript_data = []
        for segment in segments:
            transcript_data.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            
        return transcript_data

    def extract_acoustic_features(self, file_path, linguistic_data):
        """Extracts global tempo (BPM) and localized acoustic features per text segment."""
        print("[Extractor] Running localized acoustic wave analysis (Librosa)...")
        y, sr = librosa.load(file_path, sr=22050)
        
        # 1. Global Tempo (BPM) - Calculated across the whole track
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        enriched_segments = []
        for segment in linguistic_data:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            y_slice = y[start_sample:end_sample]
            
            if len(y_slice) == 0:
                segment["rms_energy"] = 0.0
                segment["spectral_centroid"] = 0.0
                segment["zcr"] = 0.0
                segment["spectral_bandwidth"] = 0.0
                enriched_segments.append(segment)
                continue
                
            # 1. Localized Brightness
            spectral_centroids = librosa.feature.spectral_centroid(y=y_slice, sr=sr)[0]
            local_centroid = float(np.mean(spectral_centroids))
            
            # 2. Localized Intensity
            rms_energy = librosa.feature.rms(y=y_slice)[0]
            local_rms = float(np.mean(rms_energy))
            
            # 3. Localized Chaos / Percussiveness (NEW)
            zcr = librosa.feature.zero_crossing_rate(y=y_slice)[0]
            local_zcr = float(np.mean(zcr))
            
            # 4. Localized Scale / Thickness (NEW)
            bandwidth = librosa.feature.spectral_bandwidth(y=y_slice, sr=sr)[0]
            local_bandwidth = float(np.mean(bandwidth))
            
            # Inject all data into the payload
            segment["rms_energy"] = round(local_rms, 4)
            segment["spectral_centroid"] = round(local_centroid, 2)
            segment["zcr"] = round(local_zcr, 4)
            segment["spectral_bandwidth"] = round(local_bandwidth, 2)
            
            enriched_segments.append(segment)
            
        return round(bpm, 2), enriched_segments

    def process_file(self, input_filepath, output_dir):
        """Runs the full extraction pipeline and saves the synchronized JSON payload."""
        file_path = Path(input_filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find input file: {input_filepath}")
            
        print(f"\n--- Processing: {file_path.name} ---")
        
        # Run Whisper first to get the time boundaries
        linguistic_data = self.extract_text(str(file_path))
        
        # Pass the boundaries into Librosa to slice the audio and extract localized data
        global_bpm, synchronized_data = self.extract_acoustic_features(str(file_path), linguistic_data)
        
        # Assemble the final synesthetic payload
        payload = {
            "filename": file_path.name,
            "global_bpm": global_bpm,
            "timeline": synchronized_data
        }
        
        # Ensure the output directory exists
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        output_file = out_dir / f"{file_path.stem}_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
            
        print(f"[Extractor] Success! Data saved to: {output_file}\n")
        return output_file

# Quick test block: This only runs if you execute this file directly
if __name__ == "__main__":
    # Create the test folders if they don't exist
    Path("data/input").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Initialize the extractor
    extractor = SynestheticExtractor()
    
    # IMPORTANT: Replace 'your_song.mp3' with your actual downloaded file name!
    extractor.process_file("data/input/tlop.mp3", "data/processed")