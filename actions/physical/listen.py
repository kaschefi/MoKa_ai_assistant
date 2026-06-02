import asyncio
import re
import time
import numpy as np
import torch
from faster_whisper import WhisperModel
import sounddevice as sd

WAKE_WORD = "hey buddy"

def extract_seconds(text):
    """Finds numbers and units in a sentence (e.g., '5 minutes')"""
    match = re.search(r'(\d+)\s*(hour|minute|second)', text.lower())
    if not match: return None

    number = int(match.group(1))
    unit = match.group(2)

    if "hour" in unit: return number * 3600
    if "minute" in unit: return number * 60
    return number

class VoiceListener:
    def __init__(self, model_size="small.en", device="cpu", compute_type="int8"):
        print("[VoiceListener] Loading Whisper model (faster-whisper)...")
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        print("[VoiceListener] Loading Silero VAD model...")
        # Load silero-vad for voice activity detection
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        # Unpack utilities provided by Silero VAD
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
         
        self.sample_rate = 16000
        # Silero VAD optimally takes chunks of 512 for 16kHz
        self.chunk_size = 512
        
    async def listen_and_transcribe(self) -> str:
        loop = asyncio.get_running_loop()
        # Queue to pass audio from the PortAudio C-thread to the Python async loop
        q = asyncio.Queue()
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio Status] {status}")
            # Safely put the bytes array into the asyncio Queue from this separate thread
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        accumulated_audio = []
        is_speaking = False
        silence_start_time = None
        
        # Configuration for VAD thresholds
        VAD_THRESHOLD = 0.5
        SILENCE_DURATION = 1.0  # seconds of silence before we assume speech ended

        # Start audio acquisition pipeline natively (zero disk I/O)
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.chunk_size, 
                               dtype='int16', channels=1, callback=callback):
            while True:
                # Wait for next chunk from the microphone callback
                chunk = await q.get()
                
                # Convert PCM16 bytes to numpy float32 for Silero VAD
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_data)

                # Process chunk through the VAD
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                
                if speech_prob >= VAD_THRESHOLD:
                    if not is_speaking:
                        # Clear old audio or reset state if just starting to speak
                        self.vad_model.reset_states()
                    is_speaking = True
                    silence_start_time = None
                    accumulated_audio.append(audio_data)
                elif is_speaking:
                    accumulated_audio.append(audio_data)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        # Silence threshold reached, finalize the chunk
                        break

        if not accumulated_audio:
            return ""

        # Concatenate buffered arrays into one continuous numpy array
        full_audio = np.concatenate(accumulated_audio)
        
        # Offload transcription inference to a background thread to prevent blocking
        text = await loop.run_in_executor(None, self._transcribe_audio, full_audio)
        
        text = text.lower().strip()
        print(f" [Voice Listener] Heard: \"{text}\"")
        
        # Look for the wake word
        if WAKE_WORD in text:
            command = text.split(WAKE_WORD, 1)[1].strip()
            return command
        return ""
        
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        # Transcribe the floating point array natively (zero disk I/O)
        segments, _ = self.whisper_model.transcribe(audio_data, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        return text

async def start_listening_task(face_library=None):
    """
    Core voice listening loop wrapping the asynchronous VoiceListener.
    Replaces the older blocking thread-based start_listening_loop.
    """
    # Local import to prevent circular dependencies
    from core.routing.brain import process_user_intent
    
    print("[Voice Listener] Initializing Zero-I/O Speech Pipeline...")
    try:
        listener = VoiceListener(device="cpu", compute_type="int8")
    except Exception as e:
        print(f"[Voice Listener] [Error] Could not initialize models: {e}")
        return

    print("[Voice Listener] Ready! Say 'Hey Buddy' followed by your command.")
    
    while True:
        try:
            # Wait for a clean wake word + command combination asynchronously
            command = await listener.listen_and_transcribe()
            
            if command:
                print(f" [Voice Listener] Triggering Command: \"{command}\"")

                # Generate a unique session thread ID
                voice_session_id = f"voice_{int(time.time())}"

                # Suspend microphone actively listening to wait for LLM/tts to finish
                print(" [Voice Listener] Suspended microphone listening during processing...")
                await process_user_intent(command, session_id=voice_session_id)
                print("[Voice Listener] Processing complete. Resuming microphone listening...")

        except Exception as e:
            print(f"[Voice Listener] Error in listen loop: {e}")
            await asyncio.sleep(1)
            continue

def start_listening():
    """
    Standalone runner for local/terminal execution of voice loop.
    Creates a new event loop and runs the listener.
    """
    asyncio.run(start_listening_task())

if __name__ == "__main__":
    print(start_listening())