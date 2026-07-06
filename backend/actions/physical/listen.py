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
        
    async def listen_and_transcribe(self, require_wake: bool = True, timeout: float = None) -> str:
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
        speech_start_time = None
        paused_speaker = False
        start_time = time.time()
        
        SILENCE_DURATION = 1.0  # seconds of silence before we assume speech ended

        # Local import to prevent circular dependencies
        from actions.physical.speak import speaker

        # Start audio acquisition pipeline natively (zero disk I/O)
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.chunk_size, 
                               dtype='int16', channels=1, callback=callback):
            while True:
                try:
                    # Timeout read to handle expectant window expiration and audio loop activity
                    chunk = await asyncio.wait_for(q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    current_time = time.time()
                    if speaker.is_playing:
                        start_time = current_time  # Reset timeout while Cozmo is speaking
                    elif not is_speaking and timeout and current_time - start_time > timeout:
                        return None
                    continue

                current_time = time.time()
                if speaker.is_playing:
                    start_time = current_time  # Reset timeout while Cozmo is speaking

                # Convert PCM16 bytes to numpy float32 for Silero VAD
                audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_data)

                # Dynamically scale VAD threshold based on speaker playing state
                vad_threshold = 0.85 if speaker.is_playing else 0.5
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                
                if speech_prob >= vad_threshold:
                    if not is_speaking:
                        # Clear old audio or reset state if just starting to speak
                        self.vad_model.reset_states()
                        is_speaking = True
                        speech_start_time = current_time
                    silence_start_time = None
                    accumulated_audio.append(audio_data)

                    # Live Reflex Interruption
                    if is_speaking and speaker.is_playing and not paused_speaker:
                        if current_time - speech_start_time > 0.25:
                            print(f"[Reflex] VAD crossed threshold for 250ms, pausing Cozmo...")
                            speaker.pause()
                            paused_speaker = True

                elif is_speaking:
                    accumulated_audio.append(audio_data)
                    if silence_start_time is None:
                        silence_start_time = current_time
                    elif current_time - silence_start_time > SILENCE_DURATION:
                        # Silence threshold reached, finalize the chunk
                        break
                else:
                    if timeout and current_time - start_time > timeout:
                        return None

        if not accumulated_audio:
            if paused_speaker:
                speaker.resume()
            return ""

        # Concatenate buffered arrays into one continuous numpy array
        full_audio = np.concatenate(accumulated_audio)
        
        # Offload transcription inference to a background thread to prevent blocking
        text = await loop.run_in_executor(None, self._transcribe_audio, full_audio)
        text_clean = text.strip()

        # Logical Evaluation Routing (Filler Filtering)
        FILLER_WORDS = {"um", "uh", "eh", "uum", "uhm", "ah"}

        if paused_speaker:
            text_lower_clean = text_clean.lower().strip(".,!?")
            if not text_lower_clean or text_lower_clean in FILLER_WORDS:
                print(f"[Filter] Detected filler sound ('{text_clean}'). Resuming Cozmo.")
                speaker.resume()
                return ""
            else:
                print("[Interruption] Valid command detected. Flushing Cozmo's queue.")
                speaker.interrupt()

        if text_clean:
            print(f" [Voice Listener] Heard Raw: \"{text_clean}\"")
        
        if not require_wake:
            return text_clean

        text_lower = text_clean.lower()
        # Look for the wake word
        if WAKE_WORD in text_lower:
            command = text_lower.split(WAKE_WORD, 1)[1].strip()
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
    
    require_wake = True
    
    while True:
        try:
            timeout = 4.0 if not require_wake else None
            if not require_wake:
                print("[Expectant Window] Listening for reply...")
            
            # Wait for a clean wake word + command combination asynchronously
            command = await listener.listen_and_transcribe(require_wake=require_wake, timeout=timeout)
            
            if command is None:
                print("[Timeout] Expectant window expired. Wake word required again.")
                require_wake = True
                continue
            
            if command:
                print(f" [Voice Listener] Triggering Command: \"{command}\"")

                # Generate a unique session thread ID
                voice_session_id = f"voice_{int(time.time())}"

                # Suspend microphone actively listening to wait for LLM/tts to finish
                print(" [Voice Listener] Suspended microphone listening during processing...")
                await process_user_intent(command, session_id=voice_session_id)
                print("[Voice Listener] Processing complete. Resuming microphone listening...")
                
                # After successful response, enter expectant window mode
                require_wake = False
            else:
                # noise or filler sounds, just continue listening
                pass

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