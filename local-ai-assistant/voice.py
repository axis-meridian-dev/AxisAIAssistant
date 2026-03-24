"""
Voice Interface — wake word detection, STT (Whisper), TTS (Piper).

Supports three modes:
1. Always-listening with wake word ("computer")
2. Push-to-talk (hotkey)
3. Both simultaneously

Dependencies:
  pip install faster-whisper sounddevice numpy piper-tts openwakeword
"""

import asyncio
import queue
import threading
import tempfile
import wave
import os
import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd

from rich.console import Console

console = Console()


class VoiceInterface:
    def __init__(self, agent, config: dict):
        self.agent = agent
        self.config = config
        self.voice_cfg = config.get("voice", {})
        
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_listening = True
        
        # Lazy-load heavy models
        self._whisper_model = None
        self._piper_voice = None
        self._wake_word_model = None
    
    @property
    def whisper(self):
        if self._whisper_model is None:
            console.print("[dim]Loading Whisper model...[/dim]")
            from faster_whisper import WhisperModel
            model_size = self.voice_cfg.get("stt_model", "base.en")
            self._whisper_model = WhisperModel(
                model_size, device="cuda", compute_type="float16"
            )
            console.print("[green]Whisper loaded.[/green]")
        return self._whisper_model
    
    @property
    def wake_word(self):
        if self._wake_word_model is None:
            try:
                from openwakeword.model import Model
                self._wake_word_model = Model(
                    wakeword_models=["hey_computer"],
                    inference_framework="onnx"
                )
                console.print("[green]Wake word model loaded.[/green]")
            except Exception as e:
                console.print(f"[yellow]Wake word disabled: {e}[/yellow]")
                self._wake_word_model = False
        return self._wake_word_model
    
    def speak(self, text: str):
        """Convert text to speech using Piper TTS."""
        voice = self.voice_cfg.get("tts_voice", "en_US-lessac-medium")
        
        try:
            # Piper via command line (most reliable)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmpfile = f.name
            
            process = subprocess.Popen(
                ["piper", "--model", voice, "--output_file", tmpfile],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.communicate(text.encode())
            
            # Play the audio
            if os.path.exists(tmpfile):
                subprocess.run(
                    ["aplay", tmpfile],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                os.unlink(tmpfile)
        except FileNotFoundError:
            # Fallback to espeak
            try:
                subprocess.run(
                    ["espeak", "-s", "160", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                console.print(f"[yellow]TTS unavailable. Response: {text}[/yellow]")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        try:
            segments, info = self.whisper.transcribe(tmpfile, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        finally:
            os.unlink(tmpfile)
    
    def record_audio(self, duration: float = None, silence_timeout: float = 2.0) -> np.ndarray:
        """Record audio until silence is detected or duration reached."""
        console.print("[bold cyan]🎤 Listening...[/bold cyan]")
        
        frames = []
        silence_frames = 0
        silence_threshold = 0.01
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(self.sample_rate * chunk_duration)
        max_frames = int((duration or 30) / chunk_duration)
        silence_limit = int(silence_timeout / chunk_duration)
        started_speaking = False
        
        def callback(indata, frame_count, time_info, status):
            self.audio_queue.put(indata.copy())
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_size,
            callback=callback
        ):
            for _ in range(max_frames):
                try:
                    chunk = self.audio_queue.get(timeout=1)
                    frames.append(chunk)
                    
                    rms = np.sqrt(np.mean(chunk ** 2))
                    
                    if rms > silence_threshold:
                        started_speaking = True
                        silence_frames = 0
                    elif started_speaking:
                        silence_frames += 1
                        if silence_frames >= silence_limit:
                            break
                except queue.Empty:
                    break
        
        if frames:
            audio = np.concatenate(frames, axis=0).flatten()
            console.print(f"[dim]Recorded {len(audio)/self.sample_rate:.1f}s of audio[/dim]")
            return audio
        return np.array([])
    
    async def process_voice_input(self):
        """Record, transcribe, process, and speak response."""
        audio = self.record_audio()
        
        if len(audio) < self.sample_rate * 0.5:  # Less than 0.5s
            return
        
        # Transcribe
        console.print("[dim]Transcribing...[/dim]")
        text = self.transcribe(audio)
        
        if not text or len(text.strip()) < 2:
            return
        
        console.print(f"[bold green]You said:[/bold green] {text}")
        
        # Process through agent
        response = await self.agent.process(text)
        console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")
        
        # Speak response
        self.speak(response)
    
    async def wake_word_loop(self):
        """Listen for wake word, then process voice input."""
        ww = self.wake_word
        if ww is False:
            console.print("[yellow]Wake word disabled. Using push-to-talk only.[/yellow]")
            return
        
        console.print(f"[green]Listening for wake word: '{self.voice_cfg.get('wake_word', 'computer')}'[/green]")
        
        chunk_size = 1280  # ~80ms at 16kHz
        
        def callback(indata, frame_count, time_info, status):
            self.audio_queue.put(indata.copy())
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=chunk_size,
            callback=callback
        ):
            while self.is_listening:
                try:
                    chunk = self.audio_queue.get(timeout=1)
                    prediction = ww.predict(chunk.flatten())
                    
                    # Check if wake word detected
                    for mdl_name, score in prediction.items():
                        if score > 0.5:
                            console.print(f"[bold green]Wake word detected![/bold green]")
                            self.speak("Yes?")
                            await self.process_voice_input()
                            break
                except queue.Empty:
                    await asyncio.sleep(0.01)
    
    async def push_to_talk_loop(self):
        """Listen for hotkey, then process voice input."""
        from pynput import keyboard
        
        hotkey_str = self.voice_cfg.get("push_to_talk_key", "ctrl+space")
        console.print(f"[green]Push-to-talk: Hold {hotkey_str}[/green]")
        
        recording_event = asyncio.Event()
        
        def on_activate():
            recording_event.set()
        
        # Parse hotkey
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(f"<{hotkey_str.replace('+', '>+<')}>"),
            on_activate
        )
        
        def for_canonical(f):
            return lambda k: f(listener.canonical(k))
        
        listener = keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release)
        )
        listener.start()
        
        while self.is_listening:
            await recording_event.wait()
            recording_event.clear()
            await self.process_voice_input()
    
    async def run(self):
        """Run the voice interface with both wake word and push-to-talk."""
        console.print("\n[bold cyan]🎙️  Voice Mode Active[/bold cyan]")
        console.print("[dim]Say the wake word or use push-to-talk. Ctrl+C to exit.[/dim]\n")
        
        tasks = [
            asyncio.create_task(self.wake_word_loop()),
            asyncio.create_task(self.push_to_talk_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.is_listening = False
            console.print("\n[dim]Voice mode ended.[/dim]")
