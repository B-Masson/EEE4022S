# Trimming technique obtained from the following source:
# https://stackoverflow.com/a/29550200
from pydub import AudioSegment

def detectSilence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0
    assert chunk_size > 0 # Avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def trim(filename):
    sound = AudioSegment.from_file(filename, format="wav")
    start_trim = detectSilence(sound)
    end_trim = detectSilence(sound.reverse())
    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    trimmed_sound.export(filename, format="wav")