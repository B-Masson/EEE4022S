#VOCARM Project
#Audio recording file - LINUX EDITION
#Richard Masson
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

samplerate = 44100  # Hertz
duration = 1 # Recording length

def record(filename):
    print("* recording")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=2, blocking=True)
    print("************************")
    print("* done recording")
    sf.write(filename, mydata, samplerate)

def detectSilence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0
    assert chunk_size > 0 # Avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    if (trim_ms == duration*1000): # If no sound (reaches end)
        print("NO SOUND DETECTED.")
        trim_ms = 0 # Keep the whole sample (it can be properly identified as silence if that is trained in)
    else:
        print("Sound detected at:", trim_ms, "ms")
    return max(trim_ms-10,0)

def trim(filename):
    sound = AudioSegment.from_file(filename, format="wav")
    start_trim = detectSilence(sound)
    trimmed_sound = sound[start_trim:]
    trimmed_sound.export(filename, format="wav")