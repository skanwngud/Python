import speech_recognition as sr

r = sr.Recognizer()
havard = sr.AudioFile('c:/data/sushi2.wav')
with havard as source:
    audio = r.record(source)
print(r.recognize_google(audio, language='ko-KR'))