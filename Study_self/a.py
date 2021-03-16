import speech_recognition as sr

r = sr.Recognizer()
havard = sr.AudioFile('c:/data/sushi2.wav')
with havard as source:
    audio = r.record(source)
print(r.recognize_google(audio, language='ko-KR'))

# a = 472
# b = str(385)
# b0 = int(b[0])
# b1 = int(b[1])
# b2 = int(b[2])
# b3 = int(b)

# print(a*b0, a*b1, a*b2, a*b3, sep='\n')