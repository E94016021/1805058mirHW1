import librosa
import os

rootPath =os.path.abspath('..')
audioPath = rootPath+"/MIRdata/genres"
textPath = rootPath+"/MIRdata/gtzan_key-master/gtzan_key/genres"

genre = ['pop','blues','metal','rock','hiphop']

# print(os.path.abspath('.'))
# print(os.path.abspath('..'))
# print(audioPath)
# print(textPath)

for x in genre:
    audio = os.listdir(os.path.join(audioPath, x))
    text = os.listdir(os.path.join(textPath, x))
    audio = sorted(audio)
    print(audio)

