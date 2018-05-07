import os
import librosa

genre = ['pop', 'blues', 'metal', 'rock', 'hiphop']
rootPath = os.path.abspath('..')


class Data:

    def __init__(self, genre,
                 audioPath=rootPath + "/MIRdata/genres",
                 textPath=rootPath + "/MIRdata/gtzan_key-master/gtzan_key/genres", ):
        self.audioPath, self.textPath = audioPath, textPath
        self.genre = genre

        audio = sorted(os.listdir(os.path.join(audioPath, genre)))
        text = sorted(os.listdir(os.path.join(textPath, genre)))

        self.data = list(zip(audio, text))
        # print("list:\n" + str(self.data))
        self.len = len(self.data)

    def __getitem__(self, idx):
        audioFile, textFile = self.data[idx]
        audioFile = os.path.join(self.audioPath, self.genre, audioFile)
        textFile = os.path.join(self.textPath, self.genre, textFile)

        au, sr = librosa.load(audioFile)

        with open(textFile, mode='r') as f:
            key = f.readline()
            key = int(key)

        return au, sr, key


for x in genre:
    # print(x, "data info")
    data = Data(x)
