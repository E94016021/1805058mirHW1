import librosa
import os
from dataLoader import Data

rootPath = os.path.abspath('..')
audioPath = rootPath + "/MIRdata/genres"
textPath = rootPath + "/MIRdata/gtzan_key-master/gtzan_key/genres"

genre = ['pop', 'blues', 'metal', 'rock', 'hiphop']

# print(os.path.abspath('.'))
# print(os.path.abspath('..'))
# print(audioPath)
# print(textPath)

# with open('file.txt', 'w') as f:
#     print('hello world', file=f)


f = open(genre[0]+'.txt', 'w')
print('fuck u u u u u ', file=f)
f.close()
print("gg")

# d = Data('pop')
# z = d.data[0][0]
# print(str(z).strip(".au"))
# z1 = d.data[1][0]
# print("%s   %s" % (str(z), audioPath))
# print(z1)
# k = d[0]
# print(k)
# # k = d.__getitem__()
# print(str(os.path.splitext(d.__getitem__(0))))

# for x in genre:
#     audio = os.listdir(os.path.join(audioPath, x))
#     text = os.listdir(os.path.join(textPath, x))
#     audio = sorted(audio)
#     print(audio)
