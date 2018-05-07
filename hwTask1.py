import numpy as np
import librosa.feature
import librosa

from concurrent.futures import ProcessPoolExecutor
from dataLoader import Data
from template import template


def R(x: np.ndarray, y: np.ndarray):
    """
    :param x:
    :param y:
    :return:

    """
    a = sum([(x[k] - x.mean()) * (y[k] - y.mean()) for k in range(12)])
    b1 = sum([(x[k] - x.mean()) ** 2 for k in range(12)])
    b2 = sum([(y[k] - y.mean()) ** 2 for k in range(12)])
    b = (b1 * b2) ** 0.5

    return a / b


def run(genre, question):
    d = Data(genre)
    counter = 0

    if question == 'q1':
        # Parallelization of the load file process
        with ProcessPoolExecutor(max_workers=4) as executor:
            i = 0
            f = open(question + genre + '.txt', mode='w')
            for au, sr, key in executor.map(d.__getitem__, range(d.len)):
                key_pred = match_key(au, sr, 100)
                print("%s   %s" % (str(d.data[i][0].strip(".au")), key_pred), file=f)
                i += 1
                if key == key_pred:
                    counter += 1
        acc = counter / d.len
        accs.append(acc)
        f.close()

    elif question == 'q3':
        # Parallelization of the load file process
        with ProcessPoolExecutor(max_workers=4) as executor:
            i = 0
            f = open(question + genre + '.txt', mode='w')
            for au, sr, key in executor.map(d.__getitem__, range(d.len)):
                key_pred = match_key(au, sr, gamma=100)
                print("%s   %s" % (str(d.data[i][0].strip(".au")), key_pred), file=f)
                i += 1
                counter = counter + q3_score(key, key_pred)
        acc = counter / d.len
        accs.append(acc)
        f.close()

    else:
        print("q# error")


def run_q2(genre, gamma):
    d = Data(genre)
    counter = 0  # Parallelization of the load file process
    with ProcessPoolExecutor(max_workers=4) as executor:
        i = 0
        f = open('q2_gamma' + str(gamma) + genre + '.txt', mode='w')
        for au, sr, key in executor.map(d.__getitem__, range(d.len)):
            key_pred = match_key(au, sr, gamma)
            print("%s   %s" % (str(d.data[i][0].strip(".au")), key_pred), file=f)
            i += 1
            if key == key_pred:
                counter += 1
    acc = counter / d.len
    accs.append(acc)
    f.close()


def match_key(au, sr, gamma):
    """
    :param au: audio file
    :param sr: sample rate
    :param gamma: gamma parameter of nonlinear transform
    :return: key label

    """

    # librosa get chroma with clp
    chroma = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=au, sr=sr)))

    vector = np.sum(chroma, axis=1)
    ans = np.array([R(template[k], vector) for k in range(24)])

    return (ans.argmax() + 3) % 24


def q3_score(ans, preds):
    """

    :param ans:
    :param preds:
    :return:
    """
    new_accuracy = 0

    a, p = ans, preds
    if p == a:
        new_accuracy += 1.
    if p < 12 and a < 12:
        if p == (a + 7) % 12:
            new_accuracy += 0.5
    elif p >= 12 and a >= 12:
        p -= 12
        a -= 12
        if p == (a + 7) % 12:
            new_accuracy += 0.5

    if p < 12 <= a:
        a -= 12
        if p == (a + 3) % 12:
            new_accuracy += 0.3
    elif p >= 12 > a:
        p -= 12
        if (p + 3) % 12 == a:
            new_accuracy += 0.3

    if p == (a + 12) % 24:
        new_accuracy += 0.2

    return new_accuracy


if __name__ == "__main__":
    print("start")
    genres = ['pop', 'blues', 'metal', 'rock', 'hiphop']
    questions = ['q1', 'q2', 'q3']
    gammas = [1, 10, 100, 1000]

    accs = []

    for genre in genres:
        run(genre, 'q1')
        mix = list(zip(genres, accs))
    print("q1 ")
    print(mix)

    for gamma in gammas:
        accs = []
        for genre in genres:
            run_q2(genre, gamma)
            mix = list(zip(genres, accs))
        print("q2 & gamma =", gamma)
        print(mix)

    accs = []
    for genre in genres:
        run(genre, 'q3')
        mix = list(zip(genres, accs))
    print("q3 ")
    print(mix)

    print("finished")

"""
start
q1 
[('pop', 0.16), ('blues', 0.07), ('metal', 0.06), ('rock', 0.19), ('hiphop', 0.01)]
q2 & gamma = 1
[('pop', 0.22), ('blues', 0.05), ('metal', 0.07), ('rock', 0.27), ('hiphop', 0.04)]
q2 & gamma = 10
[('pop', 0.2), ('blues', 0.07), ('metal', 0.06), ('rock', 0.22), ('hiphop', 0.03)]
q2 & gamma = 100
[('pop', 0.16), ('blues', 0.07), ('metal', 0.06), ('rock', 0.19), ('hiphop', 0.01)]
q2 & gamma = 1000
[('pop', 0.16), ('blues', 0.07), ('metal', 0.05), ('rock', 0.16), ('hiphop', 0.01)]
q3 
[('pop', 0.29900000000000004), ('blues', 0.129), ('metal', 0.12000000000000004), ('rock', 0.32600000000000007), ('hiphop', 0.06799999999999999)]

Process finished with exit code 0

"""
