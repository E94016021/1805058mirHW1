import numpy as np
import librosa.feature
import librosa
import time

from concurrent.futures import ProcessPoolExecutor
from dataLoader import Data
from ksTemplate import kstemplate


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
    # print("start analyse", genre)

    if question == 'q1':
        # Parallelization of the load file process
        with ProcessPoolExecutor(max_workers=4) as executor:
            for au, sr, key in executor.map(d.__getitem__, range(d.len)):
                key_pred = ks_match_key(au, sr, 100)
                if key == key_pred:
                    counter += 1
        acc = counter / d.len
        # print("\n", genre, "acc =", acc, "\n")
        accs.append(acc)
    elif question == 'q3':
        # Parallelization of the load file process
        with ProcessPoolExecutor(max_workers=4) as executor:
            for au, sr, key in executor.map(d.__getitem__, range(d.len)):
                key_pred = ks_match_key(au, sr, 100)
                counter = counter + q3_score(key, key_pred)
        acc = counter / d.len
        # print("\n", genre, "acc =", acc, "\n")
        accs.append(acc)
    else:
        print("q# error")


def run_q2(genre, gamma):
    d = Data(genre)
    counter = 0  # Parallelization of the load file process
    with ProcessPoolExecutor(max_workers=4) as executor:
        for au, sr, key in executor.map(d.__getitem__, range(d.len)):
            key_pred = ks_match_key(au, sr, gamma)
            if key == key_pred:
                counter += 1
    acc = counter / d.len
    accs.append(acc)


def ks_match_key(au, sr, gamma):
    """
    :param au: audio file
    :param sr: sample rate
    :param gamma: gamma parameter of nonlinear transform
    :return: key label

    """
    ##########################################################################
    # tonic = template.argmax()
    # correlation = np.array([R(template[k], tonic) for k in range(24)])
    # major = correlation[tonic]
    # minor = correlation[tonic + 12]
    #
    # if major > minor:
    #     return (correlation.argmax() + 3) % 12  # convert to gtzan key
    # else:
    #     return (correlation.argmax() + 3) % 12 + 12  # convert to gtzan key
    ##########################################################################

    # librosa get chroma with clp
    chroma = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=au, sr=sr)))

    # # normalize chroma
    # chroma = chroma / np.tile(np.sum(np.abs(chroma) ** 2, axis=0) ** (1. / 2),
    #                           (chroma.shape[0], 1))

    vector = np.sum(chroma, axis=1)
    ans = np.array([R(kstemplate[k], vector) for k in range(24)])

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

    # Relative major/minor
    if p < 12 <= a:
        a -= 12
        if p == (a + 3) % 12:
            new_accuracy += 0.3
    elif p >= 12 > a:
        p -= 12
        if (p + 3) % 12 == a:
            new_accuracy += 0.3

    # Parallel major/minor
    if p == (a + 12) % 24:
        new_accuracy += 0.2

    return new_accuracy


if __name__ == "__main__":
    ts = time.time()
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

    te = time.time()
    print("cost %  sec" % (te - ts))

"""
q1 
[('pop', 0.43), ('blues', 0.22), ('metal', 0.22), ('rock', 0.3), ('hiphop', 0.12)]
q2 & gamma = 1
[('pop', 0.45), ('blues', 0.27), ('metal', 0.25), ('rock', 0.37), ('hiphop', 0.11)]
q2 & gamma = 10
[('pop', 0.44), ('blues', 0.25), ('metal', 0.23), ('rock', 0.35), ('hiphop', 0.12)]
q2 & gamma = 100
[('pop', 0.43), ('blues', 0.22), ('metal', 0.22), ('rock', 0.3), ('hiphop', 0.12)]
q2 & gamma = 1000
[('pop', 0.43), ('blues', 0.21), ('metal', 0.2), ('rock', 0.29), ('hiphop', 0.12)]
q3 
[('pop', 0.498), ('blues', 0.248), ('metal', 0.273), ('rock', 0.40900000000000003), ('hiphop', 0.169)]

"""
