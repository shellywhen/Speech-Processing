#        !/usr/bin/env python 2.7
from watson_developer_cloud import SpeechToTextV1
from os.path import join, dirname, realpath, exists
import sys
import json
import numpy as np
import functools
import parselmouth as pm
from re import sub
try:
    from nltk.corpus import cmudict
except:
    import nltk
    nltk.download('cmudict')
    from nltk.corpus import cmudict


filler_dict = [
    'uh', 'um', 'like', 'basically', 'well', 'er', 'hmm', 'actually', 'very', 'seriously',
    'that', 'just', 'only', 'really', 'slightly', 'almost', 'seemed', 'perhaps', 'maybe',
    'simply', 'somehow', 'absolutely', 'now',  'okay', 'so', 'right', 'mhm',
    'totally', 'literally', 'clearly'
]
filler_phrase = [
    'sort of', 'kind of', 'a little',  'uh huh', 'or something',
    'you see', 'you know', 'i mean', 'believe me', 'i guess', 'i suppose'
]


def syllables(word):  # referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def count_syllable(word):  # return syllable counts of a given word
    d = cmudict.dict()
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:  # if word not found in cmudict
        return syllables(word)


def get_path(url):  # return the absolute path of the relative path
    return join(dirname(realpath(__file__)),url)


def audio2text(config,url):  # return the response from IBM for speech recognition
    speech_to_text = SpeechToTextV1(
            username=config['username'],
            password=config['password'],
            url='https://stream.watsonplatform.net/speech-to-text/api')
    #        ibm.com/watson/developercloud/speech-to-text/api/v1/python.html?python#recognize-sessionless
    with open(get_path(url), 'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,  # file
            content_type='audio/wav',  # specify audio type
            model='en-US_BroadbandModel',  # speech recognition model
            smart_formatting=False,  # identify proper noun
            timestamps=True,  # return timestamps of each word
            max_alternatives=1)  # number of guessed word
        return speech_recognition_results


def data_processing(raw_result):  # parse the result from IBM Watson Cloud
    text = []
    results = raw_result['results']
    for res in results:
        alternatives = res['alternatives']
        alt = alternatives[0]
        word_cnt = len(alt['timestamps'])
        sentence = alt['transcript'].capitalize()
        start_time = alt['timestamps'][0][1]
        end_time = alt['timestamps'][word_cnt-1][1]
        text.append({'line': sentence, 'start': start_time, 'end': end_time})
    return text


def draw_amplitude(sound):
    sns.set()
    snd = sound
    plt.figure()
    plt.plot(snd.xs(),snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")


def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


class Audio(object):

    def __init__(self, config, url):
        self.config = config
        self.audio_path = get_path(url)
        self.audio_name = url[url.find('/')+1:url.find('.')]
        self.transcript = data_processing(audio2text(config, url))
        self.segment_len = len(self.transcript)

    def segment(self):
        audio = pm.Sound(self.audio_path)
        for index, sentence in enumerate(self.transcript):
            #       speaking rate (syllables/second)
            l_line = sentence['line'].lower()
            line = l_line.split()
            syllable_count = reduce(lambda x, y: x + y, map(count_syllable, line)) if sys.version_info[0] < 3 \
                else functools.reduce(lambda x, y: x + y, map(count_syllable, line))
            time_delta = sentence['end'] - sentence['start']
            if time_delta == 0:
                time_delta = 0.01
            sentence['speaking_rate'] = syllable_count / time_delta  # eliminate the error of dividing 0
            #       filler rate  (filler words/ last time)
            filler_count = 0
            for word in line:
                if word in filler_dict:
                    filler_count += 1
            for word in filler_phrase:
                filler_count += l_line.count(word)
            sentence['filler_rate'] = filler_count/time_delta
            sentence['filler_count'] = filler_count

            #       pitch variety ( the difference value between 95 percentile of pitch and that of 5% percentile)
            tmp_segment = audio.extract_part(from_time=sentence['start'], to_time=sentence['end'])
            tmp_pitch = tmp_segment.to_pitch().selected_array['frequency']
            tmp_pitch[tmp_pitch == 0] = np.nan
            tmp_upper_bound = np.nanpercentile(a=tmp_pitch, q=95)
            tmp_lower_bound = np.nanpercentile(a=tmp_pitch, q=5)
            sentence['pitch_variety'] = tmp_upper_bound - tmp_lower_bound
            #       make comments
            sentence['comment'] = Comment(sentence).comment
        return self

    def output(self):  # output the transcript as json file
        print("saved as "+self.audio_name+'_transcript.json')
        with open(self.audio_name+'_transcript.json', 'w') as f:
            json.dump(self.transcript, f)
        return self

    def visualize(self, PITCH=False, AMPLITUDE=False, INTENSITY=False, SPECTROGRAM=False,from_time =0,to_time=0):
        #       this part is too flexible to wrap as a function
        #       this part is in case of further demand
        import matplotlib.pyplot as plt
        import seaborn as sns
        snd = pm.Sound(self.audio_path)
        if to_time != 0:
            audio = snd.extract_part(from_time, to_time)
        plt.figure()
        if PITCH:
            pitch = snd.to_pitch()
            draw_pitch(pitch)
        if AMPLITUDE:
            draw_amplitude(snd)
        if INTENSITY:
            intensity = snd.to_intensity()
            draw_intensity(intensity)
        if SPECTROGRAM:
            spectrogram = snd.to_spectrogram
            draw_spectrogram(spectrogram)
        plt.xlim([snd.xmin, snd.xmax])


class Comment(object):
    def __init__(self, sentence):
        self.comment = {}
        if 'speaking_rate' not in sentence or 'filler_rate' not in sentence or 'pitch_variety' not in sentence:
            print('Error: Invalid Key!')
            return
        self.comment['speaking_rate'] = 'fast' if sentence['speaking_rate'] > 5 else 'slow' if sentence['speaking_rate'] < 3 else 'good'
        self.comment['pitch_variety'] = 'monotone' if sentence['pitch_variety'] < 120 else 'good'
        self.comment['filler_rate'] = 'good' if sentence['filler_rate'] < 5 else 'many' if sentence['filler_rate'] > 15 else 'some'


if __name__ == '__main__':
    config = {'username': 'USERNAME','password': 'PASSWORD'}
    Audio(config, 'resource/test.wav').segment().output()