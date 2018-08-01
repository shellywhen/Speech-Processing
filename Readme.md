# Speech Processing

 A simple python toolkit that supports multiple analysis tasks on an audio file of speech. This is in service of "Speaker+", a web-based intelligent training environment that auguments close-loop skill acquisition.

```python
import SpeechProcessing as SP
config = {'username':'USERNAME','password':'PASSWORD'}
speech_transcript = SP.Audio(config,'RELATIVE_PATH.wav'）
# return an Audio object list with sentences and corresponding timestamps
phonetic_info = speech_transcript.segment().output()
# get further phonetic information of each sentences and output a json file as shown in 'test_transcript.json'
```

- **Speech Recognition**: transcripts in a stence-level under IBM Watson Cloud Service. 
- **Speech Analysis**: speaking rate(syl/s), fillers (fillers/minute), pitch variety(Hz)

## Requirements

- Python 2.7(recommended and tested), 3.4, 3.5, 3.6(tested)
- [IBM Watson Cloud](https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/python.html?python#introduction) (for speech recognition)
  - Installation: `pip install watson-developer-cloud`
  - Credentials: [IBM Cloud Dashboard Page](https://console.bluemix.net/dashboard/apps?category=watson)
    - usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    - passwords are mixed-case alphanumeric strings
- [NLTK](https://www.nltk.org/) (for syllable dictionary)
  - Installation: `pip install nltk`
- [Parselmouth](http://parselmouth.readthedocs.io/en/latest/installation.html) (for pitch)

  - Installation: `pip install praat-parselmouth `

## Citation

```tex
@article{parselmouth,
    author = "Yannick Jadoul and Bill Thompson and Bart de Boer",
    title = "Introducing {P}arselmouth: A {P}ython interface to {P}raat",
    journal = "Journal of Phonetics",
    year = "in press"
}

@misc{praat,
    title = "{P}raat: doing phonetics by computer [{C}omputer program]",
    author = "Paul Boersma and David Weenink",
    howpublished = " Version 6.0.37, retrieved 3 February 2018 \url{http://www.praat.org/}",
    year = "2018"
}
```
