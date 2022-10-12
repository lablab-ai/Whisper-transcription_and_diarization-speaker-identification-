# Whisper transcription and diarization (speaker-identification)

How to use OpenAIs Whisper to transcribe and diarize audio files

## What is Whisper?

Whisper is an State-of-the-Art speech recognition system from OpenAI that has been trained on 680,000 hours 
of multilingual and multitask supervised data collected from the web. This large and diverse 
dataset leads to improved robustness to accents, background noise and technical language. In 
addition, it enables transcription in multiple languages, as well as translation from those 
languages into English. OpenAI released the models and code to serve as a foundation for building useful
applications that leverage speech recognition.

One big downside of Whisper is though, that it can not tell you who is speaking in a conversation.
That's a problem when analyzing conversations. This is where diarization comes in. Diarization is 
the process of identifying who is speaking in a conversation.

In this tutorial you will learn how to identify the speakers, and then match them with the transcriptions of Whisper.
We will use `pyannote-audio` to accomplish this. Let's get started!

### Preparing the audio

First, we need to prepare the audio file. We will use the first 20 minutes of Lex Fridmans podcast with Yann download.
To download the video and extract the audio, we will use `yt-dlp` package. 

```bash
!pip install -U yt-dlp
```

We will also need [ffmpeg](https://www.wikihow.com/Install-FFmpeg-on-Windows) installed

```bash
!wget -O - -q  https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz | xz -qdc| tar -x
```

Now we can do the actual download and audio extraction via the command line.

```bash
!yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format wav  -o download.wav -- https://youtu.be/SGzMElJ11Cc
```

Now we have the `download.wav` file in our working directory. Let's cut the first 20 minutes of the audio. We can use the pydub package for this with just a few lines of code.

```bash
!pip install pydub
```
```python
from pydub import AudioSegment

t1 = 0 * 1000 # works in milliseconds
t2 = 20 * 60 * 1000

newAudio = AudioSegment.from_wav("download.wav")
a = newAudio[t1:t2]
a.export("audio.wav", format="wav") 
```

`audio.wav` is now the first 20 minutes of the audio file.

### Pyannote's Diarization

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on PyTorch 
machine learning framework, it provides a set of trainable end-to-end neural building blocks that 
can be combined and jointly optimized to build speaker diarization pipelines. `pyannote.audio` also
comes with pretrained models and pipelines covering a wide range of domains for voice activity 
detection, speaker segmentation, overlapped speech detection, speaker embedding reaching 
state-of-the-art performance for most of them.

Installing Pyannote and running it on the video audio to generate the diarizations.

```bash
!pip install pyannote.audio
```

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
```

```python
DEMO_FILE = {'uri': 'blabal', 'audio': 'audio.wav'}
dz = pipeline(DEMO_FILE)  

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))
```

Lets print this out to see what it looks like.

```pyhton
print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
```

The output:
```
(<Segment(2.03344, 36.8128)>, 0, 'SPEAKER_00')
(<Segment(38.1122, 51.3759)>, 0, 'SPEAKER_00')
(<Segment(51.8653, 90.2053)>, 1, 'SPEAKER_01')
(<Segment(91.2853, 92.9391)>, 1, 'SPEAKER_01')
(<Segment(94.8628, 116.497)>, 0, 'SPEAKER_00')
(<Segment(116.497, 124.124)>, 1, 'SPEAKER_01')
(<Segment(124.192, 151.597)>, 1, 'SPEAKER_01')
(<Segment(152.018, 179.12)>, 1, 'SPEAKER_01')
(<Segment(180.318, 194.037)>, 1, 'SPEAKER_01')
(<Segment(195.016, 207.385)>, 0, 'SPEAKER_00')
```

This looks pretty good already, but let's clean the data a little bit:
  
```python

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

import re
dz = open('diarization.txt').read().splitlines()
dzList = []
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = millisec(start) - spacermilli
  end = millisec(end)  - spacermilli
  lex = not re.findall('SPEAKER_01', string=l)
  dzList.append([start, end, lex])

print(*dzList[:10], sep='\n')
```
```
[33, 34812, True]
[36112, 49375, True]
[49865, 88205, False]
[89285, 90939, False]
[92862, 114496, True]
[114496, 122124, False]
[122191, 149596, False]
[150018, 177119, False]
[178317, 192037, False]
[193015, 205385, True]
```

Now we have the diarization data in a list. The first two numbers are the start and end time of 
the speaker segment in milliseconds. The third number is a boolean that tells us if the speaker 
is Lex or not. 

### Preparing audio file from the diarization

Next, we will attach the audio segements according to the diarization, with a spacer as the delimiter.

```python
from pydub import AudioSegment
import re 

sounds = spacer
segments = []

dz = open('diarization.txt').read().splitlines()
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = int(millisec(start)) #milliseconds
  end = int(millisec(end))  #milliseconds
  
  segments.append(len(sounds))
  sounds = sounds.append(audio[start:end], crossfade=0)
  sounds = sounds.append(spacer, crossfade=0)

sounds.export("dz.wav", format="wav") #Exports to a wav file in the current path.
```

```python
print(segments[:8])
```

```bash
[2000, 38779, 54042, 94382, 98036, 121670, 131297, 160702]
```

### Transcription with Whisper 

Next, we will use Whisper to transcribe the different segments of the audio file. Important: There is 
a version conflict with pyannote.audio resulting in an error. Our workaround is to 
first run Pyannote and then whisper. You can safely ignore the error.

Installing Open AI Whisper.
```bash
!pip install git+https://github.com/openai/whisper.git 
```

Running Open AI whisper on the prepared audio file. It writes the transcription into a file. You can 
adjust the model size to your needs. You can find all models on the model card on Github.

```bash
!whisper dz.wav --language en --model base
```

```
[00:00.000 --> 00:04.720]  The following is a conversation with Yann LeCun,
[00:04.720 --> 00:06.560]  his second time on the podcast.
[00:06.560 --> 00:11.160]  He is the chief AI scientist at Meta, formerly Facebook,
[00:11.160 --> 00:15.040]  professor at NYU, touring award winner,
[00:15.040 --> 00:17.600]  one of the seminal figures in the history
[00:17.600 --> 00:20.460]  of machine learning and artificial intelligence,
...
```

In order to work with .vtt files, we need to install the webvtt-py library.

```bash
!pip install -U webvtt-py
```

Lets take a look at the data:
```python
import webvtt

captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read('dz.wav.vtt')]
print(*captions[:8], sep='\n')
```

```
[0, 4720, 'The following is a conversation with Yann LeCun,']
[4720, 6560, 'his second time on the podcast.']
[6560, 11160, 'He is the chief AI scientist at Meta, formerly Facebook,']
[11160, 15040, 'professor at NYU, touring award winner,']
[15040, 17600, 'one of the seminal figures in the history']
[17600, 20460, 'of machine learning and artificial intelligence,']
[20460, 23940, 'and someone who is brilliant and opinionated']
[23940, 25400, 'in the best kind of way,']
...
```

### Matching the Transcriptions and the Diarizations

Next, we will match each transcribtion line to some diarizations, and display everything by
generating a HTML file. To get the correct timing, we should take care of the parts in original 
audio that were in no diarization segment. We append a new div for each segment in our audio.


```pyhton 
# we need this fore our HTML file (basicly just some styling)
preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <title>Lexicap</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .e {\n            display: inline-block;\n        }\n        .t {\n            display: inline-block;\n        }\n        #player {\n\t\tposition: sticky;\n\t\ttop: 20px;\n\t\tfloat: right;\n\t}\n    </style>\n  </head>\n  <body>\n    <h2>Yann LeCun: Dark Matter of Intelligence and Self-Supervised Learning | Lex Fridman Podcast #258</h2>\n  <div  id="player"></div>\n    <script>\n      var tag = document.createElement(\'script\');\n      tag.src = "https://www.youtube.com/iframe_api";\n      var firstScriptTag = document.getElementsByTagName(\'script\')[0];\n      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n      var player;\n      function onYouTubeIframeAPIReady() {\n        player = new YT.Player(\'player\', {\n          height: \'210\',\n          width: \'340\',\n          videoId: \'SGzMElJ11Cc\',\n        });\n      }\n      function setCurrentTime(timepoint) {\n        player.seekTo(timepoint);\n   player.playVideo();\n   }\n    </script><br>\n'
postS = '\t</body>\n</html>'

from datetime import timedelta

html = list(preS)

for i in range(len(segments)):
  idx = 0
  for idx in range(len(captions)):
    if captions[idx][0] >= (segments[i] - spacermilli):
      break;
  
  while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
    c = captions[idx]  
    
    start = dzList[i][0] + (c[0] -segments[i])

    if start < 0: 
      start = 0
    idx += 1

    start = start / 1000.0
    startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                            (int)(start % 3600 // 60), 
                                            start % 60)
    
    html.append('\t\t\t<div class="c">\n')
    html.append(f'\t\t\t\t<a class="l" href="#{startStr}" id="{startStr}">link</a> |\n')
    html.append(f'\t\t\t\t<div class="s"><a href="javascript:void(0);" onclick=setCurrentTime({int(start)})>{startStr}</a></div>\n')
    html.append(f'\t\t\t\t<div class="t">{"[Lex]" if dzList[i][2] else "[Yann]"} {c[2]}</div>\n')
    html.append('\t\t\t</div>\n\n')

html.append(postS)
s = "".join(html)

with open("lexicap.html", "w") as text_file:
    text_file.write(s)
print(s)
```


---

[![Artificial Intelligence Hackathons, tutorials and Boilerplates](https://storage.googleapis.com/lablab-static-eu/images/github/lablab-banner.jpg)](https://lablab.ai)


## Join the LabLab Discord


![Discord Banner 1](https://discordapp.com/api/guilds/877056448956346408/widget.png?style=banner1)  
On lablab discord, we discuss this repo and many other topics related to artificial intelligence! Checkout upcoming [Artificial Intelligence Hackathons](https://lablab.ai) Event


[![Acclerating innovation through acceleration](https://storage.googleapis.com/lablab-static-eu/images/github/nn-group-loggos.jpg)](https://newnative.ai)
