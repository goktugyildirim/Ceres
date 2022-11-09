import speech_recognition as sr
import pyaudio
from gtts import gTTS  # google text to speech
import pyttsx3


import pyaudio
import websockets
import asyncio
import base64
import json
# from openai_helper import ask_computer
# from api_secrets import API_KEY_ASSEMBLYAI


class Ceres1:
    def __init__(self):
        self.speec_recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)

    def listen(self):
        with sr.Microphone() as source:
            audio = self.speec_recognizer.listen(source)
        try:
            transcript = self.speec_recognizer.recognize_google(audio)
            print("You said: " + transcript)
            return transcript.lower()
        except sr.UnknownValueError:
            print("Could not understand!")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))


    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def respond(self, transcript):
        if self.there_exists(["hello", "ceres", "hey"], transcript):
            pass


    def there_exists(self, terms, transcript):
        for term in terms:
            if term in transcript:
                return True


if __name__== "__main__":

    ceres = Ceres1()
    tr = ceres.listen()
    ceres.respond(tr)



