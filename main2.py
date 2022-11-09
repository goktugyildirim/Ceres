import websockets
import asyncio
import base64
import json
from urllib.parse import urlencode
import pyaudio

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

# starts recording
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER)



word_boost = ["vehicle", "car"]
params = {"sample_rate": RATE, "word_boost": json.dumps(word_boost)}
# the AssemblyAI endpoint we're going to hit
URL = f"wss://api.assemblyai.com/v2/realtime/ws?{urlencode(params)}"



class CommandQueue:
    def __init__(self):
        self.queue = []
        self.max_size = 10

    def add(self, command):
        if len(self.queue) < self.max_size:
            self.queue.append(command)

    def get(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            return None

    def clear(self):
        self.queue = []

    def size(self):
        return len(self.queue)


async def send_receive():
    print(f'Connecting websocket to url ${URL}')
    async with websockets.connect(
            URL,
            extra_headers=(("Authorization", '34aa95c603bc404b81036eca3b1d6ab4'),),
            ping_interval=1,
            ping_timeout=20
    ) as _ws:
        await asyncio.sleep(0.1)
        session_begins = await _ws.recv()
        print(session_begins)
        async def send():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data": str(data)})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"
                await asyncio.sleep(0.01)

            return True

        async def receive():
            while True:
                try:
                    result_str = await _ws.recv()
                    transcript = json.loads(result_str)['text'].lower()
                    confidence = json.loads(result_str)['confidence']

                    if confidence > 0.1:
                        print('- You said: ' + transcript)
                        for word in json.loads(result_str)['words']:
                            print(word['text'], '  |  ', word['confidence'], '  |  ', word['start'], '-', word['end'])
                        print('###############################################################')

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"

        send_result, receive_result = await asyncio.gather(send(), receive())


asyncio.run(send_receive())
