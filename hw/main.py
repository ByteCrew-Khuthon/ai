import torch
import torch.nn as nn
import time
import threading
import random
import requests
import json
from model import CustomModel
from get_audio import process_audio, get_wav_spec
from secrets import URL_COUGH, URL_TH


THRESHOLD = 0.024
mse = nn.MSELoss()
model = CustomModel()
checkpoint = torch.load("best.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

def coughJob() :
    while True :
        mel = process_audio()
        out = model(mel)
        loss = mse(mel, out)
        if loss.item() < THRESHOLD :
            # print("Detected!", loss.item())
            res = requests.post(URL_COUGH)
            print('COUGH : ', res)


def temphuJob() :
    while True :
        tmp = random.uniform(26.0, 28.0)
        hum = random.uniform(75.0, 80.0)
        header = {
    "Content-Type" : "application/json"
        }
        data = {
                "barnId" : 1,
                "temp" : tmp,
                "humidity" : hum,
                }
        res = requests.post(URL_TH, data = json.dumps(data), headers=header)
        print('TH : ', res)
        # print('Temp : ', tmp, ', Hum : ', hum)
        time.sleep(30)

t1 = threading.Thread(target = coughJob)
t2 = threading.Thread(target = temphuJob)

t1.start()
t2.start()