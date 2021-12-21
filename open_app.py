#Import Packages
import speech_recognition as sr  
import playsound # to play saved mp3 file 
from gtts import gTTS # google text to speech 
import os # to save/open files 
import wolframalpha # to calculate strings into formula 
from selenium import webdriver # to control browser operations 
import subprocess
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


r = sr.Recognizer()
mic = sr.Microphone(device_index=3)





# create lists of applications 
# list the directory and remove the extension
d = '/Applications'
#apps = list(map(lambda x: x.split('.app')[0], os.listdir(d)))
records = []
apps = os.listdir(d)

for app in apps:
    record = {}
    record['voice_command'] = 'open ' + app.split('.app')[0]
    record['sys_command'] = 'open ' + d +'/%s' %app.replace(' ','\ ')
    records.append(record)


es = Elasticsearch(['localhost:9200'])
bulk(es, records, index='voice_assistant', doc_type='text', raise_on_error=True)

def search_es(query):
    res = es.search(index="voice_assistant", doc_type="text", body={                     
    "query" :{
        "match": {
            "voice_command": {
                "query": query,
                "fuzziness": 2
            }
            }
        },
    })
    return res['hits']['hits'][0]['_source']['sys_command']