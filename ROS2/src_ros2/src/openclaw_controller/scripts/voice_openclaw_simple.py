#!/usr/bin/env python3
"""
OpenClaw 语音对话程序 (纯 Python 版本)
"""

import sys
import os
import json
import requests
import argparse

sys.path.insert(0, '/home/ubuntu/large_models')
from config import *
from speech import speech

config_path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(config_path) as f:
    config = json.load(f)

GATEWAY_URL = f"http://127.0.0.1:{config['gateway']['port']}"
TOKEN = config['gateway']['auth']['token']


def call_openclaw(prompt: str, stream: bool = False) -> str:
    url = f"{GATEWAY_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openclaw",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }
    
    if stream:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
        print("Reply: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        chunk = json.loads(line[6:])
                        content = chunk['choices'][0].get('delta', {}).get('content', '')
                        print(content, end="", flush=True)
                    except:
                        pass
        print()
    else:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        return response.json()['choices'][0]['message']['content']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stream", action="store_true", help="enable streaming mode")
    args = parser.parse_args()
    
    print("Voice assistant started. Press Ctrl+C to exit.\n")
    asr = speech.RealTimeASR()
    
    while True:
        try:
            print("Listening...")
            text = asr.asr()
            if not text or text.strip() == "":
                continue
            
            print(f"Input: {text}")
            reply = call_openclaw(text, stream=args.stream)
            if not args.stream:
                print(f"Reply: {reply}")
            
        except KeyboardInterrupt:
            print("\nExited.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
