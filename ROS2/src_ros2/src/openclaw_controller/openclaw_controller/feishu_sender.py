#!/usr/bin/env python3
# encoding: utf-8

import json
import uuid
import time
import requests
from openclaw_controller.config import feishu_app_id, feishu_app_secret, feishu_chat_id

class FeishuSender:
    def __init__(self):
        self.app_id = feishu_app_id
        self.app_secret = feishu_app_secret
        self.chat_id = feishu_chat_id
        self._token = None
        self._token_expires_at = 0
        
        self.user_open_id = self._get_chat_user_id()
        print(f"Feishu ID: {self.user_open_id}")
    
    def _get_token(self):
        if self._token and time.time() < self._token_expires_at:
            return self._token
        
        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=10
        )
        data = resp.json()
        self._token = data["tenant_access_token"]
        self._token_expires_at = time.time() + data.get("expire", 7200) - 300
        return self._token
    
    def _get_chat_user_id(self):
        try:
            token = self._get_token()
            resp = requests.get(
                f"https://open.feishu.cn/open-apis/im/v1/chats/{self.chat_id}/members",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            data = resp.json()
            if data.get("code") == 0 and data.get("data", {}).get("items"):
                return data["data"]["items"][0]["member_id"]
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def send_text(self, text, sender_name="user"):
        try:
            token = self._get_token()
            url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"

            if sender_name == "robot":
                payload = {
                    "receive_id": self.chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text})
                }
            else:
                payload = {
                    "receive_id": self.chat_id,
                    "msg_type": "interactive",
                    "content": json.dumps({
                        "config": {
                            "wide_screen_mode": True
                        },
                        "header": {
                            "template": "blue",
                            "title": {
                                "content": f"You",
                                "tag": "plain_text"
                            }
                        },
                        "elements": [{
                            "tag": "markdown",
                            "content": text
                        }]
                    })
                }
            
            headers = {"Authorization": f"Bearer {token}"}
            if sender_name != "robot" and self.user_open_id:
                headers["X-User-Key"] = self.user_open_id
                payload["uuid"] = uuid.uuid4().hex[:16]
            
            requests.post(url, json=payload, headers=headers, timeout=10)
        except Exception as e:
            print(f"Error: {e}")
