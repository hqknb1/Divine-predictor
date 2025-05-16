# -*- coding: utf-8 -*-
"""
文件名: gui_app.py
功能: 极简RAG算命问答系统的可视化气泡对话窗口，支持本地embedding检索+DeepSeek API生成。
依赖: torch, transformers, faiss, pandas, tkinter, requests
作者: 你的名字
日期: 2024-05-16

主要特性：
- 支持中文检索+大模型生成的RAG对话
- Tkinter气泡式聊天窗口，体验类微信/QQ
- 聊天内容可上下滚动，历史消息不丢失
- 系统自动开场白，体验更自然
"""
import os
import tkinter as tk
from tkinter import Scrollbar, Canvas, Frame, Entry, Button
import threading
import json
import numpy as np
import pandas as pd
import torch
import faiss
import requests
from transformers import AutoTokenizer, AutoModel

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-xxxxxx")  # 默认值可留空或写""
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 数据与模型路径
DATA_PATH = 'fortune-telling.json'
EMBEDDING_MODEL = 'D:/models/text2vec-base-chinese'  # 本地embedding模型目录

# 数据加载

def load_data(path):
    """加载JSON格式的问答数据为DataFrame"""
    with open(path, 'r', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

# 向量化相关

def get_embedding(model, tokenizer, text, device):
    """将文本转为embedding向量"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    faiss.normalize_L2(emb)
    return emb

def build_faiss_index(embeddings):
    """构建FAISS向量索引"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search(index, query_emb, top_k=3):
    """向量检索，返回最相关的索引"""
    scores, idxs = index.search(query_emb, top_k)
    return idxs[0], scores[0]

# 调用DeepSeek大模型API

def call_deepseek_api(query, context):
    """调用DeepSeek API生成答案"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = "你是一位精通中国传统命理学的算命大师，善于结合历史问答知识为用户提供权威、详细的命理解读。"
    user_content = f"用户问题：{query}\n\n相关历史问答：\n{context}\n\n请结合以上内容，生成专业、详细的算命解答。"
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    return result['choices'][0]['message']['content']

# 聊天气泡组件
class ChatBubble(Frame):
    """单条消息的气泡组件"""
    def __init__(self, master, text, is_user):
        super().__init__(master, bg='white')
        anchor = 'e' if is_user else 'w'
        bubble_bg = '#DCF8C6' if is_user else '#F1F0F0'
        bubble_fg = 'black'
        padx = (60, 10) if is_user else (10, 60)
        label = tk.Label(self, text=text, bg=bubble_bg, fg=bubble_fg, wraplength=350, justify='left', font=('微软雅黑', 12), padx=10, pady=6)
        label.pack(anchor=anchor, padx=padx, pady=2)
        self.pack(anchor=anchor, fill='x', pady=2, padx=5)

# 主聊天窗口
class RAGChatGUI:
    """RAG算命对话窗口，支持气泡聊天、滚动、开场白"""
    def __init__(self, master, data, index, model, tokenizer, device):
        self.master = master
        self.data = data
        self.index = index
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        master.title("极简RAG算命对话气泡窗口")
        master.geometry('600x600')
        master.configure(bg='white')

        # 聊天内容区（可滚动）
        self.canvas = Canvas(master, bg='white', highlightthickness=0)
        self.scrollbar = Scrollbar(master, orient='vertical', command=self.canvas.yview)
        self.chat_frame = Frame(self.canvas, bg='white')
        self.chat_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.chat_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='top', fill='both', expand=True, padx=0, pady=0)
        self.scrollbar.pack(side='right', fill='y')

        # 支持鼠标滚轮滚动
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)  # Windows
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)    # Linux
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)    # Linux

        # 输入区
        self.entry = Entry(master, font=('微软雅黑', 13))
        self.entry.pack(side='left', fill='x', expand=True, padx=(10,0), pady=(0,10), ipady=6)
        self.entry.bind('<Return>', self.on_send)
        self.send_button = Button(master, text="发送", command=self.on_send, font=('微软雅黑', 12), bg='#4CAF50', fg='white')
        self.send_button.pack(side='right', padx=(5,10), pady=(0,10))

        # 系统开场白
        self.add_bubble("贫道神算子，精通命理卜卦。阁下有何疑惑，尽管问来。", is_user=False)

    def _on_mousewheel(self, event):
        # 支持鼠标滚轮上下滚动
        if event.num == 5 or event.delta == -120:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta == 120:
            self.canvas.yview_scroll(-1, "units")

    def on_send(self, event=None):
        """发送消息事件"""
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.add_bubble(user_input, is_user=True)
        self.entry.delete(0, tk.END)
        threading.Thread(target=self.answer, args=(user_input,)).start()

    def answer(self, query):
        """处理用户提问，检索+API生成"""
        self.add_bubble("贫道正在勘探天机，请莫要着急...", is_user=False)
        query_emb = get_embedding(self.model, self.tokenizer, query, self.device)
        idxs, _ = search(self.index, query_emb)
        context = ''
        for i in idxs:
            q = self.data.iloc[i]["Question"]
            a = self.data.iloc[i]["Response"]
            context += f"问：{q}\n答：{a}\n"
        try:
            answer = call_deepseek_api(query, context)
            self.add_bubble(answer, is_user=False)
        except Exception as e:
            self.add_bubble(f"调用DeepSeek API失败：{e}", is_user=False)

    def add_bubble(self, text, is_user):
        """添加一条聊天气泡"""
        bubble = ChatBubble(self.chat_frame, ("你：" if is_user else "神算子：") + text, is_user)
        self.master.update_idletasks()
        self.canvas.yview_moveto(1.0)

# 程序入口

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_data(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device).eval()
    texts = (data['Question'] + ' ' + data['Response']).tolist()
    embeddings = np.vstack([get_embedding(model, tokenizer, t, device) for t in texts])
    index = build_faiss_index(embeddings)

    root = tk.Tk()
    app = RAGChatGUI(root, data, index, model, tokenizer, device)
    root.mainloop()

if __name__ == '__main__':
    main() 