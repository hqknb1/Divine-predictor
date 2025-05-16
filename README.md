<!--
README.md
本文件为极简RAG算命问答系统的使用说明，包含依赖安装、运行方法、功能简介等内容。
-->
# 极简RAG算命问答系统

> 🚀 一个本地可用、支持中文检索+大模型API生成的RAG（检索增强生成）对话系统，窗口化气泡聊天体验，适合学习、演示和二次开发。

---

## ⭐ 项目亮点
- 支持中文问题检索，返回最相关历史问答，结合 DeepSeek API 生成专业解答
- 本地 embedding 检索，无需本地大语言模型，显存/内存压力极小
- Tkinter 可视化气泡对话窗口，体验类微信/QQ
- 代码极简，易于理解和扩展
- 支持完全离线 embedding（只需提前下载模型）

---

## 📦 依赖安装
```bash
pip install -r requirements.txt
```
- 需 Python 3.8+
- Windows 自带 tkinter，Linux 需 `sudo apt install python3-tk`

---

## 📄 数据准备
将你的 `fortune-telling.json` 数据文件放在 `rag_minimal` 目录下。

---

## 🔑 API配置
- 你需要一个有效的 DeepSeek API Key（已在代码中预留变量，可替换为你自己的）。

---

## 🧠 本地Embedding模型准备
- 推荐使用 [Jerry0/text2vec-large-chinese](https://modelscope.cn/models/Jerry0/text2vec-large-chinese) 或 [GanymedeNil/text2vec-base-chinese](https://huggingface.co/GanymedeNil/text2vec-base-chinese)
- 下载后放到如 `D:/models/text2vec-base-chinese`，并在 `gui_app.py` 里设置 `EMBEDDING_MODEL` 路径

---

## 🚀 快速运行
```bash
python gui_app.py
```
- 启动后会弹出气泡对话窗口，输入你的问题即可体验。

---

## 🖥️ 功能说明
- 支持中文检索+大模型生成的RAG对话
- 聊天内容可上下滚动，历史消息不丢失
- 支持鼠标滚轮、拖动滚动条
- 系统自动开场白，体验更自然

---

## 🛠️ 常见问题
- **embedding模型下载慢/失败？**
  - 推荐用 modelscope/huggingface 网页手动下载，或让朋友帮忙下载后本地加载
- **API Key无效？**
  - 请到 DeepSeek 官网注册获取
- **窗口不显示/乱码？**
  - 检查 tkinter 是否安装，或尝试更换字体

---

## 📢 贡献&交流
- 欢迎 issue、PR、二次开发！
- 有问题可在 GitHub issue 区留言

---

## License
MIT 