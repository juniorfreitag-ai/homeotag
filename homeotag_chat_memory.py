
import os
import json
import time
import numpy as np
import flet as ft
import mysql.connector
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from openai import OpenAI

# Carregar variáveis do .env
load_dotenv()

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "homeotag")
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-4o-mini")
MODEL_EMBED = os.getenv("MODEL_EMBED", "text-embedding-3-small")
SITES_FONTE = [s.strip() for s in os.getenv("SITES_FONTE", "").split(",") if s.strip()]

client = OpenAI(api_key=OPENAI_API_KEY)

# Conexão MySQL
def db_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )

# Embeddings
def embed_text(text: str):
    resp = client.embeddings.create(model=MODEL_EMBED, input=text)
    return resp.data[0].embedding

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# Recupera trechos do banco
def retrieve_snippets(question: str, top_k: int = TOP_K):
    q_vec = np.array(embed_text(question), dtype=np.float32)
    cn = db_conn()
    cur = cn.cursor(dictionary=True)
    cur.execute("SELECT id, titulo, trecho, vetor, fonte FROM conteudo_pdf")
    rows = cur.fetchall()
    cur.close()
    cn.close()
    scored = []
    for r in rows:
        try:
            vec_list = json.loads(r["vetor"])
            v = np.array(vec_list, dtype=np.float32)
            sim = cosine(q_vec, v)
            scored.append({
                "id": r["id"], "titulo": r["titulo"], "trecho": r["trecho"],
                "score": sim, "fonte": r.get("fonte")
            })
        except:
            continue
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def build_context(snippets, max_chars: int = MAX_CONTEXT_CHARS):
    context_parts = []
    total = 0
    for s in snippets:
        chunk = f"[Fonte: {s['titulo']}]\n{s['trecho'].strip()}"
        if total + len(chunk) > max_chars:
            break
        context_parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(context_parts)

def chat_completion(messages):
    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content

# ------------------- INTERFACE COM MEMÓRIA -------------------
def main(page: ft.Page):
    page.title = "Homeotag • Assistente com Memória"
    page.scroll = "auto"

    # Histórico da conversa (memória temporária)
    conversation_history = [
        {"role": "system", "content": "Você é um assistente homeopático que responde baseado em repertórios fornecidos."}
    ]

    chat_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    msg_input = ft.TextField(hint_text="Digite sua pergunta...", expand=True, multiline=True, min_lines=1, max_lines=4, color="black")
    send_btn = ft.FloatingActionButton(icon=ft.Icons.SEND)

    def bubble(text, is_user):
        bg = ft.Colors.BLUE_50 if is_user else ft.Colors.GREY_100
        align = ft.alignment.center_right if is_user else ft.alignment.center_left
        return ft.Container(
            content=ft.Text(text, selectable=True, color="black"),
            bgcolor=bg, padding=12, border_radius=16, alignment=align,
            width=page.width * 0.8 if page.width else None
        )

    def send_message(e=None):
        text = (msg_input.value or "").strip()
        if not text:
            return
        # Mostra mensagem do usuário
        chat_view.controls.append(bubble(text, is_user=True))
        page.update()
        msg_input.value = ""

        # Recupera contexto do banco
        snippets = retrieve_snippets(text, top_k=TOP_K)
        context_text = build_context(snippets, max_chars=MAX_CONTEXT_CHARS)
        if context_text.strip():
            conversation_history.append({"role": "system", "content": f"Contexto recuperado:\n\n{context_text}"})

        # Adiciona pergunta do usuário na memória
        conversation_history.append({"role": "user", "content": text})

        try:
            answer = chat_completion(conversation_history)
            # Adiciona resposta do assistente na memória
            conversation_history.append({"role": "assistant", "content": answer})
        except Exception as ex:
            answer = f"Erro ao consultar o modelo: {ex}"

        chat_view.controls.append(bubble(answer, is_user=False))
        page.update()

    send_btn.on_click = send_message
    msg_input.on_submit = send_message

    page.add(ft.Column([ft.Row([msg_input, send_btn]), chat_view], expand=True))

if __name__ == "__main__":
    ft.app(target=main)
