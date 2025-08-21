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

def db_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )

def embed_text(text: str):
    resp = client.embeddings.create(model=MODEL_EMBED, input=text)
    return resp.data[0].embedding

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

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
            scored.append({"id": r["id"], "titulo": r["titulo"], "trecho": r["trecho"], "score": sim, "fonte": r.get("fonte")})
        except:
            continue
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def build_context(snippets, max_chars: int = MAX_CONTEXT_CHARS):
    context_parts = []
    used = []
    total = 0
    for s in snippets:
        chunk = f"[Fonte: {s['titulo']}]\n{s['trecho'].strip()}"
        if total + len(chunk) > max_chars:
            break
        context_parts.append(chunk)
        used.append(s)
        total += len(chunk)
    return "\n\n---\n\n".join(context_parts), used

def chat_completion(messages):
    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content

# ------------------- CRAWLER -------------------
VISITADOS = set()
def extrair_texto(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except:
        return None
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["nav", "footer", "header", "script", "style"]):
        tag.extract()
    texto = "\n\n".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())
    return texto.strip() if texto else None

def gerar_embedding(texto):
    resp_emb = client.embeddings.create(model=MODEL_EMBED, input=texto[:3000])
    return resp_emb.data[0].embedding

def salvar_no_banco(titulo, trecho, vetor, fonte):
    cn = db_conn()
    cur = cn.cursor()
    cur.execute(
        "INSERT INTO conteudo_pdf (titulo, trecho, vetor, fonte) VALUES (%s, %s, %s, %s)",
        (titulo, trecho, json.dumps(vetor), fonte)
    )
    cn.commit()
    cur.close()
    cn.close()

def crawler(base_url, titulo_site, log_fn):
    fila = [base_url]
    dominio = urlparse(base_url).netloc
    while fila:
        url = fila.pop(0)
        if url in VISITADOS:
            continue
        VISITADOS.add(url)
        log_fn(f"[BAIXANDO] {url}")
        texto = extrair_texto(url)
        if texto:
            vetor = gerar_embedding(texto)
            salvar_no_banco(titulo_site, texto[:1000], vetor, url)
        time.sleep(1)
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if urlparse(link).netloc == dominio and link not in VISITADOS:
                    if "#" not in link and not link.endswith((".pdf", ".jpg", ".png", ".gif")):
                        fila.append(link)
        except:
            pass

# ------------------- INTERFACE SIMPLES -------------------
def main(page: ft.Page):
    page.title = "Homeotag • Assistente (Simples)"
    page.scroll = "auto"
    chat_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    msg_input = ft.TextField(hint_text="Digite sua pergunta...", expand=True, multiline=True, min_lines=1, max_lines=4)
    send_btn = ft.FloatingActionButton(icon=ft.Icons.SEND)
    atualizar_btn = ft.ElevatedButton(text="Atualizar Base", icon=ft.Icons.UPDATE)

    def bubble(text, is_user):
        bg = ft.Colors.BLUE_50 if is_user else ft.Colors.GREY_100
        align = ft.alignment.center_right if is_user else ft.alignment.center_left
        return ft.Container(content=ft.Markdown(text, selectable=True), bgcolor=bg, padding=12, border_radius=16, alignment=align, width=page.width * 0.8 if page.width else None)

    def send_message(e=None):
        text = (msg_input.value or "").strip()
        if not text:
            return
        chat_view.controls.append(bubble(text, is_user=True))
        page.update()
        msg_input.value = ""
        snippets = retrieve_snippets(text, top_k=TOP_K)
        context_text, used_sources = build_context(snippets, max_chars=MAX_CONTEXT_CHARS)
        messages = [{"role": "system", "content": "Você é um assistente homeopático que responde baseado em repertórios fornecidos."}]
        if context_text.strip():
            messages.append({"role": "system", "content": f"Contexto recuperado:\n\n{context_text}"})
        messages.append({"role": "user", "content": text})
        try:
            answer = chat_completion(messages)
        except Exception as ex:
            answer = f"Erro ao consultar o modelo: {ex}"
        chat_view.controls.append(bubble(answer, is_user=False))
        page.update()

    def log_status(msg):
        chat_view.controls.append(bubble(msg, is_user=False))
        page.update()

    def atualizar_base(e=None):
        chat_view.controls.append(bubble("🔄 Iniciando atualização da base...", is_user=False))
        page.update()
        for site in SITES_FONTE:
            if "|" in site:
                url, titulo = site.split("|", 1)
            else:
                url, titulo = site, site
            crawler(url.strip(), titulo.strip(), log_status)
        chat_view.controls.append(bubble("✅ Base atualizada com sucesso!", is_user=False))
        page.update()

    atualizar_btn.on_click = atualizar_base
    send_btn.on_click = send_message
    msg_input.on_submit = send_message

    page.add(ft.Column([ft.Row([msg_input, send_btn]), atualizar_btn, chat_view], expand=True))

if __name__ == "__main__":
    ft.app(target=main)
