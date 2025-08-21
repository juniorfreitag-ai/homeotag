import flet as ft
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configuração da API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt inicial estruturado
SYSTEM_PROMPT = {
    "role": "system",
    "content": """Você é um assistente homeopático especializado. 
Suas respostas devem sempre ser organizadas de forma clara, usando Markdown:

- Introdução breve e personalizada.
- Seção **🔍 Dados relevantes** (resumo dos sintomas apresentados).
- Seção **🧭 Abordagem homeopática inicial** (rubricas e repertorização).
- Seção **📝 Medicamentos considerados** (com explicação curta de cada um).
- Seção **❓ Perguntas complementares** (organizadas em áreas: gerais, mentais, ginecológicos, sono, antecedentes etc.).
- Seção **📌 Orientações para conduta inicial** (com recomendações gerais).

Use listas, subtítulos e destaque os nomes dos medicamentos.
Nunca faça apenas um texto corrido — sempre formate de forma organizada.
"""
}

# Memória da conversa
conversation_history = [SYSTEM_PROMPT]

# Armazena última resposta para salvar/copiar
last_response = {"text": ""}


def chat_completion(messages):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content



def main(page: ft.Page):
    page.title = "Homeotag • Assistente v2"
    page.scroll = "auto"

    chat_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    msg_input = ft.TextField(
        hint_text="Digite sua pergunta...",
        expand=True,
        multiline=True,
        min_lines=1,
        max_lines=4,
        color="black"
    )

    def send_message(e):
        user_message = msg_input.value.strip()
        if not user_message:
            return

        # Mostra no chat
        chat_view.controls.append(ft.Text(f"👤 Você: {user_message}", weight="bold"))
        page.update()

        # Adiciona ao histórico
        conversation_history.append({"role": "user", "content": user_message})

        # Gera resposta
        resposta = chat_completion(conversation_history)

        # Salva no histórico
        conversation_history.append({"role": "assistant", "content": resposta})

        # Salva última resposta para PDF/copiar
        last_response["text"] = resposta

        # Mostra resposta em Markdown
        chat_view.controls.append(ft.Markdown(resposta, selectable=True))
        page.update()

        msg_input.value = ""
        page.update()

    def copy_response(e):
        if last_response["text"]:
            page.set_clipboard(last_response["text"])
            page.snack_bar = ft.SnackBar(ft.Text("✅ Resposta copiada!"))
            page.snack_bar.open = True
            page.update()

    def save_response_pdf(e):
        if last_response["text"]:
            filename = "resposta.pdf"
            page.snack_bar.open = True
            page.update()

    send_btn = ft.IconButton(icon=ft.Icons.SEND, on_click=send_message)
    copy_btn = ft.IconButton(icon=ft.Icons.COPY, on_click=copy_response, tooltip="Copiar resposta")

    page.add(
        ft.Column(
            [
                chat_view,
                ft.Row([msg_input, send_btn, copy_btn]),
            ],
            expand=True,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
