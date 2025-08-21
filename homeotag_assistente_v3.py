import flet as ft
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Memória da conversa (por enquanto só em RAM)
conversation_history = [
    {
        "role": "system",
        "content": (
            "Você é um assistente homeopático clínico avançado, especialista em repertorização. "
            "Seu estilo de resposta deve ser organizado em Markdown, com subtítulos, listas e destaques. "
            "Sempre siga a seguinte estrutura: \n\n"
            "### 🔍 Dados clínicos relevantes\n"
            "Liste em tópicos os principais dados que o usuário trouxe.\n\n"
            "### 🧭 Abordagem homeopática inicial\n"
            "Liste rubricas sugeridas e faça uma repertorização preliminar.\n\n"
            "### 📝 Medicações consideradas\n"
            "Apresente os medicamentos mais indicados com uma breve justificativa clínica.\n\n"
            "### ❓ Perguntas complementares\n"
            "Liste perguntas que aprofundem a anamnese para individualizar o caso.\n\n"
            "### 📌 Orientações iniciais\n"
            "Sugira condutas ou caminhos possíveis, sempre destacando que a decisão final é do médico homeopata.\n\n"
            "Responda de forma clara, estruturada e didática, como faria um professor de homeopatia."
        )
    }
]

def main(page: ft.Page):
    page.title = "Homeotag Assistente v3"
    page.scroll = "adaptive"
    page.theme_mode = "light"

    chat_column = ft.Column(expand=True, scroll="auto")
    input_field = ft.TextField(
        label="Digite sua pergunta ou caso clínico",
        multiline=True,
        expand=True
    )

    def add_message(role, content):
        # Renderiza mensagens como Markdown para melhor visualização
        if role == "user":
            msg = ft.Markdown(
                f"**Você:** {content}",
                selectable=True,
                extension_set="gitHubWeb"
            )
        else:
            msg = ft.Markdown(
                f"**Assistente:**\n\n{content}",
                selectable=True,
                extension_set="gitHubWeb"
            )
        chat_column.controls.append(msg)
        page.update()

    def send_message(e):
        user_message = input_field.value.strip()
        if not user_message:
            return

        # Adiciona pergunta do usuário
        conversation_history.append({"role": "user", "content": user_message})
        add_message("user", user_message)
        input_field.value = ""
        page.update()

        # Chama a API OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_message})
        add_message("assistant", assistant_message)

    def copy_last_response(e):
        # Copiar última resposta do assistente
        last_response = ""
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
        if last_response:
            page.set_clipboard(last_response)
            page.snack_bar = ft.SnackBar(ft.Text("Resposta copiada!"))
            page.snack_bar.open = True
            page.update()

    def save_last_response(e):
        # Salvar última resposta em TXT
        last_response = ""
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
        if last_response:
            with open("resposta_assistente.txt", "w", encoding="utf-8") as f:
                f.write(last_response)
            page.snack_bar = ft.SnackBar(ft.Text("Resposta salva em resposta_assistente.txt"))
            page.snack_bar.open = True
            page.update()

    # Área principal
    page.add(
        chat_column,
        ft.Row([
            input_field,
            ft.IconButton(icon=ft.Icons.SEND, on_click=send_message),
        ]),
        ft.Row([
            ft.ElevatedButton("Copiar resposta", on_click=copy_last_response),
            ft.ElevatedButton("Salvar em TXT", on_click=save_last_response),
        ])
    )

# Rodar app
ft.app(target=main)
