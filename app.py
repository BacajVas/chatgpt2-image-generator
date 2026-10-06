"""Local conversational image studio powered by Diffusers and Gradio."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import gradio as gr

from core import conversation_context, generate_and_remember
from memory import MemoryStore
from models import ModelHub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", PROJECT_DIR / "data")).expanduser().resolve()
if DATA_DIR.is_relative_to(PROJECT_DIR) and DATA_DIR != PROJECT_DIR / "data":
    raise ValueError("Keep APP_DATA_DIR outside this Git repository, or use data/.")
memory = MemoryStore(DATA_DIR)
models = ModelHub()
chat_lock = threading.Lock()


def gallery_items() -> list[tuple[str, str]]:
    return [
        (item["image_path"], item["prompt"][:90])
        for item in memory.generations()
        if Path(item["image_path"]).is_file()
    ]


def history_choices() -> list[tuple[str, int]]:
    return [
        (f"#{item['id']} · {item['prompt'][:65]}", item["id"])
        for item in memory.generations()
    ]


def chat_items() -> list[dict]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in memory.messages(limit=100)
    ]


def generate(prompt: str, style: str, steps: int, seed: int):
    try:
        item = generate_and_remember(memory, models, prompt, style, steps, seed)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Image generation failed")
        raise gr.Error(
            "Не удалось создать картинку. Проверьте загрузку модели, интернет "
            "и свободную память; подробности есть в терминале."
        ) from exc
    return (
        item["image_path"],
        f"Сохранено в памяти. Номер {item['id']}, seed {item['seed']}.",
        gallery_items(),
        gr.update(choices=history_choices(), value=item["id"]),
    )


def load_generation(record_id: int):
    if record_id is None:
        raise gr.Error("Выберите картинку из истории.")
    item = memory.generation(int(record_id))
    if not item or not Path(item["image_path"]).is_file():
        raise gr.Error("Сохранённая картинка не найдена.")
    return (
        item["prompt"], item["style"], item["steps"], item["seed"],
        item["image_path"], f"Загружено из памяти: номер {item['id']}.",
    )


def save_name(name: str):
    try:
        memory.set_name(name)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return f"Запомнил имя: {memory.profile()['display_name']}."


def save_style(style: str):
    try:
        memory.set_style(style)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return "Предпочитаемый стиль сохранён."


def save_notes(notes: str):
    try:
        memory.set_notes(notes)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return "Заметки о вас сохранены."


def send_message(text: str):
    text = text.strip()
    if not text or len(text) > 2000:
        raise gr.Error("Введите сообщение длиной до 2000 символов.")
    with chat_lock:
        try:
            answer = models.reply(conversation_context(memory, text))
            memory.add_exchange(text, answer)
        except Exception as exc:
            logger.exception("Chat failed")
            raise gr.Error(
                "Не удалось получить ответ. Проверьте загрузку разговорной модели, "
                "интернет и свободную память; подробности есть в терминале."
            ) from exc
        return chat_items(), "", "Разговор сохранён в памяти."


def clear_chat():
    with chat_lock:
        memory.clear_messages()
        return [], "История разговора удалена. Сохранённые картинки остались."


def refine_prompt(prompt: str):
    prompt = prompt.strip()
    if not prompt or len(prompt) > 800:
        raise gr.Error("Сначала опишите картинку — до 800 символов.")
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's image idea as one vivid, concise English "
                "text-to-image prompt. Keep the requested subject and composition. "
                "Output only the improved prompt, without explanation."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        improved = models.reply(messages)
    except Exception as exc:
        logger.exception("Prompt refinement failed")
        raise gr.Error(
            "Не удалось улучшить описание. Проверьте загрузку разговорной "
            "модели и свободную память."
        ) from exc
    return improved, "Описание улучшено. Проверьте его перед генерацией."


with gr.Blocks(title="GPT Bro — картинки и разговор") as demo:
    gr.Markdown(
        "# GPT Bro\nЛокальный генератор картинок с памятью о разговоре. "
        "По умолчанию работает только на этом компьютере."
    )
    with gr.Tab("Создать картинку"):
        prompt_box = gr.Textbox(
            label="Что нарисовать?", lines=3,
            placeholder="Например: деревянная лодка на спокойном озере на рассвете",
        )
        style_box = gr.Textbox(
            label="Любимый стиль — запоминается",
            value=memory.profile()["preferred_style"],
            placeholder="Например: фотореализм, мягкий утренний свет",
        )
        with gr.Row():
            steps_box = gr.Slider(1, 4, value=1, step=1, label="Шаги генерации")
            seed_box = gr.Number(value=-1, precision=0, label="Seed (-1 = случайный)")
        with gr.Row():
            generate_button = gr.Button("Создать", variant="primary")
            refine_button = gr.Button("Улучшить описание с ИИ")
            style_button = gr.Button("Запомнить стиль")
        result_image = gr.Image(label="Результат", type="pil")
        image_status = gr.Markdown()
        gr.Markdown(
            "Модель картинок: [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) "
            "— Powered by Stability AI. Первое создание скачает её файлы."
        )
        with gr.Accordion("Предыдущие картинки", open=False):
            history_gallery = gr.Gallery(
                label="Сохранённые результаты", value=gallery_items(),
                columns=3, height=270, object_fit="contain",
            )
            history_picker = gr.Dropdown(
                label="Выбрать прошлый запрос", choices=history_choices()
            )
            with gr.Row():
                load_button = gr.Button("Вернуть запрос и настройки")
                refresh_button = gr.Button("Обновить историю")

    with gr.Tab("Разговор"):
        gr.Markdown(
            "Помощник помнит последние реплики и созданные картинки. "
            "Вся история хранится локально; имя здесь не является входом в аккаунт."
        )
        with gr.Row():
            name_box = gr.Textbox(
                value=memory.profile()["display_name"], label="Как к вам обращаться?"
            )
            name_button = gr.Button("Запомнить имя")
        name_status = gr.Markdown()
        notes_box = gr.Textbox(
            value=memory.profile()["notes"], label="Что помнить о вас?",
            lines=2, placeholder="Например: люблю реалистичные пейзажи",
        )
        notes_button = gr.Button("Запомнить заметки")
        notes_status = gr.Markdown()
        conversation = gr.Chatbot(
            label="Чат", type="messages", value=chat_items(), height=420,
            allow_tags=False,
        )
        message_box = gr.Textbox(label="Сообщение", lines=2)
        with gr.Row():
            send_button = gr.Button("Отправить", variant="primary")
            clear_button = gr.Button("Удалить историю разговора")
        chat_status = gr.Markdown(
            f"Разговорная модель: {models.chat_model_id}. "
            "При первом сообщении она будет загружена."
        )

    generate_button.click(
        generate, inputs=[prompt_box, style_box, steps_box, seed_box],
        outputs=[result_image, image_status, history_gallery, history_picker],
    )
    refine_button.click(refine_prompt, inputs=prompt_box,
                        outputs=[prompt_box, image_status])
    style_button.click(save_style, inputs=style_box, outputs=image_status)
    load_button.click(
        load_generation, inputs=history_picker,
        outputs=[prompt_box, style_box, steps_box, seed_box, result_image, image_status],
    )
    refresh_button.click(
        lambda: (gallery_items(), gr.update(choices=history_choices())),
        outputs=[history_gallery, history_picker],
    )
    name_button.click(save_name, inputs=name_box, outputs=name_status)
    notes_button.click(save_notes, inputs=notes_box, outputs=notes_status)
    send_button.click(send_message, inputs=message_box,
                      outputs=[conversation, message_box, chat_status])
    message_box.submit(send_message, inputs=message_box,
                       outputs=[conversation, message_box, chat_status])
    clear_button.click(clear_chat, outputs=[conversation, chat_status])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        allowed_paths=[str(memory.images_dir)],
    )
