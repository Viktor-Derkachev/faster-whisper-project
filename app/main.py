import streamlit as st
import requests
from faster_whisper import WhisperModel
from pathlib import Path

# Настройка страницы без лишних мигающих элементов
st.set_page_config(page_title="AI Transcriber", layout="wide")
st.title("Транскрибация и AI Анализ")

OLLAMA_URL = "http://ollama:11434"
DATA_DIR = Path("/data")

@st.cache_resource
def load_whisper():
    return WhisperModel("medium", device="cpu", compute_type="int8")

def query_ollama(prompt, language):
    MODEL_NAME = "llama3.2:3b"

    # Добавляем строгое указание языка в промпт
    lang_instruction = "Отвечай только на русском языке." if language == "Русский" else "Respond only in English."

    full_prompt = f"{lang_instruction}\n\nПроанализируй текст ниже. Выдели главную мысль и 3-5 ключевых моментов.\n\nТекст:\n{prompt}"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120
        )
        return response.json().get("response", "Ошибка: не удалось получить текст от AI")
    except Exception as e:
        return f"Ошибка связи с Ollama: {str(e)}"

# --- ТРАНСКРИБАЦИЯ ---
uploaded_file = st.file_uploader("Загрузите видео или аудио", type=['mov', 'mp4', 'mp3', 'wav', 'm4a'])

if uploaded_file:
    input_path = DATA_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 1. Начать транскрибацию", use_container_width=True):
        model = load_whisper()
        progress_bar = st.progress(0)

        segments, info = model.transcribe(str(input_path), language="ru")

        full_text = ""
        for s in segments:
            full_text += s.text.strip() + " "
            progress_bar.progress(min(s.end / info.duration, 1.0))

        st.session_state['text'] = full_text
        st.success("Текст успешно распознан")

# --- АНАЛИЗ ---
if 'text' in st.session_state:
    st.divider()

    col_text, col_settings = st.columns([2, 1])

    with col_text:
        text_area = st.text_area("Результат транскрибации:", value=st.session_state['text'], height=300)

    with col_settings:
        st.subheader("Настройки анализа")
        # Выбор языка для Ollama
        target_lang = st.selectbox("Язык ответа AI:", ["Русский", "Английский"])

        if st.button("🧠 2. Анализировать текст", use_container_width=True):
            # Используем обычный spinner, он выглядит аккуратнее
            with st.spinner('AI готовит саммари...'):
                result = query_ollama(text_area, target_lang)
                st.session_state['analysis'] = result

    if 'analysis' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Итоги анализа")
        st.info(st.session_state['analysis'])

        # Кнопка скачивания
        st.download_button(
            label="📥 Скачать результат анализа",
            data=st.session_state['analysis'],
            file_name=f"analysis_{target_lang}.txt",
            mime="text/plain"
        )