import json

import requests
import streamlit as st
import yaml

from rag import DEFAULT_CONFIG, RagPipeline


st.set_page_config(page_title="Lab2 — RAG", layout="wide")
st.title("Вопросы по документам")

config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
questions_path = DEFAULT_CONFIG.parent / config["evaluation"]["questions"]
questions = json.loads(questions_path.read_text(encoding="utf-8"))
questions_by_id = {question["id"]: question for question in questions}


def clear_result():
    st.session_state.pop("result", None)


def select_question():
    question_id = st.session_state["example_id"]
    st.session_state["question"] = (
        questions_by_id[question_id]["question"]
        if question_id is not None
        else ""
    )
    clear_result()


st.selectbox(
    "Готовый вопрос",
    options=[None, *questions_by_id],
    format_func=lambda question_id: (
        f"{question_id}: {questions_by_id[question_id]['question']}"
        if question_id is not None
        else "Свой вопрос"
    ),
    key="example_id",
    on_change=select_question,
)

question = st.text_area(
    "Вопрос",
    key="question",
    height=120,
    on_change=clear_result,
)

if st.button("Получить ответ", disabled=not question.strip()):
    clear_result()

    try:
        with st.spinner("Ищем фрагменты и готовим ответ…"):
            if "pipeline" not in st.session_state:
                st.session_state["pipeline"] = RagPipeline()

            st.session_state["result"] = st.session_state["pipeline"].ask(
                question.strip()
            )
    except requests.RequestException as error:
        st.error(f"Не удалось получить ответ от Ollama: {error}")

result = st.session_state.get("result")

if result is not None:
    st.subheader("Ответ")
    st.markdown(result["answer"])
    st.caption(f"Поиск и генерация: {result['elapsed_seconds']:.1f} с")

    if result["done_reason"] == "length":
        st.warning("Достигнут лимит генерации: ответ мог оборваться.")

    st.subheader("Найденные фрагменты")
    for source in result["sources"]:
        title = (
            f"[{source['rank']}] {source['source']} — "
            f"PDF-страница {source['page']}"
        )
        with st.expander(title):
            st.caption(f"Squared L2: {source['squared_l2']:.4f}")
            st.text(source["text"])

reference = next(
    (item for item in questions if item["question"] == question.strip()),
    None,
)

if reference is not None:
    with st.expander("Эталон для сравнения"):
        st.write(reference["answer"])

        for source in reference["references"]:
            st.write(f"{source['source']} — PDF-страница {source['page']}")
            for quote in source.get("quotes", []):
                st.text(quote)