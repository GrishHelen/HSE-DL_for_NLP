import reflex as rx
from text_sugg.classes import TextSuggestion
import pickle


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = "text_sugg.classes"
        return super().find_class(module, name)


def make_text_suggestor():
    with open("models/text_suggestion.pickle", "rb") as file:
        text_suggestion = MyCustomUnpickler(file).load()
    return text_suggestion


text_suggestion = make_text_suggestor()


class TextSuggestionState(rx.State):
    current_text: str = ""
    suggestions: list[str] = []
    active_suggestion_index: int = 0
    show_suggestions: bool = False

    def on_mount(self):
        pass

    def update_suggestions(self):
        """Обновление подсказок при изменении текста"""
        if not self.current_text.strip():
            self.show_suggestions = False
            self.suggestions = []
            return

        words = self.current_text.split()
        if words:
            suggs = text_suggestion.suggest_text(
                self.current_text,
                n_words=3,
                n_texts=5,
                max_complete_words=3,
                n_best_words=2,
            )
            self.suggestions = [" ".join(sugg) for sugg in suggs]
            print("???", self.suggestions)
            self.show_suggestions = bool(len(self.suggestions) > 0)
            self.active_suggestion_index = 0

    def set_current_text(self, text: str):
        """Установка текста с обновлением подсказок"""
        self.current_text = "".join(text.split("\n"))
        print("!!!", text)
        self.update_suggestions()

    def handle_key_down(self, key):
        """Обработка клавиш для навигации по подсказкам"""
        print("Pressed key:", key)

        if key in ["ArrowDown", "ArrowRight"] and self.show_suggestions:
            self.active_suggestion_index = min(
                self.active_suggestion_index + 1, len(self.suggestions) - 1
            )
        elif key in ["ArrowUp", "ArrowLeft"] and self.show_suggestions:
            self.active_suggestion_index = max(self.active_suggestion_index - 1, 0)
        elif key in ["Enter", "Tab"] and self.show_suggestions:
            self.select_suggestion(self.suggestions[self.active_suggestion_index])
        elif key == "Escape":
            self.show_suggestions = False

    def select_suggestion(self, suggestion: str):
        """Выбор подсказки"""
        text = self.current_text
        if not text:
            return

        words = text.split()
        if words:
            words[-1] = suggestion
            self.current_text = " ".join(words) + " "

        self.show_suggestions = False
        self.update_suggestions()


def header() -> rx.Component:
    """The header and the description."""
    return rx.box(
        rx.heading("Text Suggestor", size="8"),
        rx.text(
            "Write some text, and the model will suggest continuation",
            color_scheme="gray",
        ),
        margin_bottom="1rem",
    )


def suggestion_button(suggestion, index, active_index, on_click):
    return rx.button(
        suggestion,
        on_click=on_click,
        bg=rx.cond(active_index == index, "blue.500", "gray.100"),
        color=rx.cond(active_index == index, "white", "black"),
        _hover={"bg": rx.cond(active_index == index, "blue.600", "gray.200")},
        width="100%",
        text_align="left",
        padding="0.5rem 1rem",
    )


def text_input_with_suggestions():
    return rx.box(
        rx.vstack(
            rx.text_area(
                value=TextSuggestionState.current_text,
                on_key_down=TextSuggestionState.handle_key_down,
                on_change=TextSuggestionState.set_current_text,
                placeholder="Начните вводить текст...",
                width="100%",
                min_height="100px",
            ),
            rx.cond(
                TextSuggestionState.show_suggestions,
                rx.card(
                    rx.vstack(
                        rx.text("Подсказки:", font_weight="bold"),
                        rx.foreach(
                            TextSuggestionState.suggestions,
                            lambda suggestion, i: suggestion_button(
                                suggestion,
                                i,
                                TextSuggestionState.active_suggestion_index,
                                lambda: TextSuggestionState.select_suggestion(
                                    suggestion
                                ),
                            ),
                        ),
                        width="100%",
                    ),
                    width="100%",
                    margin_top="1rem",
                ),
            ),
            width="100%",
            align_items="start",
        ),
        width="100%",
        max_width="600px",
        margin="0 auto",
    )


def index():
    """The main view."""
    return rx.container(
        header(),
        text_input_with_suggestions(),
        padding="2rem",
        max_width="600px",
        margin="auto",
    )


app = rx.App(
    theme=rx.theme(
        appearance="light", has_background=True, radius="large", accent_color="blue"
    ),
)
app.add_page(index, title="Translator")
