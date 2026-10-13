from __future__ import annotations

from typing import Dict, List


class ConversationManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append({'role': 'user', 'content': message})

    def add_assistant_message(self, message: str):
        self.history.append({'role': 'assistant', 'content': message})

    def rewrite_follow_up(self, current_question: str, prior_question: str | None = None) -> str:
        if not prior_question:
            return current_question
        lower = current_question.lower()
        if any(token in lower for token in ['it', 'they', 'this', 'that', 'those', 'these', 'he', 'she']):
            return f"Using the earlier context about '{prior_question}', answer: {current_question}"
        return current_question

    def clear(self):
        self.history.clear()
