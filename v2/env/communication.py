from typing import List, Dict
from v2.agents.observation_space import Message

class MessageBus:
    def __init__(self):
        self.messages: List[Message] = []
        self.step_counter = 0

    def add_message(self, sender: str, content: str):
        if content and content.strip():
            self.messages.append(Message(sender=sender, content=content, step=self.step_counter))

    def get_messages(self) -> List[Message]:
        return self.messages

    def step(self):
        self.step_counter += 1

    def reset(self):
        self.messages = []
        self.step_counter = 0
