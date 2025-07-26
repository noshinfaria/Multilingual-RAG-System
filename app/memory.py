from collections import deque

class ChatMemory:
    def __init__(self, max_len=5):
        self.history = deque(maxlen=max_len)
    
    def add(self, user, bot):
        self.history.append({"user": user, "bot": bot})
    
    def get(self):
        return [f"User: {msg['user']}\nBot: {msg['bot']}" for msg in self.history]
