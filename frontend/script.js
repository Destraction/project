const sessionId = "session_" + Math.random().toString(36).slice(2, 10);
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

const messagesDiv = document.getElementById("messages");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

function addMessage(text, type) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.innerText = text;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

ws.onopen = () => {
  addMessage("Подключено к серверу. Можно писать сообщение.", "system");
};

ws.onclose = () => {
  addMessage("Соединение закрыто.", "system");
};

ws.onerror = () => {
  addMessage("Ошибка соединения.", "system");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "system") addMessage(data.content, "system");
  else if (data.type === "recommendation") addMessage(data.message, "assistant");
  else if (data.type === "success") addMessage(data.content, "system");
  else if (data.type === "error") addMessage(data.content, "system");
  else if (data.content) addMessage(data.content, "assistant");
  else if (data.message) addMessage(data.message, "assistant");
};

function sendUserMessage() {
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  ws.send(JSON.stringify({ message: text }));
  messageInput.value = "";
}

sendBtn.onclick = sendUserMessage;
messageInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendUserMessage();
});

