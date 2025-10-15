from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Pick a small model for Jetson Nano
MODEL_NAME = "finetuned"  # or any other model you want to use it with
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Jetson Chat</title>
      <style>
        body { font-family: sans-serif; margin: 20px; }
        #chat { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; }
        .msg { margin: 5px 0; }
        .user { font-weight: bold; color: blue; }
        .bot { font-weight: bold; color: green; }
      </style>
    </head>
    <body>
      <h1>Chat with Jetson</h1>
      <div id="chat"></div>
      <input id="input" type="text" placeholder="Type a message..." style="width:80%">
      <button onclick="sendMessage()">Send</button>

      <script>
        const chatBox = document.getElementById("chat");

        function addMessage(sender, text) {
          const msg = document.createElement("div");
          msg.className = "msg " + sender;
          msg.innerHTML = `<span class="${sender}">${sender}:</span> ${text}`;
          chatBox.appendChild(msg);
          chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendMessage() {
          const input = document.getElementById("input");
          const userText = input.value;
          input.value = "";
          addMessage("user", userText);

          const res = await fetch("/generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({prompt: userText})
          });
          const data = await res.json();
          addMessage("bot", data.response);
        }
      </script>
    </body>
    </html>
    """

@app.post("/generate")
def generate(request: Request):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens or 200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {"response": text}

