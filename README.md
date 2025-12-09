# Language Model Server Demo on Jetson Orin

A lightweight FastAPI server for running local language models — optimized for the **NVIDIA Jetson AGX Orin**, but compatible with **any device** (with or without CUDA/GPU support).

---

## Overview
- **Language:** Python  
- **Main file:** `server.py`  
- **Device:** NVIDIA Jetson AGX Orin (or any CUDA-capable device for GPU acceleration; CPU fallback supported)  
- **Model:** Any small-to-medium language model from [Hugging Face](https://huggingface.co/) or a locally stored model
- Model used for initial implementation (A finetuned and quantized version of OPT. The model was finetuned on a publicly available Tesla car manual PDF.): https://huggingface.co/bladebreaker17/finetuned_quantized_model_for_server_test/tree/main

By default, the model directory should be named `finetuned`, but you can change this in `server.py`.

---

## Getting Started

### Prepare Your Model
Place your language model in the same directory as `server.py`.  
You can:
- Rename the model folder to **`finetuned`**, **or**
- Update the model path inside `server.py` to match your folder name.

---

### Install Dependencies
Make sure you have the following Python modules installed:

```bash
pip install torch transformers fastapi uvicorn
````

> Most users already have `torch` and `transformers`; you may only need to install `fastapi` and `uvicorn`.

---

### Find Your Device’s IP Address

Run the following command in a terminal:

```bash
hostname -I
```

---

### Start the Server

In the same folder as `server.py`, run:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

You should see confirmation that your FastAPI server is live.

---

### Access the Server

From any device on the same network (phone, laptop, tablet, etc.), open a browser and go to:

```
http://<your_device_ip>:8000/
```

**Example:**

```
http://192.168.1.42:8000/
```

You can now chat with your local language model through the web interface.

---

## Performance Notes

* **GPU (CUDA):** ~4 seconds per response
* **CPU only (Jetson Orin CPU):** ~30 seconds per response

> For best results, run the server on a device with CUDA-enabled GPU support.

---

## Example Use Cases

* On-device AI chat assistant
* Offline LLM experimentation
* Local inference testing for finetuned models
