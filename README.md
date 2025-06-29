# End‑to‑end Medical Chatbot using LangChain

## How to run?

### STEPS:

1. Clone the repository:
```bash
git clone https://github.com/Tushar7012/End-to-End-Medical-Chatbot
cd End-to-end-Medical-Chatbot-using-LangChain
```

2. Create a conda environment:
```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Pinecone credentials:
```env
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

5. Download the quantized LLaMA‑2 model and place it inside the `model/` directory:
```
llama-2-7b-chat.ggmlv3.q4_0.bin
```
(You can get the file from the Hugging Face link provided in the repo.)

6. Build and store the vector index:
```bash
python store_index.py
```

7. Run the application:
```bash
python app.py
```
Navigate to: `http://localhost:5000` in your browser.

## Tech Stack Used
- Python  
- LangChain  
- Flask  
- Pinecone

## License
[MIT](LICENSE)
