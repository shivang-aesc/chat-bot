# FAQ Bot API

A FastAPI-based FAQ bot that uses OpenAI embeddings for question matching. This bot can handle a large set of FAQs and provide accurate answers based on semantic similarity.

## Features

- Add and manage FAQs through API endpoints
- Automatic embedding generation using OpenAI's API
- Semantic question matching using cosine similarity
- Swagger documentation
- Comprehensive logging
- Postman collection for easy testing

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Running the Application

Start the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Add FAQs
- **POST** `/faqs`
- Adds new FAQs to the system and generates embeddings
- Request body:
```json
{
    "faqs": [
        {
            "question": "How do I reset my password?",
            "answer": "Go to settings > account > reset password."
        }
    ]
}
```

### Ask Question
- **POST** `/ask`
- Ask a question and get the most relevant answer
- Request body:
```json
{
    "text": "How can I reset my password?"
}
```

## Postman Collection

A Postman collection is included in the repository (`FAQ_Bot_API.postman_collection.json`). You can import this into Postman to test the API endpoints.

## Logging

Logs are stored in the `logs` directory. The application uses the `loguru` library for comprehensive logging.

## Data Storage

- FAQs are stored in `data/faqs.json`
- Embeddings are stored in `data/faq_embeddings.npy` 