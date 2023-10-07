import uuid
from flask import Flask, request, Response
import json
import requests

from .src.models.dto.ollama import OllamaParams, OllamaOptions
from .src.models.dto.openai import OpenAIResponse, OpenAIErrorResponse, Choice, Delta

app = Flask(__name__)


@app.route("/chat/completions", methods=["GET", "POST"])
def completions():
    openai_received_params = request.get_json()

    system = ""
    user = ""
    for message in openai_received_params.get("messages", []):
        if message.get("role") == "system":
            system = message.get("content", "")
        elif message.get("role") == "user":
            user = message.get("content", "")

    ollama_params = OllamaParams(
        openai_received_params["model"],
        user,
        OllamaOptions(temperature=openai_received_params.get("temperature")),
        system=system,
        template=openai_received_params.get("template", ""),
    )

    data = ollama_params.generate_post_data()

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        stream=True,
        json=data,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )

    if response.status_code == 400:
        error = OpenAIErrorResponse.parse({
                "error": {
                    "message": response.json().get("error"),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            })
        return error.to_json(), 400

    return Response(
        generate_completion(response, ollama_params), mimetype="text/event-stream"
    )


def generate_completion(response: requests.Response, ollama_params: OllamaParams):
    completion_id = str(uuid.uuid4())
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            openai_response = OpenAIResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=chunk.get("created_at"),
                model=ollama_params.model,
                choices=[
                    Choice(
                        index=0,
                        delta=Delta(content=chunk.get("response", "")),
                        finished_reason="stop" if chunk.get("done") else None,
                    ),
                ],
            )
            yield f"data: {openai_response.to_json()}\n\n"
    return "data: [DONE]"


if __name__ == "__main__":
    app.debug = True
    app.run(threaded=True)
