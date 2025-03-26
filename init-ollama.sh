#!/bin/bash

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags > /dev/null && [ $attempt -lt $max_attempts ]; do
    echo "Attempt $((attempt+1))/$max_attempts: Server not ready yet..."
    sleep 5
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: Server failed to start after $max_attempts attempts"
    exit 1
fi

echo "Ollama server is ready. Pulling model..."
ollama pull llama3.1:8b

echo "Verifying model was pulled..."
ollama list
if [ $? -ne 0 ]; then
    echo "Error: Model verification failed"
    exit 1
fi

echo "Setup completed successfully. Keeping server running..."
# Wait for the server process
wait $SERVER_PID 