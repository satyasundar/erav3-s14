## Docker deployment of Models

1. Build the docker image for the server
```
docker build -t server-image:latest -f docker-server/Dockerfile .
```

2. Test the model server from command line
```
curl -X POST "http://localhost:8000/generate" \
-H "Content-Type: application/json" \
-d '{"prompt": "Hello, world!", "max_length": 50, "temperature": 0.7}'
```

3. Build the docker image for the client
```
docker build -t client-image:latest -f docker-client-ui/Dockerfile .
```

3. Make a docker network
```
docker network create my-network
```

3. Run the server
```
docker run -d -p 8000:8000 --name model-container --network my-network server-image:latest
```

4. Run the client
```
docker run -d -p 8080:8080 --name client-container --network my-network client-image:latest
```

5. Access the client UI at http://localhost:8080    
6. Send a request to the server at http://localhost:8000/generate    
7. The server will return a response with the generated text
