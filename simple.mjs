import { Ollama } from "langchain/llms/ollama";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "zephyr",
});

const answer = await ollama.call(`why is the sky blue?`);

console.log(answer);