// nvm use v18.9.0

import { Ollama } from 'langchain/llms/ollama';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import '@tensorflow/tfjs-node';
import { TensorFlowEmbeddings } from 'langchain/embeddings/tensorflow';
import { RetrievalQAChain } from 'langchain/chains';

const ollama = new Ollama({
  baseUrl: 'http://localhost:11434',
  // model: "mistral-openorca:latest",
  model: "zephyr",
});

const loader = new CheerioWebBaseLoader(
  'https://gist.githubusercontent.com/dongyuwei/42626b4bded6ac16fccb4234154a0a05/raw/47f7ac7e94e97585b631a746a97552a0b0e8c258/PIME.md'
);
const data = await loader.load();
console.log('data:', data);

// Split the text into 500 character chunks. And overlap each chunk by 20 characters
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20,
});
const splitDocs = await textSplitter.splitDocuments(data);

// Then use the TensorFlow Embedding to store these chunks in the datastore
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  new TensorFlowEmbeddings()
);

const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const result = await chain.call({
  query: "Where is PIME's log?",
});
console.log(result.text);
// `mistral-openorca:latest`：
//  The logs for PIME can be found in the following location: C:\Users\\AppData\Local\PIME\Log.

// `zephyr`:
// The logs for PIME can be found in the following location on a Windows 11 machine: `C:\Users\<username>\AppData\Local\PIME\Log` (where `<username>` is replaced with your own username).

