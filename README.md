# LumiRAG-IR: A Unified Multimodal RAG Large Model Bridging Text and Image Retrieval

## 1. Introduction

This document presents the implementation details of the paper LumiRAG-IR: A Unified Multimodal RAG Large Model Bridging Text and Image Retrieval.

## 2. Training Data
The training data used in the paper will be released on Hugging Face.

## 3. Training Scripts

### 3.1 Instruction Tuning

For the code of the Instruction Tuning phase, please refer to [https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md](https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md).

### 3.2 Reinforcement Learning

```bash
cd ./rlhf/verl
bash recipe/dapo/run_dapo_qwen2.5_3b_rag.sh
```
For more information, please refer to [https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md](https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md).
