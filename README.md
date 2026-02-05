# LumiRAG-IR: A Unified Multimodal RAG Large Model Bridging Text and Image Retrieval

## 1. Introduction

This document presents the implementation details of the paper LumiRAG-IR: A Unified Multimodal RAG Large Model Bridging Text and Image Retrieval.

## 2. Training Data
The training data used in the paper will be released on Hugging Face.

## 3. Training Scripts

For the code of the Instruction Tuning phase, please refer to [https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md](https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md).

### Reinforcement Learning

```bash
# Execute the DAPO training script
cd ./rlhf/verl
bash recipe/dapo/run_dapo_qwen_3b_rag.sh
```
For more information, please refer to [https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md](https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/rlhf/docs/RL_training.md).
