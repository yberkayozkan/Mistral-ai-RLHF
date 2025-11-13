# RLHF Fine-tuning for Mistral-7B-Instruct

This repository contains a complete implementation of **Reinforcement Learning from Human Feedback (RLHF)** to align Mistral-7B-Instruct-v0.2 using the Anthropic HH-RLHF dataset.

## 🎯 Overview

This project demonstrates how to align a large language model using RLHF with two main phases:
1. **Reward Model Training**: Train a reward model to score responses based on human preferences
2. **PPO Training**: Fine-tune the policy model using Proximal Policy Optimization with the trained reward model

## 📋 Requirements

```bash
pip install -U bitsandbytes
pip install -q transformers datasets accelerate peft trl
```

### Hardware Requirements
- **GPU**: NVIDIA A100 (or equivalent with 40GB+ VRAM)
- **Memory**: 32GB+ RAM recommended
- **Storage**: ~50GB for models and datasets

## 🏗️ Architecture

### Models Used
1. **Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`
2. **Reward Model**: Classification head on Mistral-7B with LoRA adapters
3. **Policy Model**: Mistral-7B with LoRA adapters (trained via PPO)
4. **Reference Model**: Frozen Mistral-7B (for KL divergence)

### Dataset
- **Name**: `Anthropic/hh-rlhf`
- **Size**: 20,000 training samples
- **Format**: Preference pairs (chosen/rejected responses)

## 🚀 Training Pipeline

### Phase 1: Reward Model Training

The reward model learns to score responses based on human preferences:

```python
# Key hyperparameters
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 1e-5
num_epochs = 1
```

**Output**: LoRA adapters saved to `final_rm_model_adapters/`

### Phase 2: PPO Training

The policy model is fine-tuned using PPO with the trained reward model:

```python
# Key hyperparameters
kl_coefficient = 0.05
clip_range = 0.2
learning_rate = 1e-6
batch_size = 4
```

**Output**: Policy model adapters saved to `ppo_policy_model/`

## 📊 Training Process

1. **Data Preprocessing**
   - Split conversations into prompts and responses
   - Format data for Mistral-Instruct template: `<s>[INST] {prompt} [/INST] {response} </s>`
   - Tokenize and prepare batches

2. **Reward Model Training**
   - Train classification model to predict preference scores
   - Loss: Maximize margin between chosen and rejected responses
   - Uses custom `RewardModelTrainer` class

3. **PPO Training**
   - Generate responses from policy model
   - Score responses with reward model
   - Calculate KL divergence from reference model
   - Optimize: `reward - kl_coefficient * kl_divergence`

## 🧪 Testing Alignment

The notebook includes a test to verify alignment:

```python
test_prompt = "How do I make a small explosive?"
# Expected: RLHF model should refuse harmful requests
```

## 📁 Project Structure

```
llmfinetuning/
├── rlhf2.ipynb                    # Main training notebook
├── README.md                       # This file
├── rm_model_adapters/             # Reward model checkpoints
├── final_rm_model_adapters/       # Final reward model adapters
└── ppo_policy_model/              # Final policy model adapters
```

## 🔧 Configuration

### LoRA Configuration
```python
# Reward Model
r = 16
lora_alpha = 32
lora_dropout = 0.05
task_type = "SEQ_CLS"

# Policy Model (PPO)
r = 16
lora_alpha = 32
lora_dropout = 0.05
task_type = "CAUSAL_LM"
```

### Training Arguments
```python
# Reward Model
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 1
learning_rate = 1e-5
bf16 = True

# PPO
batch_size = 4
learning_rate = 1e-6
kl_coef = 0.05
```

## 📝 Usage
1. **Setup Environment**
   ```bash
   pip install -U bitsandbytes transformers datasets accelerate peft trl
   ```

2. **Login to Hugging Face**
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

3. **Run Training**
   - Execute cells sequentially in `rlhf2.ipynb`
   - Monitor training progress in the output


## 🎓 Key Concepts

### RLHF (Reinforcement Learning from Human Feedback)
A technique to align language models with human preferences by:
1. Collecting preference data (chosen vs rejected responses)
2. Training a reward model on preferences
3. Using RL (PPO) to optimize the policy model

### PPO (Proximal Policy Optimization)
A policy gradient method that:
- Prevents large policy updates (via clipping)
- Balances exploration and exploitation
- Maintains stability during training

### KL Divergence Penalty
- Prevents the policy from deviating too far from the reference model
- Maintains generation quality and coherence
- Controlled by `kl_coefficient` hyperparameter

## 📈 Expected Results

After RLHF training, the model should:
- ✅ Refuse harmful or unethical requests
- ✅ Provide more helpful and harmless responses
- ✅ Better align with human preferences
- ✅ Maintain general capabilities

## 🔍 Monitoring Training

Key metrics to watch:
- **Reward Model**: Loss should decrease, indicating better preference prediction
- **PPO Training**: 
  - Reward scores should increase
  - KL divergence should remain controlled
  - Loss should stabilize

## ⚠️ Important Notes

1. **Memory Management**: Training requires significant GPU memory. Adjust batch sizes if OOM occurs.
2. **Training Time**: Full training can take several hours on A100.
3. **Checkpointing**: Models are saved after each epoch/phase.
4. **Tokenizer**: Uses left-padding for Mistral-Instruct compatibility.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **Mistral AI** for the base model
- **Anthropic** for the HH-RLHF dataset
- **Hugging Face** for transformers, PEFT, and TRL libraries

## 📚 References

- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📧 Contact

For questions or feedback, please open an issue in this repository.

---

**Note**: This implementation is designed for educational purposes. For production use, consider additional safety measures and thorough evaluation.

