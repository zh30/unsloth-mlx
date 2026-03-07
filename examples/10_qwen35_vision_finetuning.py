"""
Qwen3.5 Vision Fine-Tuning with MLX-Tune

This example mirrors Unsloth's Qwen3.5 Vision notebook:
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(0_8B)_Vision.ipynb

Instead of CUDA/Triton, this runs natively on Apple Silicon using MLX.
Just change the import line to switch between Unsloth and MLX-Tune!

Usage:
    python examples/10_qwen35_vision_finetuning.py
"""

# ===========================================================================
# MLX-Tune imports (Unsloth equivalent)
# ===========================================================================
# Unsloth (CUDA):
#   from unsloth import FastVisionModel
#   from unsloth.trainer import UnslothVisionDataCollator
#   from trl import SFTTrainer, SFTConfig
#
# MLX-Tune (Apple Silicon) - SAME API:
from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig

# ===========================================================================
# Step 1: Load the model
# ===========================================================================
print("=" * 70)
print("Step 1: Loading Qwen3.5-0.8B Vision Model")
print("=" * 70)

model, processor = FastVisionModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-bf16",
    load_in_4bit=False,  # Use False for 16bit LoRA (better quality)
    use_gradient_checkpointing="unsloth",  # For long context
)

# ===========================================================================
# Step 2: Add LoRA adapters
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters")
print("=" * 70)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Fine-tune vision layers
    finetune_language_layers=True,     # Fine-tune language layers
    finetune_attention_modules=True,   # Fine-tune attention
    finetune_mlp_modules=True,         # Fine-tune MLP
    r=16,                              # LoRA rank
    lora_alpha=16,                     # Recommended: alpha == r
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ===========================================================================
# Step 3: Prepare the dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Preparing Dataset")
print("=" * 70)

from datasets import load_dataset

dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
print(f"Dataset loaded: {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
print(f"Sample text: {dataset[2]['text'][:100]}...")

# Format dataset for vision fine-tuning
# All vision tasks should use this format:
# [
#   {"role": "user", "content": [{"type": "text", "text": Q}, {"type": "image", "image": img}]},
#   {"role": "assistant", "content": [{"type": "text", "text": A}]},
# ]

instruction = "Write the LaTeX representation for this image."


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
    return {"messages": conversation}


converted_dataset = [convert_to_conversation(sample) for sample in dataset]
print(f"Converted {len(converted_dataset)} samples to conversation format")

# ===========================================================================
# Step 4: Test inference BEFORE training
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference Test")
print("=" * 70)

FastVisionModel.for_inference(model)

image = dataset[2]["image"]
prompt = "Write the LaTeX representation for this image."

print(f"Prompt: {prompt}")
print("Generating response (before training)...")

try:
    response = model.generate(
        prompt=prompt,
        image=image,
        max_tokens=128,
        temperature=1.5,
    )
    print(f"Response: {response}")
except Exception as e:
    print(f"Pre-training inference error (expected for some models): {e}")

# ===========================================================================
# Step 5: Train the model
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: Training")
print("=" * 70)

FastVisionModel.for_training(model)

trainer = VLMSFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=converted_dataset,
    args=VLMSFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adam",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining metrics: {trainer_stats.metrics}")

# ===========================================================================
# Step 6: Test inference AFTER training
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Inference Test")
print("=" * 70)

FastVisionModel.for_inference(model)

image = dataset[2]["image"]
prompt = "Write the LaTeX representation for this image."

print(f"Prompt: {prompt}")
print("Generating response (after training)...")

try:
    response = model.generate(
        prompt=prompt,
        image=image,
        max_tokens=128,
        temperature=1.5,
    )
    print(f"Response: {response}")
except Exception as e:
    print(f"Post-training inference error: {e}")

# ===========================================================================
# Step 7: Save the model
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Model")
print("=" * 70)

# Save LoRA adapters only (recommended for sharing)
model.save_pretrained("qwen_lora")
print("LoRA adapters saved to qwen_lora/")

# Uncomment to save merged model (larger but self-contained)
# model.save_pretrained_merged("qwen_merged", processor)

print("\n" + "=" * 70)
print("Done! Vision fine-tuning complete.")
print("=" * 70)
