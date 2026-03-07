"""
Vision Language Model (VLM) Support for MLX-Tune

Provides Unsloth-compatible API for Vision-Language models using mlx-vlm:
- Qwen3.5 (natively multimodal)
- Qwen2-VL / Qwen3-VL
- LLaVA 1.5 / 1.6
- Pixtral
- PaliGemma
- Gemma 3
- And other VLMs supported by mlx-vlm

Usage (matches Unsloth exactly):
    from mlx_tune import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-bf16",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        r=16, lora_alpha=16,
    )
"""

from typing import Optional, Any, List, Dict, Union, Tuple
from pathlib import Path
import warnings
import json


# Check for mlx-vlm availability
try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import stream_generate as vlm_stream_generate
    from mlx_vlm.utils import prepare_inputs, load_image_processor
    from mlx_vlm.trainer.utils import (
        get_peft_model as vlm_get_peft_model,
        find_all_linear_names,
        apply_lora_layers,
        freeze_model,
    )
    from mlx_vlm.trainer.trainer import (
        Trainer as VLMTrainerInternal,
        Dataset as VLMDataset,
        save_adapter,
    )
    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False


def _require_mlx_vlm():
    if not HAS_MLX_VLM:
        raise ImportError(
            "mlx-vlm is required for vision model support. "
            "Install with: uv pip install mlx-vlm"
        )


class FastVisionModel:
    """
    Unsloth-compatible API for Vision Language Models on Apple Silicon.

    Provides the same API patterns as Unsloth's FastVisionModel but uses
    mlx-vlm under the hood for Apple Silicon optimization.

    Example:
        >>> from mlx_tune import FastVisionModel
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     "mlx-community/Qwen3.5-0.8B-bf16",
        ... )
        >>> model = FastVisionModel.get_peft_model(model, r=16)
        >>> FastVisionModel.for_training(model)
        >>> # ... train ...
        >>> FastVisionModel.for_inference(model)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        use_gradient_checkpointing: Union[bool, str] = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a pretrained Vision Language Model.

        Args:
            model_name: HuggingFace model ID (e.g., "mlx-community/Qwen3.5-0.8B-bf16")
            max_seq_length: Maximum sequence length
            dtype: Data type (auto-selected by MLX)
            load_in_4bit: Whether to use 4-bit quantized model
            use_gradient_checkpointing: Enable gradient checkpointing ("unsloth" or True)
            **kwargs: Additional arguments passed to mlx_vlm.load()

        Returns:
            Tuple of (VLMModelWrapper, processor)
        """
        _require_mlx_vlm()

        print(f"Loading VLM: {model_name}")

        model, processor = vlm_load(model_name, **kwargs)

        # Load image processor if available
        image_processor = None
        try:
            image_processor = load_image_processor(model_name)
        except Exception:
            pass

        wrapped = VLMModelWrapper(
            model=model,
            processor=processor,
            image_processor=image_processor,
            max_seq_length=max_seq_length or 2048,
            model_name=model_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        return wrapped, processor

    @staticmethod
    def get_peft_model(
        model: "VLMModelWrapper",
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Any] = None,
        **kwargs,
    ) -> "VLMModelWrapper":
        """
        Add LoRA adapters to a VLM for fine-tuning (Unsloth-compatible API).

        Args:
            model: VLMModelWrapper from from_pretrained()
            finetune_vision_layers: Whether to fine-tune vision encoder layers
            finetune_language_layers: Whether to fine-tune language model layers
            finetune_attention_modules: Whether to fine-tune attention layers
            finetune_mlp_modules: Whether to fine-tune MLP layers
            r: LoRA rank
            target_modules: Specific target modules (auto-detected if None)
            lora_alpha: LoRA scaling parameter
            lora_dropout: LoRA dropout rate
            bias: Bias training mode ("none", "all", "lora_only")
            random_state: Random seed
            use_rslora: Use rank-stabilized LoRA
            loftq_config: LoftQ configuration

        Returns:
            Model with LoRA adapters configured (applied on first train call)
        """
        _require_mlx_vlm()

        # Store LoRA config for deferred application
        model.lora_config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "finetune_vision_layers": finetune_vision_layers,
            "finetune_language_layers": finetune_language_layers,
            "finetune_attention_modules": finetune_attention_modules,
            "finetune_mlp_modules": finetune_mlp_modules,
            "target_modules": target_modules,
        }
        model.lora_enabled = True

        # Determine which linear layers to target
        if target_modules is not None:
            linear_layers = target_modules
        else:
            linear_layers = _get_target_modules(
                model.model,
                finetune_attention_modules=finetune_attention_modules,
                finetune_mlp_modules=finetune_mlp_modules,
            )

        # Compute alpha as scale factor (mlx-vlm uses alpha as scale = alpha/rank)
        alpha_scale = lora_alpha / r if r > 0 else 1.0

        # Actually apply LoRA now using mlx-vlm's get_peft_model
        model.model = vlm_get_peft_model(
            model.model,
            linear_layers,
            rank=r,
            alpha=alpha_scale,
            dropout=lora_dropout,
            freeze=True,
            verbose=True,
        )
        model._lora_applied = True
        model._linear_layers = linear_layers

        print(f"LoRA configured: r={r}, alpha={lora_alpha}, "
              f"modules={linear_layers}")

        return model

    @staticmethod
    def for_training(model: "VLMModelWrapper") -> "VLMModelWrapper":
        """Enable training mode for the VLM."""
        model.inference_mode = False
        model.model.train()
        return model

    @staticmethod
    def for_inference(model: "VLMModelWrapper") -> "VLMModelWrapper":
        """Enable inference mode for the VLM."""
        model.inference_mode = True
        model.model.eval()
        return model


def _get_target_modules(
    model,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
) -> List[str]:
    """Auto-detect target modules based on finetune flags."""
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_modules = ["gate_proj", "up_proj", "down_proj"]

    target = []
    if finetune_attention_modules:
        target.extend(attention_modules)
    if finetune_mlp_modules:
        target.extend(mlp_modules)

    if not target:
        # Fallback: find all linear layers
        try:
            target = find_all_linear_names(model.language_model)
        except Exception:
            target = attention_modules + mlp_modules

    return target


class VLMModelWrapper:
    """
    Wrapper around mlx-vlm models providing Unsloth-compatible interface.

    Handles LoRA application, training, inference, and model saving.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        image_processor: Any = None,
        max_seq_length: int = 2048,
        model_name: Optional[str] = None,
        use_gradient_checkpointing: Union[bool, str] = False,
    ):
        self.model = model
        self.processor = processor
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # LoRA state
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False
        self._linear_layers = None
        self._adapter_path: Optional[Path] = None

        # Mode
        self.inference_mode = False

    @property
    def config(self):
        """Access the underlying model config."""
        return self.model.config

    def generate(
        self,
        prompt: Optional[str] = None,
        image: Optional[Any] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate response for image+text input.

        Args:
            prompt: Text prompt
            image: Image path(s) or PIL Image
            image_path: Alternative path to image file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Print generation details

        Returns:
            Generated text response
        """
        _require_mlx_vlm()

        # Handle image_path as alias
        if image_path and image is None:
            image = image_path

        result = vlm_generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=image,
            verbose=verbose,
            max_tokens=max_tokens,
            temp=temperature,
            **kwargs,
        )

        # GenerationResult has .text attribute
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def stream_generate(self, prompt: str, image: Optional[Any] = None, **kwargs):
        """Stream-generate response for image+text input."""
        _require_mlx_vlm()
        return vlm_stream_generate(
            self.model, self.processor,
            prompt=prompt, image=image, **kwargs,
        )

    def save_pretrained(self, output_dir: str, **kwargs):
        """Save LoRA adapters."""
        _require_mlx_vlm()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._lora_applied:
            adapter_file = output_dir / "adapters.safetensors"
            save_adapter(self.model, str(adapter_file))
            print(f"Adapters saved to {output_dir}")
        else:
            print("No LoRA adapters to save. Train the model first.")

    def save_pretrained_merged(
        self,
        output_dir: str,
        tokenizer_or_processor: Any = None,
        save_method: str = "merged_16bit",
        **kwargs,
    ):
        """Save merged model (base + LoRA fused)."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        actual_model = self.model

        # Fuse LoRA layers if applied
        if self._lora_applied:
            from mlx_vlm.trainer.lora import LoRaLayer
            fused_count = 0
            for name, module in actual_model.named_modules():
                if isinstance(module, LoRaLayer):
                    # LoRaLayer has original_layer, A, B, alpha
                    original = module.original_layer
                    # Compute fused weight: W + alpha * B^T @ A^T
                    lora_weight = (module.B.T @ module.A.T) * module.alpha
                    if isinstance(original, nn.QuantizedLinear):
                        warnings.warn(
                            f"Cannot fuse LoRA into quantized layer {name}. "
                            "Use a non-quantized base model for merged saving."
                        )
                        continue
                    original.weight = original.weight + lora_weight
                    # Replace LoRA layer with original
                    parts = name.split(".")
                    parent = actual_model
                    for p in parts[:-1]:
                        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                    setattr(parent, parts[-1], original)
                    fused_count += 1
            if fused_count:
                print(f"Fused {fused_count} LoRA layers into base model")

        # Save weights
        weights = dict(tree_flatten(actual_model.parameters()))
        from mlx_lm.utils import save_model
        save_model(str(output_dir), actual_model)

        # Save processor/tokenizer
        if tokenizer_or_processor is not None:
            tokenizer_or_processor.save_pretrained(str(output_dir))

        # Save config
        if hasattr(actual_model, "config"):
            config_dict = actual_model.config.__dict__ if hasattr(actual_model.config, "__dict__") else actual_model.config
            with open(output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

        print(f"Merged model saved to {output_dir}")

    def save_pretrained_gguf(
        self,
        output_dir: str,
        tokenizer_or_processor: Any = None,
        quantization_method: str = "q8_0",
        **kwargs,
    ):
        """
        Save model in GGUF format.

        Note: VLM GGUF export has limited support. The language model
        portion can be exported, but vision components may not be included.
        """
        warnings.warn(
            "GGUF export for VLMs is experimental. Only the language model "
            "portion will be exported. For full VLM deployment, use "
            "save_pretrained() or save_pretrained_merged() instead.",
            UserWarning,
        )
        # Save merged first, then attempt GGUF conversion
        merged_dir = Path(output_dir) / "_merged_tmp"
        self.save_pretrained_merged(str(merged_dir), tokenizer_or_processor)

        from mlx_tune.trainer import export_to_gguf
        export_to_gguf(
            str(merged_dir),
            output_path=str(Path(output_dir) / "model.gguf"),
            quantization=quantization_method,
        )

    def set_adapter_path(self, path: str):
        """Set the adapter save/load path."""
        self._adapter_path = Path(path)

    def get_adapter_path(self) -> Optional[Path]:
        """Get the current adapter path."""
        return self._adapter_path

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Avoid infinite recursion for attributes accessed during __init__
        if name in ("model", "processor", "lora_config", "lora_enabled",
                     "_lora_applied", "_linear_layers", "_adapter_path",
                     "inference_mode", "image_processor", "max_seq_length",
                     "model_name", "use_gradient_checkpointing"):
            raise AttributeError(name)
        return getattr(self.model, name)


class UnslothVisionDataCollator:
    """
    Data collator for vision fine-tuning (Unsloth-compatible API).

    Handles batching of image+text data for VLM training. This is the
    equivalent of Unsloth's UnslothVisionDataCollator.

    Usage:
        >>> from mlx_tune import UnslothVisionDataCollator
        >>> collator = UnslothVisionDataCollator(model, processor)
        >>> batch = collator(samples)
    """

    def __init__(self, model: Any, processor: Any):
        """
        Args:
            model: VLMModelWrapper or raw mlx-vlm model
            processor: The model's processor/tokenizer
        """
        self.model = model.model if isinstance(model, VLMModelWrapper) else model
        self.processor = processor
        self._config = self.model.config.__dict__ if hasattr(self.model.config, "__dict__") else self.model.config

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        Collate a list of vision samples into a training batch.

        Each sample should have a 'messages' key with the conversation format:
        [
            {"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]},
            {"role": "assistant", "content": [{"type": "text", ...}]},
        ]
        """
        import mlx.core as mx

        all_input_ids = []
        all_pixel_values = []
        all_masks = []
        extra_kwargs = {}

        for sample in samples:
            messages = sample.get("messages", sample.get("conversations", []))
            images = []

            # Extract images from messages and build clean messages for template
            clean_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    final_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                final_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image":
                                img = item.get("image")
                                if img is not None:
                                    images.append(img)
                                # Keep image marker for processor's chat template
                                final_content.append({"type": "image"})
                    clean_messages.append({"role": msg["role"], "content": final_content})
                else:
                    clean_messages.append(msg)

            # Use processor's own apply_chat_template (NOT mlx-vlm's prompt_utils)
            # The processor correctly inserts vision tokens like <|vision_start|><|image_pad|><|vision_end|>
            prompt = self._apply_chat_template(clean_messages)

            # Get image token index from config
            image_token_index = self._config.get(
                "image_token_index",
                self._config.get("image_token_id"),
            )

            # Prepare inputs using mlx-vlm (handles image tokenization)
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=None,
                prompts=[prompt],
                image_token_index=image_token_index,
            )

            # Ensure input_ids/mask are 2D (batch, seq) — prepare_inputs
            # can return 3D when there are no images
            input_ids = inputs["input_ids"]
            if input_ids.ndim == 3:
                input_ids = input_ids.reshape(-1, input_ids.shape[-1])
            all_input_ids.append(input_ids)

            if inputs.get("pixel_values") is not None:
                all_pixel_values.append(inputs["pixel_values"])

            mask = inputs.get("attention_mask", mx.ones_like(input_ids))
            if mask.ndim == 3:
                mask = mask.reshape(-1, mask.shape[-1])
            all_masks.append(mask)

            # Collect extra kwargs (model-specific like image_grid_thw)
            for k, v in inputs.items():
                if k not in ("input_ids", "pixel_values", "attention_mask"):
                    if k not in extra_kwargs:
                        extra_kwargs[k] = v

        # Stack into batch
        batch = {
            "input_ids": mx.concatenate(all_input_ids, axis=0) if all_input_ids else None,
            "attention_mask": mx.concatenate(all_masks, axis=0) if all_masks else None,
        }
        if all_pixel_values:
            batch["pixel_values"] = mx.concatenate(all_pixel_values, axis=0)
        else:
            batch["pixel_values"] = None

        batch.update(extra_kwargs)
        return batch

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template using the processor (inserts proper vision tokens)."""
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        elif hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            return "\n".join(
                f"{m['role']}: {m.get('content', '')}" for m in messages
            )


class VLMSFTTrainer:
    """
    Supervised Fine-Tuning Trainer for Vision Language Models.

    Uses mlx-vlm's native Trainer and Dataset classes under the hood.
    Provides Unsloth/TRL-compatible API.

    Example:
        >>> from mlx_tune import FastVisionModel, VLMSFTTrainer, UnslothVisionDataCollator
        >>>
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     "mlx-community/Qwen3.5-0.8B-bf16",
        ... )
        >>> model = FastVisionModel.get_peft_model(model, r=16, lora_alpha=16)
        >>> FastVisionModel.for_training(model)
        >>>
        >>> trainer = VLMSFTTrainer(
        ...     model=model,
        ...     tokenizer=processor,
        ...     data_collator=UnslothVisionDataCollator(model, processor),
        ...     train_dataset=converted_dataset,
        ...     args=VLMSFTConfig(max_steps=30, learning_rate=2e-4),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        args: Any = None,
        **kwargs,
    ):
        _require_mlx_vlm()

        self.wrapper = model if isinstance(model, VLMModelWrapper) else None
        self.actual_model = model.model if self.wrapper else model
        self.processor = tokenizer or (model.processor if self.wrapper else None)
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Parse training args
        if args is not None:
            self.learning_rate = getattr(args, "learning_rate", 2e-4)
            self.max_steps = getattr(args, "max_steps", None)
            self.num_train_epochs = getattr(args, "num_train_epochs", 1)
            self.batch_size = getattr(args, "per_device_train_batch_size", 1)
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
            self.warmup_steps = getattr(args, "warmup_steps", 0)
            self.logging_steps = getattr(args, "logging_steps", 10)
            self.output_dir = getattr(args, "output_dir", "./vlm_outputs")
            self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear")
            self.weight_decay = getattr(args, "weight_decay", 0.0)
            self.seed = getattr(args, "seed", 3407)
            self.max_length = getattr(args, "max_length", 2048)
            self.train_on_completions = getattr(args, "train_on_completions", False)
        else:
            self.learning_rate = kwargs.get("learning_rate", 2e-4)
            self.max_steps = kwargs.get("max_steps", None)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 1)
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
            self.warmup_steps = kwargs.get("warmup_steps", 0)
            self.logging_steps = kwargs.get("logging_steps", 10)
            self.output_dir = kwargs.get("output_dir", "./vlm_outputs")
            self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "linear")
            self.weight_decay = kwargs.get("weight_decay", 0.0)
            self.seed = kwargs.get("seed", 3407)
            self.max_length = kwargs.get("max_length", 2048)
            self.train_on_completions = kwargs.get("train_on_completions", False)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """
        Train the VLM using mlx-vlm's native training loop.

        Returns:
            Training metrics dict
        """
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from tqdm import tqdm

        print("=" * 70)
        print("Starting VLM Fine-Tuning")
        print("=" * 70)

        # Ensure model is in training mode
        self.actual_model.train()

        # Set up optimizer
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # Set up the mlx-vlm Trainer
        trainer = VLMTrainerInternal(
            self.actual_model,
            optimizer,
            train_on_completions=self.train_on_completions,
        )

        # Determine total steps
        dataset_len = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 0
        if self.max_steps:
            total_steps = self.max_steps
        elif dataset_len > 0:
            total_steps = (dataset_len // self.batch_size) * self.num_train_epochs
        else:
            total_steps = 100
            print(f"Warning: Could not determine dataset size, using {total_steps} steps")

        # Prepare dataset
        # If train_dataset is a list of dicts with 'messages' key (Unsloth format),
        # use our collator; otherwise try mlx-vlm's Dataset directly
        use_collator = (
            self.data_collator is not None
            and isinstance(self.train_dataset, list)
        )

        if use_collator:
            # Use data collator for Unsloth-format data
            print(f"Training with data collator, {len(self.train_dataset)} samples")
            return self._train_with_collator(trainer, total_steps)
        else:
            # Try mlx-vlm's Dataset
            print("Training with mlx-vlm Dataset")
            return self._train_with_vlm_dataset(trainer, total_steps)

    def _train_with_collator(self, trainer, total_steps):
        """Train using our UnslothVisionDataCollator."""
        import mlx.core as mx
        from tqdm import tqdm

        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0
        step = 0
        epoch = 0

        while step < total_steps:
            epoch += 1
            for i in range(0, len(self.train_dataset), self.batch_size):
                if step >= total_steps:
                    break

                batch_samples = self.train_dataset[i : i + self.batch_size]
                batch = self.data_collator(batch_samples)

                loss = trainer.train_step(batch)
                mx.eval(trainer.model, trainer.optimizer.state)

                loss_val = loss.item()
                total_loss += loss_val
                step += 1

                progress.update(1)
                if step % self.logging_steps == 0:
                    avg_loss = total_loss / step
                    progress.set_postfix(
                        {"loss": f"{loss_val:.4f}", "avg_loss": f"{avg_loss:.4f}"}
                    )

        progress.close()

        # Save adapters
        adapter_dir = Path(self.output_dir) / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))

        if self.wrapper:
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(step, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")
        print(f"Adapters saved to: {adapter_dir}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})

    def _train_with_vlm_dataset(self, trainer, total_steps):
        """Train using mlx-vlm's native Dataset."""
        import mlx.core as mx
        from tqdm import tqdm

        config = self.actual_model.config.__dict__ if hasattr(self.actual_model.config, "__dict__") else self.actual_model.config

        # Wrap dataset
        vlm_dataset = VLMDataset(
            self.train_dataset,
            config,
            self.processor,
            image_processor=self.wrapper.image_processor if self.wrapper else None,
        )

        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0

        for step in range(total_steps):
            idx = step % len(vlm_dataset)
            batch = vlm_dataset[idx * self.batch_size : (idx + 1) * self.batch_size]

            loss = trainer.train_step(batch)
            mx.eval(trainer.model, trainer.optimizer.state)

            loss_val = loss.item()
            total_loss += loss_val
            progress.update(1)

            if (step + 1) % self.logging_steps == 0:
                avg = total_loss / (step + 1)
                progress.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{avg:.4f}"})

        progress.close()

        # Save adapters
        adapter_dir = Path(self.output_dir) / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))

        if self.wrapper:
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(total_steps, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")
        print(f"Adapters saved to: {adapter_dir}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})


class _TrainerStats:
    """Simple container for training metrics (matches HF Trainer output)."""
    def __init__(self, metrics: Dict):
        self.metrics = metrics


class VLMSFTConfig:
    """
    Training configuration for VLM fine-tuning.

    Mirrors TRL's SFTConfig for API compatibility.
    """
    def __init__(
        self,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: Optional[int] = None,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        optim: str = "adam",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        output_dir: str = "outputs",
        report_to: str = "none",
        remove_unused_columns: bool = False,
        dataset_text_field: str = "",
        dataset_kwargs: Optional[Dict] = None,
        max_length: int = 2048,
        train_on_completions: bool = False,
        **kwargs,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.seed = seed
        self.output_dir = output_dir
        self.report_to = report_to
        self.remove_unused_columns = remove_unused_columns
        self.dataset_text_field = dataset_text_field
        self.dataset_kwargs = dataset_kwargs or {}
        self.max_length = max_length
        self.train_on_completions = train_on_completions
        # Store any extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_vlm_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    image_column: str = "image",
    text_column: str = "text",
) -> Any:
    """
    Load a VLM dataset from HuggingFace Hub or local path.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_path: Local dataset path (JSONL)
        split: Dataset split to load
        image_column: Column name for images
        text_column: Column name for text

    Returns:
        Loaded dataset
    """
    if dataset_name:
        from datasets import load_dataset
        return load_dataset(dataset_name, split=split)
    elif dataset_path:
        import json
        with open(dataset_path) as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError("Provide dataset_name or dataset_path")
