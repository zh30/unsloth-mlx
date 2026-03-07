"""
Tests for VLM (Vision Language Model) support.

Tests the FastVisionModel API, VLMModelWrapper, UnslothVisionDataCollator,
VLMSFTTrainer, and VLMSFTConfig.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import json
import tempfile


# ============================================================================
# Test FastVisionModel API
# ============================================================================


class TestFastVisionModelAPI:
    """Test that FastVisionModel has the correct Unsloth-compatible API."""

    def test_has_from_pretrained(self):
        from mlx_tune import FastVisionModel
        assert hasattr(FastVisionModel, "from_pretrained")
        assert callable(FastVisionModel.from_pretrained)

    def test_has_get_peft_model(self):
        from mlx_tune import FastVisionModel
        assert hasattr(FastVisionModel, "get_peft_model")
        assert callable(FastVisionModel.get_peft_model)

    def test_has_for_training(self):
        from mlx_tune import FastVisionModel
        assert hasattr(FastVisionModel, "for_training")
        assert callable(FastVisionModel.for_training)

    def test_has_for_inference(self):
        from mlx_tune import FastVisionModel
        assert hasattr(FastVisionModel, "for_inference")
        assert callable(FastVisionModel.for_inference)


# ============================================================================
# Test VLMModelWrapper
# ============================================================================


class TestVLMModelWrapper:
    """Test VLMModelWrapper functionality."""

    def _make_wrapper(self):
        from mlx_tune.vlm import VLMModelWrapper
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "qwen3_5"}
        mock_model.train = MagicMock()
        mock_model.eval = MagicMock()
        mock_processor = MagicMock()

        wrapper = VLMModelWrapper(
            model=mock_model,
            processor=mock_processor,
            max_seq_length=2048,
            model_name="test-model",
        )
        return wrapper

    def test_init_defaults(self):
        wrapper = self._make_wrapper()
        assert wrapper.model_name == "test-model"
        assert wrapper.max_seq_length == 2048
        assert wrapper.lora_enabled is False
        assert wrapper._lora_applied is False
        assert wrapper.inference_mode is False

    def test_lora_config_storage(self):
        wrapper = self._make_wrapper()
        wrapper.lora_config = {"r": 16, "lora_alpha": 16}
        wrapper.lora_enabled = True
        assert wrapper.lora_config["r"] == 16
        assert wrapper.lora_enabled is True

    def test_adapter_path_tracking(self):
        wrapper = self._make_wrapper()
        assert wrapper.get_adapter_path() is None
        wrapper.set_adapter_path("/tmp/adapters")
        assert wrapper.get_adapter_path() == Path("/tmp/adapters")

    def test_for_training_sets_mode(self):
        from mlx_tune import FastVisionModel
        wrapper = self._make_wrapper()
        FastVisionModel.for_training(wrapper)
        assert wrapper.inference_mode is False
        wrapper.model.train.assert_called_once()

    def test_for_inference_sets_mode(self):
        from mlx_tune import FastVisionModel
        wrapper = self._make_wrapper()
        FastVisionModel.for_inference(wrapper)
        assert wrapper.inference_mode is True
        wrapper.model.eval.assert_called_once()

    def test_save_pretrained_no_lora(self):
        wrapper = self._make_wrapper()
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper.save_pretrained(tmpdir)
            # No adapters should be saved since LoRA not applied
            assert not (Path(tmpdir) / "adapters.safetensors").exists()


# ============================================================================
# Test VLMSFTConfig
# ============================================================================


class TestVLMSFTConfig:
    """Test VLMSFTConfig (TRL SFTConfig compatibility)."""

    def test_default_values(self):
        from mlx_tune import VLMSFTConfig
        config = VLMSFTConfig()
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.warmup_steps == 5
        assert config.learning_rate == 2e-4
        assert config.output_dir == "outputs"
        assert config.max_length == 2048

    def test_custom_values(self):
        from mlx_tune import VLMSFTConfig
        config = VLMSFTConfig(
            per_device_train_batch_size=4,
            max_steps=100,
            learning_rate=1e-4,
            output_dir="my_output",
        )
        assert config.per_device_train_batch_size == 4
        assert config.max_steps == 100
        assert config.learning_rate == 1e-4
        assert config.output_dir == "my_output"

    def test_unsloth_vision_params(self):
        """Test Unsloth-specific params that are needed for vision finetuning."""
        from mlx_tune import VLMSFTConfig
        config = VLMSFTConfig(
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        assert config.remove_unused_columns is False
        assert config.dataset_text_field == ""
        assert config.dataset_kwargs == {"skip_prepare_dataset": True}


# ============================================================================
# Test UnslothVisionDataCollator
# ============================================================================


class TestUnslothVisionDataCollator:
    """Test the vision data collator."""

    def test_import(self):
        from mlx_tune import UnslothVisionDataCollator
        assert UnslothVisionDataCollator is not None

    def test_init_with_wrapper(self):
        from mlx_tune.vlm import UnslothVisionDataCollator, VLMModelWrapper
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "test", "image_token_index": 151655}
        mock_processor = MagicMock()

        wrapper = VLMModelWrapper(
            model=mock_model,
            processor=mock_processor,
            model_name="test",
        )
        collator = UnslothVisionDataCollator(wrapper, mock_processor)
        # Should unwrap to get actual model
        assert collator.model is mock_model

    def test_init_with_raw_model(self):
        from mlx_tune.vlm import UnslothVisionDataCollator
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "test"}
        mock_processor = MagicMock()

        collator = UnslothVisionDataCollator(mock_model, mock_processor)
        assert collator.model is mock_model


# ============================================================================
# Test VLMSFTTrainer
# ============================================================================


class TestVLMSFTTrainer:
    """Test VLMSFTTrainer initialization and configuration."""

    def _make_trainer(self, **kwargs):
        from mlx_tune.vlm import VLMSFTTrainer, VLMSFTConfig, VLMModelWrapper

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "test"}
        mock_model.train = MagicMock()
        mock_processor = MagicMock()

        wrapper = VLMModelWrapper(
            model=mock_model,
            processor=mock_processor,
            model_name="test",
        )

        config = VLMSFTConfig(**kwargs) if kwargs else VLMSFTConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            trainer = VLMSFTTrainer(
                model=wrapper,
                tokenizer=mock_processor,
                train_dataset=[],
                args=config,
            )
            return trainer

    def test_init_with_config(self):
        trainer = self._make_trainer(learning_rate=1e-4, max_steps=50)
        assert trainer.learning_rate == 1e-4
        assert trainer.max_steps == 50

    def test_init_defaults(self):
        trainer = self._make_trainer()
        assert trainer.learning_rate == 2e-4
        assert trainer.batch_size == 2
        assert trainer.gradient_accumulation_steps == 4

    def test_extracts_actual_model(self):
        trainer = self._make_trainer()
        # Should extract inner model from wrapper
        assert trainer.wrapper is not None
        assert trainer.actual_model is not None


# ============================================================================
# Test get_peft_model target module detection
# ============================================================================


class TestTargetModules:
    """Test target module detection for LoRA."""

    def test_default_attention_and_mlp(self):
        from mlx_tune.vlm import _get_target_modules
        mock_model = MagicMock()
        modules = _get_target_modules(
            mock_model,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules
        assert "o_proj" in modules
        assert "gate_proj" in modules
        assert "up_proj" in modules
        assert "down_proj" in modules

    def test_attention_only(self):
        from mlx_tune.vlm import _get_target_modules
        mock_model = MagicMock()
        modules = _get_target_modules(
            mock_model,
            finetune_attention_modules=True,
            finetune_mlp_modules=False,
        )
        assert "q_proj" in modules
        assert "gate_proj" not in modules

    def test_mlp_only(self):
        from mlx_tune.vlm import _get_target_modules
        mock_model = MagicMock()
        modules = _get_target_modules(
            mock_model,
            finetune_attention_modules=False,
            finetune_mlp_modules=True,
        )
        assert "q_proj" not in modules
        assert "gate_proj" in modules


# ============================================================================
# Test load_vlm_dataset
# ============================================================================


class TestLoadVLMDataset:
    """Test VLM dataset loading."""

    def test_load_from_jsonl(self):
        from mlx_tune import load_vlm_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"image": "test.jpg", "text": "hello"}) + "\n")
            f.write(json.dumps({"image": "test2.jpg", "text": "world"}) + "\n")
            f.flush()

            data = load_vlm_dataset(dataset_path=f.name)
            assert len(data) == 2
            assert data[0]["text"] == "hello"

    def test_requires_dataset_source(self):
        from mlx_tune import load_vlm_dataset
        with pytest.raises(ValueError, match="Provide dataset_name or dataset_path"):
            load_vlm_dataset()


# ============================================================================
# Test Unsloth API compatibility
# ============================================================================


class TestUnslothAPICompatibility:
    """Test that the API matches Unsloth's FastVisionModel exactly."""

    def test_get_peft_model_accepts_unsloth_params(self):
        """Verify all Unsloth-specific parameters are accepted."""
        from mlx_tune.vlm import FastVisionModel, VLMModelWrapper

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "test"}
        mock_model.language_model = MagicMock()
        mock_model.train = MagicMock()
        mock_processor = MagicMock()

        wrapper = VLMModelWrapper(
            model=mock_model,
            processor=mock_processor,
            model_name="test",
        )

        # These are the exact params from the Unsloth notebook
        with patch("mlx_tune.vlm.vlm_get_peft_model", return_value=mock_model):
            result = FastVisionModel.get_peft_model(
                wrapper,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        assert result.lora_enabled is True
        assert result._lora_applied is True
        assert result.lora_config["r"] == 16

    def test_vlm_sft_trainer_accepts_sft_config(self):
        """Verify VLMSFTTrainer accepts SFTConfig-like args."""
        from mlx_tune.vlm import VLMSFTTrainer, VLMSFTConfig, VLMModelWrapper

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.__dict__ = {"model_type": "test"}
        mock_model.train = MagicMock()
        mock_processor = MagicMock()

        wrapper = VLMModelWrapper(
            model=mock_model,
            processor=mock_processor,
            model_name="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # These are the exact params from the Unsloth notebook
            trainer = VLMSFTTrainer(
                model=wrapper,
                tokenizer=mock_processor,
                data_collator=MagicMock(),
                train_dataset=[],
                args=VLMSFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    max_steps=30,
                    learning_rate=2e-4,
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.001,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=tmpdir,
                    report_to="none",
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=2048,
                ),
            )
        assert trainer.learning_rate == 2e-4
        assert trainer.max_steps == 30
        assert trainer.batch_size == 2


# ============================================================================
# Test imports are all available
# ============================================================================


class TestImports:
    """Test all VLM-related imports work."""

    def test_fast_vision_model(self):
        from mlx_tune import FastVisionModel
        assert FastVisionModel is not None

    def test_vlm_sft_trainer(self):
        from mlx_tune import VLMSFTTrainer
        assert VLMSFTTrainer is not None

    def test_vlm_sft_config(self):
        from mlx_tune import VLMSFTConfig
        assert VLMSFTConfig is not None

    def test_unsloth_vision_data_collator(self):
        from mlx_tune import UnslothVisionDataCollator
        assert UnslothVisionDataCollator is not None

    def test_vlm_model_wrapper(self):
        from mlx_tune import VLMModelWrapper
        assert VLMModelWrapper is not None

    def test_load_vlm_dataset(self):
        from mlx_tune import load_vlm_dataset
        assert load_vlm_dataset is not None
