import gc
from unittest import mock
import builtins

import torch
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from transformers import AutoModelForSequenceClassification

from unsloth import FastLanguageModel, FastLlamaModel
from unsloth.models._utils import (
    __version__,
    offload_input_embeddings,
)
from transformers import set_seed as transformers_set_seed
from peft import PeftModelForSequenceClassification, PeftModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from transformers.models.llama.modeling_llama import logger
import inspect


def my_post_patch(model):
    # Torch.compile fails on embedding matrix??
    # Workaround randomnly fixes it for torch versions < 2.
    model.set_input_embeddings(
        torch.nn.Embedding.from_pretrained(model.get_input_embeddings().weight)
    )
    model.config.update({"unsloth_version": __version__})

    # Also patch all dtypes - BnB seems to not allocate the correct type?
    # BnB default dtype seems to be float16!
    correct_dtype = model.get_input_embeddings().weight.dtype

    for name, module in model.named_modules():
        if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
            weight = module.weight
            quant_state = weight.quant_state

            if isinstance(quant_state, list):
                # BnB seems to have float16 as default!
                module.weight.quant_state[2] = correct_dtype  # Cast to correct dtype
            else:
                # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                quant_state.dtype = correct_dtype

        # Downcast RoPE embedding to correct data type
        if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")) and (
            module.cos_cached.dtype != correct_dtype
        ):
            module.cos_cached = module.cos_cached.to(correct_dtype)
            module.sin_cached = module.sin_cached.to(correct_dtype)

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return model


def my_get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    layers_to_transform=None,
    layers_pattern=None,
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=2048,  # not used anymore
    use_rslora=False,
    modules_to_save=None,
    init_lora_weights=True,
    loftq_config={},
    temporary_location="_unsloth_temporary_saved_buffers",
    **kwargs,
):
    transformers_set_seed(random_state)

    if isinstance(model, PeftModelForSequenceClassification):
        raise TypeError("Already a PeftModel.")

    if loftq_config is None:
        loftq_config = {}

    signature = str(inspect.signature(LoraConfig))
    SUPPORTS_LOFTQ = "loftq_config" in signature
    SUPPORTS_RSLORA = "use_rslora" in signature

    assert max_seq_length <= model.max_seq_length

    if lora_dropout != 0:
        logger.warning_once(
            f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
            f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
        )
    pass

    if bias != "none":
        logger.warning_once(
            f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
            f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
        )
    pass

    if not (
        isinstance(init_lora_weights, bool)
        or init_lora_weights == "gaussian"
        or init_lora_weights == "loftq"
    ):
        raise ValueError(
            'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
        )
    pass

    if init_lora_weights == "loftq":
        if not SUPPORTS_LOFTQ:
            import peft

            raise RuntimeError(
                f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"
                "Please install PEFT 0.7.2 or higher.\n"
                "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
            )
        pass

        if loftq_config == {}:
            from peft import LoftQConfig

            logger.warning_once(
                "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"
                "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
            )
            loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)
        pass

        if hasattr(model.config, "quantization_config"):
            raise ValueError(
                "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"
                "Reload your model without any quantization by setting `load_in_4bit = False`."
            )
        pass
    pass

    assert isinstance(use_rslora, bool)
    if use_rslora:
        if not SUPPORTS_RSLORA:
            # We manually check for PEFT
            import peft

            raise RuntimeError(
                f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"
                "Please install PEFT 0.7.2 or higher.\n"
                "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
            )
        pass
    pass

    accepted_modules = frozenset(
        (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model.config.update({"unsloth_version": __version__})

    if type(modules_to_save) is tuple:
        modules_to_save = list(modules_to_save)
    pass

    train_score_head = False
    train_embed_tokens = False
    final_modules = []
    for module in target_modules:
        if module == "score":
            # logger.warning_once(
            #     "Unsloth: `score` should be placed in `modules_to_save` and not `target_modules`. "\
            #     "Luckily, we shall do it for you!"
            # )
            train_score_head = True
            if modules_to_save is None:
                modules_to_save = ["score"]
            else:
                modules_to_save.append("score")

        elif module == "embed_tokens":
            # logger.warning_once(
            #     "Unsloth: `embed_tokens` should be placed in `modules_to_save` and not `target_modules`. "\
            #     "Luckily, we shall do it for you!"
            # )
            train_embed_tokens = True
            if modules_to_save is None:
                modules_to_save = ["embed_tokens"]
            else:
                modules_to_save.append("embed_tokens")

        else:
            assert module in accepted_modules
            final_modules.append(module)

    # Check if we added new tokens!
    if hasattr(model, "_need_to_train_embeddings"):
        if not train_score_head or not train_embed_tokens:
            print(
                "Unsloth: You added new tokens but did not specify if you wanted to "
                "train the score head and embed_tokens.\nWe must turn it on for you."
            )
            train_score_head = True
            train_embed_tokens = True

            if modules_to_save is None:
                modules_to_save = ["embed_tokens"]
            else:
                modules_to_save.append("embed_tokens")

            if modules_to_save is None:
                modules_to_save = ["score"]
            else:
                modules_to_save.append("score")

    # Check for Llama-3
    # if hasattr(model._saved_temp_tokenizer, "_using_llama3_template"):
    #     if not train_embed_tokens and not train_lm_head:
    #         raise RuntimeError("")

    # First fix untrained tokens
    # Wrong - can cause reserved tokens to pop out!!
    # if train_embed_tokens or train_lm_head:
    #     fix_untrained_tokens(model, eps = 1e-16)
    # pass

    # Check modules_to_save
    if modules_to_save is not None:
        for module in modules_to_save:
            if module == "score":
                train_score_head = True
            elif module == "embed_tokens":
                train_embed_tokens = True
            else:
                raise TypeError(
                    f"Unsloth: Module = {module} is not allowed. Only 'score' and 'embed_tokens' is allowed."
                )
        pass
    pass
    if isinstance(modules_to_save, (tuple, list)):
        modules_to_save = list(set(modules_to_save))
    pass

    # Get LoRA
    arguments = dict(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=final_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.SEQ_CLS,
        layers_to_transform=layers_to_transform,
        init_lora_weights=init_lora_weights,
        loftq_config=loftq_config,
        use_rslora=use_rslora,
        modules_to_save=modules_to_save,
        **kwargs,
    )
    if not SUPPORTS_LOFTQ:
        del arguments["loftq_config"]
    if not SUPPORTS_RSLORA:
        del arguments["use_rslora"]

    _saved_temp_tokenizer = model._saved_temp_tokenizer

    lora_config = LoraConfig(**arguments)

    # First offload embed_tokens to disk
    # input_embeddings_device = model.get_input_embeddings().weight.device
    # output_embeddings_device = model.get_output_embeddings().weight.device

    if use_gradient_checkpointing == "unsloth":
        if train_embed_tokens:
            print("Unsloth: Offloading input_embeddings to disk to save VRAM")
            offload_input_embeddings(model, temporary_location)
        pass

        # Remove old items to save VRAM
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

        # if train_score:
        #     print("Unsloth: Offloading output_embeddings to disk to save VRAM")
        #     offload_output_embeddings(model, temporary_location)

        # Remove old items to save VRAM
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    model = _get_peft_model(model, lora_config)

    model._saved_temp_tokenizer = _saved_temp_tokenizer

    model = FastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)

    # Now patch score and embed_tokens
    if train_embed_tokens:
        print("Unsloth: Casting embed_tokens to float32")
        assert hasattr(model.model.model.embed_tokens, "modules_to_save")
        model.model.model.embed_tokens.modules_to_save.default.to(
            device="cuda:0", dtype=torch.float32, non_blocking=True
        )
        model.model.model.embed_tokens.modules_to_save.default.requires_grad_(True)

    if train_score_head:
        print("Unsloth: Casting score to float32")
        assert hasattr(model.model.score, "modules_to_save")
        model.model.score.modules_to_save.default.to(
            device="cuda:0", dtype=torch.float32, non_blocking=True
        )
        model.model.score.modules_to_save.default.requires_grad_(True)

    # Patch tokenizer to pad to the right
    internal_model = model
    while hasattr(internal_model, "model"):
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"

        internal_model = internal_model.model

    if hasattr(internal_model, "_saved_temp_tokenizer"):
        internal_model._saved_temp_tokenizer.padding_side = "right"

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    return model


original_isinstance = builtins.isinstance


def isinstance_patch(obj, cls):
    if original_isinstance(obj, PeftModelForSequenceClassification) and (
        cls == PeftModelForCausalLM
    ):
        return True
    else:
        return original_isinstance(obj, cls)


if __name__ == "__main__":
    # Mocking the AutoModelForCausalLM to AutoModelForSeqClassification
    def patched_from_pretrained_peft_seq_model(
        model_id: str, max_seq_length: int, num_labels: int
    ):
        with (
            mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained"
            ) as mock_causal_lm,
            mock.patch(
                "unsloth.models.llama.FastLlamaModel.post_patch"
            ) as mock_post_patch,
            mock.patch(
                "unsloth.models.llama.FastLlamaModel.get_peft_model"
            ) as mock_get_peft_model,
            mock.patch(
                "builtins.isinstance",
                new=isinstance_patch,
            ),
        ):
            mock_causal_lm.side_effect = (
                AutoModelForSequenceClassification.from_pretrained
            )
            mock_post_patch.side_effect = my_post_patch
            mock_get_peft_model.side_effect = my_get_peft_model

            model, tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
                max_seq_length=1024,
                num_labels=3,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                modules_to_save=["score"],
            )
        return model, tokenizer

    # load dataset

    print(model)  # This should be an instance of AutoModelForSeqClassification
    print(tokenizer)  # This will be an instance of AutoTokenizer as usual

    # patch model to behave like a trandformers SeqCLS model
    # add configs..

    # patch get_peft_model to use good cls type etc...
    model = FastLanguageModel.get_peft_model(
        model,
    )
