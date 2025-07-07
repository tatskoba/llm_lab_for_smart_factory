# 参考サイト https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/vision_finetuning/finetune_hf_trainer_ucf101.py

# 本コードはGPUを使用してください（CPUには対応していません）

import json
import os
import numpy as np
from pathlib import Path
import gc

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from phi3v_dataset import Phi3VDataCollator, Phi3VDataset, Phi3VEvalDataCollator, Phi3VEvalDataset
import wandb
import warnings
warnings.filterwarnings("ignore")
from system_setting import setup_env_data

use_wandb = False  # wandbによるトレーニング中の出力を行う
do_eval = False  # モデルトレーニングの前後で評価を行うかどうか

if use_wandb:
    # .envファイルからwandbの設定を取得
    WANDB_KEY = os.environ.get("WADB_KEY")             # wandbのAPIキー
    WANDB_PROJECT = os.environ.get('WANDB_PROJECT')     # プロジェクト名
    WANDB_ENTITY = os.environ.get('WANDB_ENTITY')       # ユーザー名

    # Initialize Weights & Biases設定
    wandb.require("core")
    wandb.login(key=WANDB_KEY)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, resume="allow")

# システム設定を取得
setup_env_data()
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')
model_id = MODEL_FOLDER + "phi35v"
save_dir = MODEL_FOLDER + "phi35v_finetuning_lora"    # Directory to save the model
os.makedirs(save_dir, exist_ok=True)

DATA_FOLDER = os.environ.get('DATA_FOLDER')
SAMPLE_DATASET = DATA_FOLDER + "time_series_data/images/"

torch.manual_seed(3)   # Set seed for reproducibility

#----------------------------------------------
# LoRAファインチューニングの条件設定
data_dir = DATA_FOLDER + "time_series_data"
train_jsonl_file = "time_series_train.jsonl"
val_jsonl_file = "time_series_val.jsonl"
test_jsonl_file = "time_series_test.jsonl"   # このスクリプトではtestデータは使用しない

logging_steps = 1   # トレーニング中の損失などの出力ステップ間隔

use_flash_attention = True
num_crops = 4

num_train_epochs = 3  # トレーニングepoch数指定（デモでは3エポックがよい）

batch_size = 1
batch_size_per_gpu = 1   # Batch size per GPU (adjust this to fit in GPU memory)
learning_rate = 4.0e-5
weight_decay = 0.01
bf16 = True

world_size = 1
env_rank = 0
local_rank = 0
lora_rank= 64          # LoRA rank (rank of the low-rank approximation)
lora_alpha_ratio = 16   # LoRA alpha ratio (ratio of the number of parameters in the low-rank approximation to the rank)
lora_dropout=0.02      # LoRA dropout
freeze_vision_model=False

disable_tqdm = True    # Trueにすると、ファインチューニング中のプログレスバーが表示される

userPrompt = "この時系列データは次のどのカテゴリーにあてはまりますか: stage_1_param_1_NG, stage_1_param_1_OK, stage_1_param_2_NG, stage_1_param_2_OK, stage_2_param_1_NG, stage_2_param_1_OK, stage_2_param_2_NG, stage_2_param_2_OK, stage_3_param_1_NG, stage_3_param_1_OK, stage_3_param_2_NG, stage_3_param_2_OK"

if use_wandb:
    report_to = "wandb"
else:
    report_to = "none"

#----------------------------------------------
# 事前準備したローカルデータからデータセットを作成
def create_dataset(data_dir, processor):
    data_path = Path(data_dir)
    train_dataset = Phi3VDataset(
        jsonl_file=str(data_path / train_jsonl_file),
        image_dir=str(data_path / 'images'),
        processor=processor,
    )
    eval_dataset = Phi3VEvalDataset(
        jsonl_file=str(data_path / val_jsonl_file),
        image_dir=str(data_path / 'images'),
        processor=processor,
    )

    return train_dataset, eval_dataset


def create_lora_config(rank, alpha_to_rank_ratio=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj',
        'lm_head',
    ]
    if not freeze_vision_model:
        vision_linear_modules = [
            # CLIP modules
            'q_proj',  # attention
            'k_proj',
            'v_proj',
            'out_proj',
            'fc1',  # MLP
            'fc2',
            # image projection
            'img_projection.0',
            'img_projection.2',
        ]
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=round(rank * alpha_to_rank_ratio),
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config


def create_model(model_name_or_path, use_flash_attention=False):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        low_cpu_mem_usage=True,
        use_cache=True,
    )

    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):

    model.eval()
    answers_unique = []
    generated_texts_unique = []

    eval_dataset_shard = eval_dataset.shard(world_size, env_rank)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset_shard,
        batch_size=eval_batch_size,
        collate_fn=Phi3VEvalDataCollator(processor.tokenizer.pad_token_id),
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    for batch in tqdm(eval_dataloader, disable=(env_rank != 0) or disable_tqdm):
        unique_ids = batch.pop('unique_ids')
        answers = batch.pop('answers')
        answers_unique.extend(
            {'id': i, 'answer': a.strip().strip('.').lower()} for i, a in zip(unique_ids, answers)
        )

        inputs = {k: v.to(f'cuda:{local_rank}') for k, v in batch.items()}
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64
        )

        input_len = inputs['input_ids'].size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_texts_unique.extend(
            {'id': i, 'generated_text': g.strip().strip('.').lower()}
            for i, g in zip(unique_ids, generated_texts)
        )

    # gather outputs from all ranks
    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)

    if env_rank == 0:
        assert len(answers_unique) == len(generated_texts_unique)
        acc = sum(
            a['answer'] == g['generated_text']
            for a, g in zip(answers_unique, generated_texts_unique)
        ) / len(answers_unique)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers_unique': answers_unique,
                    'generated_texts_unique': generated_texts_unique,
                    'accuracy': acc,
                }
                json.dump(save_dict, f)

        return acc
    return None


def patch_clip_for_lora(model):
    # remove unused parameters and then monkey patch
    def get_img_features(self, img_embeds):
        clip_vision_model = self.img_processor.vision_model
        hidden_states = clip_vision_model.embeddings(img_embeds)
        hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
        patch_feature = clip_vision_model.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        ).hidden_states[-1][:, 1:]
        return patch_feature

    image_embedder = model.model.vision_embed_tokens
    layer_index = image_embedder.layer_idx
    clip_layers = image_embedder.img_processor.vision_model.encoder.layers
    if layer_index < 0:
        layer_index = len(clip_layers) + layer_index
    del clip_layers[layer_index + 1 :]
    del image_embedder.img_processor.vision_model.post_layernorm
    image_embedder.get_img_features = get_img_features.__get__(image_embedder)

#----------------------------------------------

# 1. Load Model and Processor
print("1. Load Model and Processor")

# ガベージコレクションで変数を削除してメモリを解放
gc.collect()

# GPUメモリの解放
torch.cuda.empty_cache()

assert num_crops <= 16, 'num_crops must be less than or equal to 16'

accelerator = Accelerator()

with accelerator.local_main_process_first():
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        num_crops=num_crops
    )
    model = create_model(
        model_id,
        use_flash_attention=use_flash_attention,
    )

num_gpus = accelerator.num_processes
print(f'training on {num_gpus} GPUs')
assert (
    batch_size % (num_gpus * batch_size_per_gpu) == 0
), 'Batch size must be divisible by the number of GPUs'
gradient_accumulation_steps = batch_size // (num_gpus * batch_size_per_gpu)


# 2. データ準備
print("2. Data Preparation")

train_dataset, eval_dataset = create_dataset(data_dir, processor)


# 3. ファインチューニング前のモデル評価
print("3. Evaluation of the model before finetuning")

out_path = Path(save_dir)
out_path.mkdir(parents=True, exist_ok=True)

model = model.to(f'cuda:{local_rank}')

if do_eval:
    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not disable_tqdm,
        eval_batch_size=batch_size_per_gpu,
    )

    if accelerator.is_main_process:
        print(f'Accuracy before finetuning: {acc}')
else:
    print("  Evaluation skipped")
    acc = None


## 4. ファインチューニング設定
print("4. Setup LoRA")

# トレーニング引数の設定
training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=logging_steps, 
        output_dir=save_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        remove_unused_columns=False,
        report_to=report_to,
        disable_tqdm=not disable_tqdm,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
)

data_collator = Phi3VDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

# LoRAコンフィグ設定
patch_clip_for_lora(model)
lora_config = create_lora_config(
    rank=lora_rank,
    alpha_to_rank_ratio=lora_alpha_ratio,
    dropout=lora_dropout,
    freeze_vision_model=freeze_vision_model,
)

# モデルにLoRAアダプターを追加
model.add_adapter(lora_config)
model.enable_adapters()

# モデルのトークンをFreezeする
if freeze_vision_model:
    model.model.vision_embed_tokens.requires_grad_(False)

# モデルのキャッシュを無効にする
model.config.use_cache = False


## 5. モデルトレーニング
print("5. Model training")

# トレーナーを作成
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# モデルをトレーニング
trainer.train()


## 6. モデル保存
print("6. Model save")

processor.chat_template = processor.tokenizer.chat_template

trainer.save_model()
if accelerator.is_main_process:
    processor.save_pretrained(save_dir)
accelerator.wait_for_everyone()


## 7. ファインチューニング後のモデル評価
print("7. Evaluation of the model after finetuning")

if do_eval:
    # GPUメモリをクリアする
    del model
    del trainer
    __import__('gc').collect()

    torch.cuda.empty_cache()

    # モデルをロード
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
    )
    patch_clip_for_lora(model)
    model.load_adapter(save_dir)   # LoRAアダプターをロード

    model = model.to(f'cuda:{local_rank}')

    # ファインチューニング後のモデル評価
    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not disable_tqdm,
        eval_batch_size=batch_size_per_gpu,
    )
    if env_rank == 0:
        print(f'Accuracy after finetuning: {acc}')
else:
    print("  Evaluation skipped")
    acc = None
