import os
import json
import torch
import av
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(SCRIPT_DIR, "mmtrail_1k_train.jsonl")
VIDEO_FOLDER = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "llava_video_checkpoints")

# --- 1. VIDEO LOADING HELPER ---
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# --- 2. DATASET CLASS ---
class MMTrailDataset(Dataset):
    def __init__(self, data_file, video_folder, processor):
        self.data = []
        self.video_folder = video_folder
        self.processor = processor
        
        # Load JSONL
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle filename
        vid_filename = os.path.basename(item["video"])
        video_path = os.path.join(self.video_folder, vid_filename)
        
        # Get Prompt & Response
        conv = item["conversations"]
        user_input = conv[0]["value"] # "<video>\nDescribe..."
        gpt_response = conv[1]["value"]
        
        # Format Prompt: [INST] <video>\nPROMPT [/INST] ANSWER
        prompt = f"[INST] {user_input} [/INST] {gpt_response}"

        try:
            # Sample 32 Frames uniformly
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 32).astype(int)
            video_clip = read_video_pyav(container, indices)
        except Exception as e:
            print(f"Skipping bad video {video_path}: {e}")
            # Recursively try next item to prevent crash
            return self.__getitem__((idx + 1) % len(self.data))

        # Process inputs
        inputs = self.processor(
            text=prompt, 
            videos=video_clip, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        
        # Squeeze batch dims (1, T, C, H, W) -> (T, C, H, W)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0) 
        }

# --- 3. COLLATOR ---
def data_collator(batch):
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    
    # 1. Extract lists
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    pixel_values = [item['pixel_values_videos'] for item in batch]
    
    # 2. Pad Text manually (Silences the LlamaTokenizerFast warning)
    # Pad input_ids with the pad_token_id (usually 0 or 2 for Mistral/Llama)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    # Pad attention_mask with 0 (ignore)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    
    # Pad labels with -100 (ignore)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    # 3. Stack Video Tensors
    pixel_values_stacked = torch.stack(pixel_values)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values_videos": pixel_values_stacked
    }

# --- MAIN ---
def train():
    print("Loading Processor...")
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right" # Essential for training

    print(f"Loading Model: {MODEL_ID} (FP16)...")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, 
        device_map="auto",
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Enable Gradient Checkpointing (Saves massive VRAM)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading Dataset...")
    train_dataset = MMTrailDataset(DATA_FILE, VIDEO_FOLDER, processor)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, # 1 * 8 = Effective Batch Size 8
        num_train_epochs=6,
        learning_rate=2e-5,
        fp16=True,                     # Mixed Precision
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",              # No wandb
        dataloader_num_workers=2       # Speed up video loading
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(OUTPUT_DIR)
    print("Training Complete. Adapter saved.")

if __name__ == "__main__":
    train()