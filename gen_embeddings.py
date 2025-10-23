import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast

# ---------------------------
# 1️⃣ Configuration
# ---------------------------

# IMPORTANT: Update these paths to match your setup

CHECKPOINT_PATH = "gayy/price_predictor_epoch_3"  # Path to the saved model folder (e.g., from epoch 3)
TEST_CSV_PATH = "../Suryansh/train.csv" # Path to your test data file
OUTPUT_CSV_PATH = "train_embeddings.csv" # Path to save the final predictions

BATCH_SIZE = 8 # Adjust based on your GPU memory

# ---------------------------

# 2️⃣ Setup Device and Dtype

# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"Using device: {device} with dtype: {compute_dtype}")



# ---------------------------

# 3️⃣ Load Model, Processor, and Regression Head

# ---------------------------

print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")



# a. Load the base model

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True
)



# b. Load the LoRA adapters onto the base model

model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)



# c. Define and load the custom regression head

hidden_dim = 512

model.regression_head = nn.Sequential(

    nn.Linear(model.config.hidden_size, hidden_dim),

    nn.ReLU(),

    nn.Linear(hidden_dim, 1)

)

regression_head_path = os.path.join(CHECKPOINT_PATH, "regression_head.pth")

model.regression_head.load_state_dict(torch.load(regression_head_path, map_location=device))

model.regression_head.to(device)



# d. Load the processor

processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)



print("✅ Model, adapters, and processor loaded successfully.")



# ---------------------------

# 4️⃣ Test Dataset and DataLoader

# ---------------------------

class TestPriceDataset(Dataset):

    """

    Modified dataset for inference. It does not expect a 'price' column.

    """

    def __init__(self, df, processor, image_cache_dir="../Suryansh/AMLC/train"):

        self.df = df.reset_index(drop=True)

        self.processor = processor

        self.image_cache_dir = image_cache_dir

        os.makedirs(self.image_cache_dir, exist_ok=True)



    def __len__(self):

        return len(self.df)



    def _download_image(self, url, sample_id):

        filename = "".join(c for c in str(sample_id) if c.isalnum()) + ".jpg"

        cache_path = os.path.join(self.image_cache_dir, filename)

        try:

            if not os.path.exists(cache_path):

                response = requests.get(url, timeout=10)

                response.raise_for_status()

                img = Image.open(BytesIO(response.content)).convert("RGB")

                img.save(cache_path)

            else:

                img = Image.open(cache_path).convert("RGB")

            img = img.resize((224, 224))

            return img

        except (requests.exceptions.RequestException, IOError) as e:

            print(f"Warning: Could not process image for {sample_id}. Returning placeholder.")

            return Image.new('RGB', (224, 224), (0, 0, 0))



    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample_id = row['sample_id']

        catalog_content = row['catalog_content']

        image_url = row['image_link']

        

        image = self._download_image(image_url, sample_id)



        messages = [{

            "role": "user",

            "content": [

                {"type": "image"},

                {"type": "text", "text": f"Catalog: {catalog_content}\nInstruction: Encode product for price prediction."}

            ]

        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        

        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs['sample_id'] = sample_id

        return inputs



def collate_fn_test(batch):

    """Modified collate function for inference."""

    input_ids = [item['input_ids'] for item in batch]

    attention_mask = [item['attention_mask'] for item in batch]

    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])

    sample_ids = [item['sample_id'] for item in batch]



    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)

    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)



    return {

        'input_ids': input_ids,

        'attention_mask': attention_mask,

        'pixel_values': pixel_values,

        'image_grid_thw': image_grid_thw,

        'sample_ids': sample_ids

    }



# Load test data

try:

    test_df = pd.read_csv(TEST_CSV_PATH)

    test_df['catalog_content'] = test_df['catalog_content'].fillna('')

except FileNotFoundError:

    print(f"Error: {TEST_CSV_PATH} not found. Exiting.")

    exit()



test_dataset = TestPriceDataset(test_df, processor)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test)



# ---------------------------

# 5️⃣ Prediction Loop

# ---------------------------

model.eval()
all_embeddings = []
all_sample_ids = []
all_prices = []  # assuming your CSV still has a 'price' column

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting Embeddings"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device, dtype=compute_dtype)
        image_grid_thw = batch['image_grid_thw'].to(device)
        sample_ids = batch['sample_ids']

        with autocast(device_type=device, dtype=compute_dtype, enabled=(device=='cuda')):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True
            )

            # Get last hidden layer embeddings
            last_hidden_state = outputs.hidden_states[-1]

            # Use the last token’s embedding as pooled representation (same as before)
            last_token_indices = attention_mask.sum(dim=1) - 1
            pooled_embedding = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=device),
                last_token_indices
            ]

        # Move to CPU and convert to list
        all_embeddings.extend(pooled_embedding.cpu().to(torch.float32).numpy().tolist())
        all_sample_ids.extend(sample_ids)

        # Only if price column exists in your CSV
        if "price" in test_df.columns:
            all_prices.extend(test_df.loc[test_df["sample_id"].isin(sample_ids), "price"].tolist())



# ---------------------------

# 6️⃣ Save Results

# ---------------------------

# Convert embeddings into a DataFrame
embedding_df = pd.DataFrame(all_embeddings)
embedding_df.insert(0, 'sample_id', all_sample_ids)

if "price" in test_df.columns:
    embedding_df['price'] = all_prices

embedding_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\n✅ Embeddings saved successfully to {OUTPUT_CSV_PATH}")
print(embedding_df.head())
