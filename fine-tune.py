import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler
from torch.amp import autocast

class ProductDataHandler(Dataset):
    """
    Dataset to load product catalog info, images, and prices.
    Images are downloaded and cached locally to speed up subsequent runs.
    """
    def __init__(self, data_frame, tokenizer_and_processor, local_image_storage="AMLC/train"):
        self.data_frame = data_frame.reset_index(drop=True)
        self.tokenizer_and_processor = tokenizer_and_processor
        self.local_image_storage = local_image_storage
        os.makedirs(self.local_image_storage, exist_ok=True)

    def __len__(self):
        return len(self.data_frame)

    def _cache_image_from_url(self, image_link_url, product_identifier):
        """Downloads and caches an image from a URL."""
        # Use a sanitized version of product_identifier for the filename
        cached_image_name = "".join(c for c in str(product_identifier) if c.isalnum()) + ".jpg"
        local_image_path = os.path.join(self.local_image_storage, cached_image_name)
        
        try:
            if not os.path.exists(local_image_path):
                http_response = requests.get(image_link_url, timeout=10)
                http_response.raise_for_status()
                image_data = Image.open(BytesIO(http_response.content)).convert("RGB")
                image_data.save(local_image_path)
            else:
                image_data = Image.open(local_image_path).convert("RGB")
            
            # Resize image to a standard size
            image_data = image_data.resize((224, 224))
            return image_data
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Warning: Could not download or process image for sample {product_identifier}. Error: {e}. Returning a placeholder.")
            # Return a black placeholder image if download fails
            return Image.new('RGB', (224, 224), (0, 0, 0))


    def __getitem__(self, item_index):
        data_record = self.data_frame.iloc[item_index]
        product_identifier = data_record['sample_id']
        product_description = data_record['catalog_content']
        source_image_url = data_record['image_link']
        
        target_value = torch.log1p(torch.tensor(data_record['price'], dtype=torch.float32))

        # Download and process the image
        image_data = self._cache_image_from_url(source_image_url, product_identifier)

        # The model requires a specific chat format to correctly map images to `<img>` tokens.
        prompt_structure = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Catalog: {product_description}\nInstruction: Encode product for price prediction."}
                ]
            }
        ]
        # `apply_chat_template` creates the full prompt string with necessary special tokens
        formatted_text_input = self.tokenizer_and_processor.apply_chat_template(
            prompt_structure,
            tokenize=False,
            add_generation_prompt=False # Not generating text, so this is false
        )

        # Use the processor to prepare both text and image for the model
        model_inputs = self.tokenizer_and_processor(text=formatted_text_input, images=image_data, return_tensors="pt")
        # The processor adds a batch dimension, which we remove for the dataset item
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # Add price to the dictionary of inputs
        model_inputs['price'] = target_value
        
        return model_inputs


# ---------------------------
# 2️⃣ Load CSV + Train/Test Split
# ---------------------------
try:
    source_csv_path = "train.csv"
    full_df = pd.read_csv(source_csv_path)
    # Ensure catalog content is always a string to prevent errors
    full_df['catalog_content'] = full_df['catalog_content'].fillna('')
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure the file is at the correct path.")
    # Create a dummy dataframe for demonstration purposes if the file is not found
    dummy_data = {
        'sample_id': range(10),
        'catalog_content': [f'Product {i}' for i in range(10)],
        'image_link': ['https://placehold.co/600x400/EEE/31343C?text=Product' for _ in range(10)],
        'price': [10.5 * (i+1) for i in range(10)]
    }
    full_df = pd.DataFrame(dummy_data)
    print("Using a dummy dataframe for demonstration.")


training_set_df, validation_set_df = train_test_split(full_df, test_size=0.2, random_state=42)

# ---------------------------
# 3️⃣ Processor + Model
# ---------------------------
compute_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {compute_device}")

# Determine the torch dtype based on device capability
precision_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

pretrained_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
main_processor = AutoProcessor.from_pretrained(pretrained_model_name, trust_remote_code=True)
vision_language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name,
    torch_dtype=precision_type,
    device_map="auto",
    trust_remote_code=True
)

# ---------------------------
# 4️⃣ Custom Collate Function
# ---------------------------
def batch_assembler(data_batch):
    # Separate the different components of the batch
    token_ids_list = [item['input_ids'] for item in data_batch]
    attention_mask_list = [item['attention_mask'] for item in data_batch]
    image_pixel_data = torch.stack([item['pixel_values'] for item in data_batch])
    target_values = torch.stack([item['price'] for item in data_batch])
    image_layout_info = torch.stack([item['image_grid_thw'] for item in data_batch])


    # Pad input_ids and attention_mask to the length of the longest sequence in the batch
    padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=main_processor.tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return {
        'input_ids': padded_token_ids,
        'attention_mask': padded_attention_mask,
        'pixel_values': image_pixel_data,
        'image_grid_thw': image_layout_info,
        'price': target_values
    }

# ---------------------------
# 5️⃣ Regression Head + LoRA Configuration
# ---------------------------
# Freeze all parameters of the base model
for param in vision_language_model.parameters():
    param.requires_grad = False

# Add a deeper, custom regression head on top of the model
regressor_hidden_layer_size = 512 # Intermediate dimension for the regression head
vision_language_model.price_estimation_head = nn.Sequential(
    nn.Linear(vision_language_model.config.hidden_size, regressor_hidden_layer_size),
    nn.ReLU(),
    nn.Linear(regressor_hidden_layer_size, 1)
).to(compute_device)


# Configure LoRA (Low-Rank Adaptation)
lora_adapter_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    use_dora=False,
)
vision_language_model = get_peft_model(vision_language_model, lora_adapter_config)
vision_language_model.print_trainable_parameters()

# ---------------------------
# 6️⃣ Dataset + DataLoader
# ---------------------------
ITEMS_PER_BATCH = 4
ACCUMULATION_INTERVAL = 1

training_data_source = ProductDataHandler(training_set_df, main_processor)
validation_data_source = ProductDataHandler(validation_set_df, main_processor)

training_data_loader = DataLoader(training_data_source, batch_size=ITEMS_PER_BATCH, shuffle=True, collate_fn=batch_assembler)
validation_data_loader = DataLoader(validation_data_source, batch_size=ITEMS_PER_BATCH * 2, shuffle=False, collate_fn=batch_assembler)

# ---------------------------
# 7️⃣ Optimizer + Loss + GradScaler
# ---------------------------
model_optimizer = torch.optim.AdamW(vision_language_model.parameters(), lr=5e-5)
criterion = nn.L1Loss() 
gradient_scaler = GradScaler(enabled=(compute_device == 'cuda'))


total_training_epochs = 3

for current_epoch in range(total_training_epochs):
    vision_language_model.train()
    progress_bar = tqdm(training_data_loader, leave=True)
    cumulative_loss = 0
    for batch_index, current_batch in enumerate(progress_bar):
        # Move batch to the correct device
        token_ids = current_batch['input_ids'].to(compute_device)
        mask_for_attention = current_batch['attention_mask'].to(compute_device)
        pixel_data = current_batch['pixel_values'].to(compute_device)
        grid_info = current_batch['image_grid_thw'].to(compute_device)
        ground_truth_prices = current_batch['price'].to(compute_device) # Prices are in log-scale
        
        with autocast(compute_device, dtype=precision_type, enabled=(compute_device == 'cuda')):
            model_outputs = vision_language_model(
                input_ids=token_ids,
                attention_mask=mask_for_attention,
                pixel_values=pixel_data,
                image_grid_thw=grid_info,
                output_hidden_states=True
            )

            final_hidden_layer = model_outputs.hidden_states[-1]
            
            final_token_positions = mask_for_attention.sum(dim=1) - 1
            aggregated_representation = final_hidden_layer[
                torch.arange(final_hidden_layer.shape[0], device=compute_device), final_token_positions
            ]

            # Predictions are in log-scale
            predictions = vision_language_model.price_estimation_head(aggregated_representation).squeeze(-1)

            # Calculate loss on log-scale values
            batch_loss = criterion(predictions, ground_truth_prices)
        
        batch_loss = batch_loss / ACCUMULATION_INTERVAL
        
        gradient_scaler.scale(batch_loss).backward()

        if (batch_index + 1) % ACCUMULATION_INTERVAL == 0:
            gradient_scaler.unscale_(model_optimizer)
            torch.nn.utils.clip_grad_norm_(vision_language_model.parameters(), max_norm=1.0)
            
            gradient_scaler.step(model_optimizer)
            
            gradient_scaler.update()
            
            model_optimizer.zero_grad()
        
        cumulative_loss += batch_loss.item() * ACCUMULATION_INTERVAL
        progress_bar.set_description(f"Epoch {current_epoch+1}/{total_training_epochs}")
        progress_bar.set_postfix(loss=batch_loss.item() * ACCUMULATION_INTERVAL)

    # --- Evaluation with SMAPE ---
    vision_language_model.eval()
    all_predictions, all_ground_truths = [], []
    with torch.no_grad():
        for evaluation_batch in tqdm(validation_data_loader, desc="Evaluating"):
            eval_token_ids = evaluation_batch['input_ids'].to(compute_device)
            eval_attention_mask = evaluation_batch['attention_mask'].to(compute_device)
            eval_pixel_values = evaluation_batch['pixel_values'].to(compute_device)
            eval_grid_info = evaluation_batch['image_grid_thw'].to(compute_device)
            eval_prices = evaluation_batch['price'].to(compute_device) # Prices are in log-scale

            with autocast(compute_device, dtype=precision_type, enabled=(compute_device == 'cuda')):
                eval_outputs = vision_language_model(
                    input_ids=eval_token_ids,
                    attention_mask=eval_attention_mask,
                    pixel_values=eval_pixel_values,
                    image_grid_thw=eval_grid_info,
                    output_hidden_states=True
                )
                
                eval_last_hidden = eval_outputs.hidden_states[-1]
                eval_last_indices = eval_attention_mask.sum(dim=1) - 1
                eval_pooled_embedding = eval_last_hidden[
                    torch.arange(eval_last_hidden.shape[0], device=compute_device), eval_last_indices
                ]
                
                # Predictions are in log-scale
                eval_preds = vision_language_model.price_estimation_head(eval_pooled_embedding).squeeze(-1)

            all_predictions.extend(eval_preds.cpu().tolist())
            all_ground_truths.extend(eval_prices.cpu().tolist())

    # --- CHANGE: Convert predictions and targets back from log-scale before calculating metric ---
    predictions_tensor = torch.expm1(torch.tensor(all_predictions))
    ground_truths_tensor = torch.expm1(torch.tensor(all_ground_truths))
    
    # Calculate SMAPE on the original price scale
    stability_term = 1e-8
    smape_score = 100 * torch.mean(2 * torch.abs(predictions_tensor - ground_truths_tensor) / (torch.abs(ground_truths_tensor) + torch.abs(predictions_tensor) + stability_term))
    print(f"\nEpoch {current_epoch+1} | Avg Train Loss: {cumulative_loss / len(training_data_loader):.4f} | Test SMAPE: {smape_score:.4f}%\n")

    # ---------------------------
    # 9️⃣ Saving the Model Every Epoch
    # ---------------------------
    print(f"--- Saving model checkpoint for epoch {current_epoch+1} ---")
    checkpoint_directory = f"./price_predictor_epoch_{current_epoch+1}"
    os.makedirs(checkpoint_directory, exist_ok=True)

    vision_language_model.save_pretrained(checkpoint_directory)
    torch.save(vision_language_model.price_estimation_head.state_dict(), os.path.join(checkpoint_directory, "regression_head.pth"))
    main_processor.save_pretrained(checkpoint_directory)

    print(f"✅ Model for epoch {current_epoch+1} saved to {checkpoint_directory}")


# ---------------------------
# 1️⃣0️⃣ Inference Function
# ---------------------------
def estimate_cost(description, image_source_url):
    """Predicts the price for a single product given its details and image URL."""
    vision_language_model.eval()
    
    # Download and process the image
    try:
        web_response = requests.get(image_source_url, timeout=10)
        web_response.raise_for_status()
        inference_image = Image.open(BytesIO(web_response.content)).convert("RGB")
        inference_image = inference_image.resize((224, 224))
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error fetching image for prediction: {e}")
        return None

    inference_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Catalog: {description}\nInstruction: Encode product for price prediction."}
            ]
        }
    ]
    inference_prompt = main_processor.apply_chat_template(
        inference_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    inference_inputs = main_processor(text=inference_prompt, images=inference_image, return_tensors="pt")
    inference_inputs = {k: v.to(compute_device) for k, v in inference_inputs.items()}

    with torch.no_grad():
        with autocast(compute_device, dtype=precision_type, enabled=(compute_device == 'cuda')):
            inference_outputs = vision_language_model(
                input_ids=inference_inputs['input_ids'],
                attention_mask=inference_inputs['attention_mask'],
                pixel_values=inference_inputs['pixel_values'].to(compute_device),
                image_grid_thw=inference_inputs['image_grid_thw'],
                output_hidden_states=True
            )
            final_hidden_output = inference_outputs.hidden_states[-1]
            
            final_token_index = inference_inputs['attention_mask'].sum(dim=1) - 1
            final_embedding = final_hidden_output[
                torch.arange(final_hidden_output.shape[0], device=compute_device), final_token_index
            ]
            
            # Prediction from model is in log-scale
            log_scale_prediction = vision_language_model.price_estimation_head(final_embedding).squeeze(-1)
            
            # --- CHANGE: Convert prediction back from log-scale to original price scale ---
            final_prediction = torch.expm1(log_scale_prediction).item()
            
    return final_prediction

# Example usage:
print("\n--- Running Inference Example ---")
# Note: This will use the model from the *last* trained epoch
sample_description = "La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)"
sample_image_link = "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg"
estimated_price = estimate_cost(sample_description, sample_image_link)

if estimated_price is not None:
    print(f"Predicted price for '{sample_description}': ${estimated_price:.2f}")
