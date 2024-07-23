from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, TrainingArguments, Trainer
import torch
import os

def encode_data(df, tokenizer):
    inputs = tokenizer(
        df['context'].tolist(),
        df['question'].tolist(),
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    start_positions = torch.tensor(df['start_position'].tolist())
    end_positions = torch.tensor(df['end_position'].tolist())
    
    return inputs, start_positions, end_positions

def train_model(train_dataset, model_path='./model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        print("Modelo cargado desde el directorio guardado.")
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        print("Modelo cargado desde el modelo base de Huggingface.")
    
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Modelo y tokenizador guardados en el directorio.")

    return model, tokenizer
