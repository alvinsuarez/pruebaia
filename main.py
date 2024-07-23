import data_processing
import manual_data
import model_training
import qa_system
import pandas as pd
from transformers import DistilBertTokenizer
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, start_positions, end_positions):
        self.encodings = encodings
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['start_positions'] = torch.tensor(self.start_positions[idx])
        item['end_positions'] = torch.tensor(self.end_positions[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

file_path = r"C:\Users\gamin\OneDrive\Documentos\proyectico\encuestas.xlsx"

df_cleaned = data_processing.load_and_clean_data(file_path)
print("Datos cargados desde el archivo Excel:")
print(df_cleaned.head()) 

manual_df = manual_data.load_manual_data()
print("Datos manuales cargados:")
print(manual_df.head()) 


df_combined = pd.concat([df_cleaned, manual_df])

print("Datos combinados:")
print(df_combined.head())


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_inputs, start_positions, end_positions = model_training.encode_data(df_combined, tokenizer)


train_dataset = CustomDataset(train_inputs, start_positions, end_positions)


model, tokenizer = model_training.train_model(train_dataset, model_path='./model')

# Guardar el modelo entrenado
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Cargar el modelo entrenado
model, tokenizer = qa_system.load_model(model_path='./model')

# Hacer preguntas de ejemplo para una nueva empresa
context = "Nombre o razón social de la empresa: manila corp. Dirección: transversal 17 #21-82. Barrio/Comuna: Norte. La constitución de la empresa: Sociedad Limitada. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 5678. Cantidad de empleados que tiene en la empresa: 20. Tiempo de funcionamiento de la empresa en años: 5."

# Pregunta sobre el nombre de la empresa
question = "¿Cuál es el nombre de la empresa?"
answer = qa_system.answer_question(model, tokenizer, question, context)
print(f"Pregunta: {question}")
print(f"Respuesta: {answer}")

# Pregunta sobre la dirección de la empresa
question = "¿Cuál es la dirección de la empresa?"
answer = qa_system.answer_question(model, tokenizer, question, context)
print(f"Pregunta: {question}")
print(f"Respuesta: {answer}")
