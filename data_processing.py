import pandas as pd
from transformers import DistilBertTokenizer

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    
    # Imprimir nombres de columnas para verificación
    print("Columnas en el archivo Excel:")
    print(df.columns)
    
    columns_of_interest = [
        'Nombre o razón social de la empresa', 
        'Dirección', 
        'Barrio/Comuna', 
        'La constitución de la empresa', 
        'Tiene establecimiento comercial', 
        'Su empresa o negocio se encuentra formalizado ante las entidades respectivas?', 
        'Si conoce el código CIIU de la Actividad de la empresa favor escribirlo', 
        'Cantidad de empleados que tiene en la empresa', 
        'Tiempo de funcionamiento de la empresa en años'
    ]
    
    df_cleaned = df[columns_of_interest].dropna()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    contexts = []
    questions = []
    start_positions = []
    end_positions = []
    
    for index, row in df_cleaned.iterrows():
        context = " ".join([f"{col}: {row[col]}" for col in columns_of_interest])
        contexts.append(context)
        
        # Pregunta sobre el nombre de la empresa
        questions.append("¿Cuál es el nombre de la empresa?")
        answer = str(row['Nombre o razón social de la empresa'])

        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        
        # Diagnóstico: Imprimir tokens
        print(f"Contexto: {context}")
        print(f"Contexto Tokens: {tokenizer.convert_ids_to_tokens(context_tokens)}")
        print(f"Respuesta: {answer}")
        print(f"Respuesta Tokens: {tokenizer.convert_ids_to_tokens(answer_tokens)}")
        
        try:
            start_idx = context_tokens.index(answer_tokens[0])
            end_idx = start_idx + len(answer_tokens) - 1
            print(f"Posición de Inicio: {start_idx}, Posición de Fin: {end_idx}")
        except ValueError:
            print("Error: No se encontró la respuesta en el contexto")
            start_idx = 0
            end_idx = 0

        start_positions.append(start_idx)
        end_positions.append(end_idx)
        
        # Pregunta sobre la dirección de la empresa
        questions.append("¿Cuál es la dirección de la empresa?")
        answer = str(row['Dirección'])
        
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        
        # Diagnóstico: Imprimir tokens
        print(f"Contexto: {context}")
        print(f"Contexto Tokens: {tokenizer.convert_ids_to_tokens(context_tokens)}")
        print(f"Respuesta: {answer}")
        print(f"Respuesta Tokens: {tokenizer.convert_ids_to_tokens(answer_tokens)}")
        
        try:
            start_idx = context_tokens.index(answer_tokens[0])
            end_idx = start_idx + len(answer_tokens) - 1
            print(f"Posición de Inicio: {start_idx}, Posición de Fin: {end_idx}")
        except ValueError:
            print("Error: No se encontró la respuesta en el contexto")
            start_idx = 0
            end_idx = 0

        start_positions.append(start_idx)
        end_positions.append(end_idx)
    
    df_cleaned_expanded = pd.DataFrame({
        'context': contexts * 2,  # Duplicar contextos para cada pregunta
        'question': questions,
        'start_position': start_positions,
        'end_position': end_positions
    })
    
    return df_cleaned_expanded

file_path = r"C:\Users\gamin\OneDrive\Documentos\proyectico\encuestas.xlsx"
df_cleaned = load_and_clean_data(file_path)
print(df_cleaned.head())  # Imprimir las primeras filas para inspección
