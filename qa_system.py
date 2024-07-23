from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def load_model(model_path='./model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.to(device)
    
    return model, tokenizer

def answer_question(model, tokenizer, question, context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    input_ids = inputs['input_ids'].tolist()[0]

    # Obtener las posiciones de inicio y fin de la respuesta
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convertir IDs de tokens de vuelta a texto
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    # Diagnóstico: Imprimir puntuaciones y posiciones
    print(f"Pregunta: {question}")
    print(f"Contexto: {context}")
    print(f"Inicio de la Respuesta: {answer_start}")
    print(f"Fin de la Respuesta: {answer_end}")
    print(f"Respuesta Tokenizada: {tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])}")
    print(f"Respuesta: {answer}")

    return answer
