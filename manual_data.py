import pandas as pd
from transformers import DistilBertTokenizer

def load_manual_data():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    manual_data = [
    {
        'context': "Nombre o razón social de la empresa: Green Tech. Dirección: Avenida del Bosque 55. Barrio/Comuna: Sur. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 3210. Cantidad de empleados que tiene en la empresa: 40. Tiempo de funcionamiento de la empresa en años: 12.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Green Tech"
    },
    {
        'context': "Nombre o razón social de la empresa: Green Tech. Dirección: Avenida del Bosque 55. Barrio/Comuna: Sur. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 3210. Cantidad de empleados que tiene en la empresa: 40. Tiempo de funcionamiento de la empresa en años: 12.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Avenida del Bosque 55"
    },
    {
        'context': "Nombre o razón social de la empresa: Solar Solutions. Dirección: Calle Solar 33. Barrio/Comuna: Este. La constitución de la empresa: Sociedad Limitada. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 5674. Cantidad de empleados que tiene en la empresa: 25. Tiempo de funcionamiento de la empresa en años: 7.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Solar Solutions"
    },
    {
        'context': "Nombre o razón social de la empresa: Solar Solutions. Dirección: Calle Solar 33. Barrio/Comuna: Este. La constitución de la empresa: Sociedad Limitada. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 5674. Cantidad de empleados que tiene en la empresa: 25. Tiempo de funcionamiento de la empresa en años: 7.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Calle Solar 33"
    },
    {
        'context': "Nombre o razón social de la empresa: Blue Ocean. Dirección: Calle del Mar 78. Barrio/Comuna: Norte. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 7890. Cantidad de empleados que tiene en la empresa: 100. Tiempo de funcionamiento de la empresa en años: 20.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Blue Ocean"
    },
    {
        'context': "Nombre o razón social de la empresa: Blue Ocean. Dirección: Calle del Mar 78. Barrio/Comuna: Norte. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 7890. Cantidad de empleados que tiene en la empresa: 100. Tiempo de funcionamiento de la empresa en años: 20.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Calle del Mar 78"
    },
    {
        'context': "Nombre o razón social de la empresa: Creative Minds. Dirección: Avenida Creativa 101. Barrio/Comuna: Oeste. La constitución de la empresa: Persona Jurídica. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 3344. Cantidad de empleados que tiene en la empresa: 10. Tiempo de funcionamiento de la empresa en años: 5.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Creative Minds"
    },
    {
        'context': "Nombre o razón social de la empresa: Creative Minds. Dirección: Avenida Creativa 101. Barrio/Comuna: Oeste. La constitución de la empresa: Persona Jurídica. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 3344. Cantidad de empleados que tiene en la empresa: 10. Tiempo de funcionamiento de la empresa en años: 5.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Avenida Creativa 101"
    },
    {
        'context': "Nombre o razón social de la empresa: Health First. Dirección: Calle Salud 11. Barrio/Comuna: Centro. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 2233. Cantidad de empleados que tiene en la empresa: 80. Tiempo de funcionamiento de la empresa en años: 15.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Health First"
    },
    {
        'context': "Nombre o razón social de la empresa: Health First. Dirección: Calle Salud 11. Barrio/Comuna: Centro. La constitución de la empresa: Sociedad Anónima. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 2233. Cantidad de empleados que tiene en la empresa: 80. Tiempo de funcionamiento de la empresa en años: 15.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Calle Salud 11"
    },
    {
        'context': "Nombre o razón social de la empresa: Eco Friendly. Dirección: Avenida Verde 99. Barrio/Comuna: Sur. La constitución de la empresa: Sociedad Limitada. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 5544. Cantidad de empleados que tiene en la empresa: 60. Tiempo de funcionamiento de la empresa en años: 10.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Eco Friendly"
    },
    {
        'context': "Nombre o razón social de la empresa: Eco Friendly. Dirección: Avenida Verde 99. Barrio/Comuna: Sur. La constitución de la empresa: Sociedad Limitada. Tiene establecimiento comercial: Sí. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: Sí. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 5544. Cantidad de empleados que tiene en la empresa: 60. Tiempo de funcionamiento de la empresa en años: 10.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Avenida Verde 99"
    },
    {
        'context': "Nombre o razón social de la empresa: Future Innovations. Dirección: Calle Futuro 88. Barrio/Comuna: Este. La constitución de la empresa: Persona Jurídica. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 7766. Cantidad de empleados que tiene en la empresa: 20. Tiempo de funcionamiento de la empresa en años: 4.",
        'question': "¿Cuál es el nombre de la empresa?",
        'answer': "Future Innovations"
    },
    {
        'context': "Nombre o razón social de la empresa: Future Innovations. Dirección: Calle Futuro 88. Barrio/Comuna: Este. La constitución de la empresa: Persona Jurídica. Tiene establecimiento comercial: No. Su empresa o negocio se encuentra formalizado ante las entidades respectivas?: No. Si conoce el código CIIU de la Actividad de la empresa favor escribirlo: 7766. Cantidad de empleados que tiene en la empresa: 20. Tiempo de funcionamiento de la empresa en años: 4.",
        'question': "¿Cuál es la dirección de la empresa?",
        'answer': "Calle Futuro 88"
    }
]

    
    contexts = []
    questions = []
    start_positions = []
    end_positions = []
    
    for data in manual_data:
        context = data['context']
        question = data['question']
        answer = data['answer']
        
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        
        try:
            start_idx = context_tokens.index(answer_tokens[0])
            end_idx = start_idx + len(answer_tokens) - 1
        except ValueError:
            start_idx = 0
            end_idx = 0
        
        contexts.append(context)
        questions.append(question)
        start_positions.append(start_idx)
        end_positions.append(end_idx)
    
    manual_df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'start_position': start_positions,
        'end_position': end_positions
    })
    
    return manual_df
