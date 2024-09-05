
import gc, torch
import pandas as pd
gc.collect()
torch.cuda.empty_cache()
!nvidia-smi
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
import lib.INERCOAI as inercoai
ine = inercoai.inerco_ai()
docs = ine.LabSt.GetDocsFromLabelStudio('EtiquetadoSupremoPreguntas_Ingles_1')
label = ine.LabSt.GetLabelsFromProy('EtiquetadoSupremoPreguntas_Ingles_1')
import pandas as pd
from transformers import BertTokenizerFast

# Cargar el tokenizer de BERT
tokenizer = BertTokenizerFast.from_pretrained("../../00GlobalModelos/bert-base-uncased")

# Definir un template para las etiquetas
template = "Guess the label of this question {}"

translaction_dic ={
    'construcción': 'construction', 'referencia': 'reference', 'desmantelamiento': 'decommissioning', 
    'buque-tanque': 'tanker', '00_f1': '00_f1', 'urv': 'urv', 'ucl': 'ucl', 'consecuencia': 'consequence', 
    'ducto': 'pipeline', 'acceso': 'access', 'ambiental': 'environmental', 'pararrayos': 'lightning rod', 
    'cimentación': 'foundation', 'control': 'control', 'carro-tanque': 'tank car', 'drenaje': 'drainage', 
    'distribución': 'distribution', 'conformidad': 'compliance', 'ubicación': 'location', 
    'organismo': 'organism', 'básica': 'basic', 'gas': 'gas', 'emisión': 'emission', 
    'ingeniería': 'engineering', 'mantenimiento': 'maintenance', 'entrega': 'delivery', 'dique': 'dock', 
    'auto-tanque': 'auto-tank', 'pre-arranque': 'pre-startup', 'inflamable': 'flammable', 
    'biocombustible': 'biofuel', 'detalle': 'detail', 'cov': 'voc', 'estacionamiento': 'parking', 'diseño': 'design', '00_f2': '00_f2', 'emergencia': 'emergency', 'excepciones': 'exceptions', 
    'señalización': 'signage', 'eléctrica': 'electrical', 'lopa': 'lopa', 'combustible': 'fuel', 
    'subterráneo': 'underground', 'almacenamiento': 'storage', 'instalación': 'installation', 
    'vialidad': 'roadway', 'bomba': 'pump', '00_f7': '00_f7', '00_f4': '00_f4', 'tanque': 'tank', 
    'riesgo': 'risk', 'boil-over': 'boil-over', 'superficial': 'superficial', 'ambiente': 'environment', 
    'recepción': 'reception', 'distanciamiento': 'distancing', 'contraincendios': 'fire protection', 
    'oxigenante': 'oxygenate', '00_f3': '00_f3', 'tubería': 'pipeline', 'dictamen': 'report', 
    'líquido': 'liquid', '00_f5': '00_f5', '00_f6': '00_f6', 'cierre': 'closure', 'alcance': 'scope', 
    'humo': 'smoke', 'aditivos': 'additives', 'definición': 'definition', 'predio': 'property', 
    'operación': 'operation', 'srv': 'srv', 'corrosión': 'corrosion', 'evaluación': 'evaluation', 
    'fuego': 'fire', 'monoboya': 'monobuoy', 'válvula': 'valve'
}

# Traducir las etiquetas y construir el DataFrame
new_df = pd.DataFrame([{'Pregunta': x.page_content, 'labels': [translation_dict.get(label, label) for label in x.metadata['annotations']]} for x in docs])

# Desordenar el DataFrame
new_df = new_df.sample(frac=1).reset_index(drop=True)

# Crear una lista de todas las etiquetas
labels = []
for row in new_df['labels']:
    labels.extend(row)
labels = list(set(labels))  # Eliminar duplicados

# Crear diccionarios para mapear etiquetas a enteros y viceversa
label_to_int = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

# Añadir una columna por cada etiqueta al DataFrame
for i in labels:
    new_df[i] = new_df['labels'].apply(lambda x: i in x)

# Eliminar la columna 'labels'
new_df.drop('labels', axis=1, inplace=True)

# Mostrar el DataFrame actualizado
print(new_df.head())
import pandas as pd

# Supongamos que new_df contiene las muestras y las clases están representadas como columnas booleanas

# Calcula el número total de muestras
total_samples = len(new_df)

# Calcula el número de muestras para cada clase
class_counts = new_df.iloc[:, 1:].sum()  # Suponiendo que las columnas de las clases comienzan desde la segunda columna

# Muestra el número de muestras para cada clase y la proporción relativa
print("Número de muestras para cada clase:")
print(class_counts)

# Calcula la proporción de muestras para cada clase
class_proportions = class_counts / total_samples

# Muestra la proporción de muestras para cada clase
print("\nProporción de muestras para cada clase:")
print(class_proportions)

# Calcula la clase con el menor número de muestras
min_samples_class = class_counts.idxmin()

# Calcula la clase con el mayor número de muestras
max_samples_class = class_counts.idxmax()

# Muestra la clase con el menor número de muestras
print("\nClase con el menor número de muestras:")
print(min_samples_class)

# Muestra la clase con el mayor número de muestras
print("\nClase con el mayor número de muestras:")
print(max_samples_class)
import pandas as pd
from sklearn.utils import resample

# Supongamos que 'new_df' ya está definido y tiene la estructura adecuada.
print(new_df.head())

# Calcular el número máximo de muestras entre todas las clases
max_samples = new_df.drop('Pregunta', axis=1).sum(axis=0).max()

# Realizar el oversampling para cada clase
balanced_df = pd.DataFrame()
for label in new_df.columns[1:]:
    # Oversampling la clase actual para que tenga el mismo número de muestras que max_samples
    df_class = new_df[new_df[label] == True]
    df_class_balanced = resample(df_class, replace=True, n_samples=max_samples, random_state=42)
    
    # Agregar las muestras oversampled al DataFrame balanceado
    balanced_df = pd.concat([balanced_df, df_class_balanced])

# Desordenar el DataFrame balanceado
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Mostrar el DataFrame balanceado
print(balanced_df.head())

import pandas as pd
from datasets import Dataset, DatasetDict

# Suponiendo que df_duplicado es tu DataFrame ya duplicado

# Calculamos las dimensiones para el split
train_size = int(len(new_df) * 0.7)
test_size = int(len(new_df) * 0.15)
# El resto para validación
validation_size = len(new_df) - train_size - test_size

# Creamos los datasets divididos
train_dataset = Dataset.from_pandas(new_df[:train_size])
eval_dataset = Dataset.from_pandas(new_df[train_size:train_size+test_size])
validation_dataset = Dataset.from_pandas(new_df[train_size+test_size:])

# Creamos el DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': eval_dataset,
    'validation': validation_dataset
})

# Ahora dataset_dict contiene los splits 'train', 'test' y 'validation'
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Pregunta"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding



encoded_dataset = dataset_dict.map(preprocess_data, batched=True, remove_columns=dataset_dict['train'].column_names)
encoded_dataset.set_format("torch")
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    problem_type="multi_label_classification", 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="bert_Preguntas_Etiquetas_English_Doc_4",  
    evaluation_strategy="epoch",                   
    save_strategy="epoch",                         
    learning_rate=2e-5,                            
    per_device_train_batch_size=batch_size,       
    per_device_eval_batch_size=batch_size,         
    num_train_epochs=5,                            
    weight_decay=0.01,                            
    load_best_model_at_end=True,                   
    metric_for_best_model=metric_name,             
    logging_dir="./logs",                          
    logging_steps=10,                              
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model(f"bert_Preguntas_Etiquetas_English_Doc__4")