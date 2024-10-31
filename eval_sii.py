#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            #attn_implementation='flash_attention_2',
            torch_dtype='auto',
            trust_remote_code=True,
        )
        # Configure logging
        logging.basicConfig(
            filename='app.log',  # Specify the file name for the log
            filemode='w',  # 'a' for appending, 'w' for overwriting
            level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate(model, data_path, batch_size, context_length, prediction_length):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Process on device: {device}')
    print(f'Model: {model}')
    print(f'Data path: {data_path}')
    print(f'Prediction_length: {prediction_length}')
    print(f'Context_length: {context_length}')
    print(f'Batch_size: {batch_size}')

    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        model,
        device,
        context_length=context_length,
        prediction_length=prediction_length
    )
    dataset = BenchmarkEvalDataset(
        data_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )

    acc_count = 0
    all_preds = []
    all_labels = []
    all_inputs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)
            all_preds.append(preds.cpu().to(torch.float32).numpy().tolist())
            all_labels.append(labels.cpu().to(torch.float32).numpy().tolist())
            all_inputs.append(batch['inputs'].cpu().to(torch.float32).numpy().tolist())

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'ret_metric: {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]

    all_stat = metric_tensors

    item = {
        'model': model,
        'data': data_path,
        'context_length': context_length,
        'prediction_length': prediction_length,
    }

    count = all_stat[-1]
    for i, metric in enumerate(metric_list):
        val = all_stat[i] / count
        item[metric.name] = float(val.cpu().numpy())
    logging.info(item)

    return all_labels, all_preds, all_inputs

def get_context_length(pred_length):
    cntx_length = pred_length * 4
    if pred_length == 96:
        cntx_length = 512
    elif pred_length == 192:
        cntx_length = 1024
    elif pred_length == 336:
        cntx_length = 2048
    elif pred_length == 720:
        cntx_length = 3072
    return cntx_length


if __name__ == '__main__':
    prediction_length_list = [30, 60, 90]
    data_list = ['data/sii_total.csv', 'data/sii_total_todas.csv', 'data/sii_manufacturera.csv']
    name_list = ['total', 'total_todas', 'manufacturera']

    # Time MoE model params
    model = 'Maple728/TimeMoE-50M'
    batch_size = 32
    prediction_length = 30
    context_length = get_context_length(prediction_length)
    context_length = 256 # 512 #256 #1024 #mac mem 3.8Gb < 4Gb
    #data_path = 'data/sii_total.csv'
    data_path = 'data/sii_total_nocierre.csv'

    ground_truth_ll, prediction_ll, input_ll = evaluate(model, data_path, batch_size, context_length, prediction_length)

    # paint
    batch_number = -1
    slice_number = -1
    last_ground_truth = ground_truth_ll[batch_number][slice_number]
    last_prediction = prediction_ll[batch_number][slice_number]
    last_input = input_ll[batch_number][slice_number]

    ground_truth = last_input + last_ground_truth
    prediction = last_input + last_prediction

    # Generación de datos simulados
    x = np.linspace(0, len(ground_truth), len(ground_truth))

    # Creación del gráfico
    plt.figure(figsize=(34, 16))
    plt.plot(x, ground_truth, label='Ground Truth', color='blue', linewidth=1)
    plt.plot(x, prediction, label='Prediction', color='orange', linewidth=1)
    plt.legend()

    # Generate tick locations and select matching labels
    x_axis_length = len(ground_truth)
    data = pd.read_csv(data_path)
    last_x_labels = data.iloc[-x_axis_length:, 0].tolist()
    num_ticks = 10
    tick_positions = np.linspace(0, x_axis_length-1, num_ticks).astype(int)
    tick_labels = [last_x_labels[i] for i in tick_positions]
    plt.xticks(ticks=np.linspace(0, x_axis_length-1, num_ticks), labels=tick_labels, rotation=45, ha="right")
    plt.ylim(min(min(prediction), min(ground_truth))*1.1, max(max(prediction), max(ground_truth)) * 1.1)
    plt.xlim(0, x_axis_length * 1.1)

    # Guardar el gráfico en un archivo
    model_name = model.replace('/', '_')
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    fig_name = f'sii_figures/{model_name}_{data_name}_bs{batch_size}_ctx{context_length}_p{prediction_length}.png'
    plt.savefig(fig_name, format='png', dpi=400)

    plt.show()

    print()
