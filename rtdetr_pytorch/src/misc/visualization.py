# %%
import os
import json
import re
import traceback

import matplotlib.pyplot as plt

def visualize_train_loss(epoch_logs):
    x = list(range(1, len(epoch_logs)+1)) 
    y = [i['train_loss'] for i in epoch_logs]  

    plt.figure(figsize=( 3*(int(len(x)/ (10 if len(x)>10 else len(x)))), 2*(int(len(y)/ (20 if len(y)>20 else len(y))))))
    plt.xticks(x, [f"{int(i)}" for i in x])
    plt.plot(x, y)  
    plt.xlabel('Epoch')
    plt.ylabel('Train losses')
    plt.title('Train Losses')
    # plt.legend(['Legends'], fontsize="large")

    for (xi, yi) in zip(x, y):
        plt.text(xi, yi + 0.1, f"{yi:.2f}", ha='center', va='bottom')

    plt.grid(True)  
    return plt

def vizualize_apvals(epoch_logs):
    x = list(range(1, len(epoch_logs)+1))
    ap_labels = ['APval 95', 'APval 50', 'APval 75', 'APval S', 'APval M', 'APval L']
    all_ys = {
        k:[ep['test_coco_eval_bbox'][i] for ep in epoch_logs] for i, k in enumerate(ap_labels)
    }

    y_len = len(epoch_logs[0]['test_coco_eval_bbox'])
    plt.figure(figsize=( 5+3*(int(len(x)/ (10 if len(x)>10 else len(x)))), 3+2*(int(y_len/ (20 if y_len>20 else y_len)))))

    # show all the x axis labels
    for label, y in all_ys.items():
        plt.xticks(x, [f"{int(i)}" for i in x])  
        plt.plot(x, y, label=label)  

    plt.xlabel('Epoch')
    plt.ylabel('AP val')
    plt.title('AP val')
    plt.legend(ap_labels, fontsize="large")
    plt.grid(True)
    return plt


def visualize_train_log(log_path, output_dir):
    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)    

        epoch_logs = []
        with open(log_path, 'r') as log_file:
            log_content = log_file.read().split('\n')
            for line in log_content:
                if re.match(r'^[\s]*$', line):
                    continue
                epoch_logs.append(json.loads(line))

        visualize_train_loss(epoch_logs).savefig(os.path.join(output_dir, 'train_losses.png'))
        vizualize_apvals(epoch_logs).savefig(os.path.join(output_dir, 'apvals.png'))
    except Exception as e:
        print(f'ERROR: visualize_train_log: {str(e)} \n {traceback.format_exc()}')
        
if __name__=='__main__':
    visualize_train_log('/home/kevin/Custom-RT-DETR/rtdetr_pytorch/tools/train_log_parsed_json.txt', 'graphs_output_rtdetr_r50')