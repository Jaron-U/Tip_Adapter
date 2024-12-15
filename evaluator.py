import os
import os.path as osp
from collections import OrderedDict
import pandas as pd

class Evaluator:
    def __init__(self, cfg, label2name, tip_adapter=False):
        self.label2name = label2name
        self.label_length = len(label2name)
        self.correct = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        self.thresholds = cfg['thresholds']
        if tip_adapter:
            self.thresholds = cfg['tip_thresholds']
        self.cfg = cfg
        
        self.results = {
            label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in range(self.label_length)
        }
        self.results_by_threshold = {
            label: [
                {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(len(self.thresholds[label]))
            ] for label in range(self.label_length)
        }
        self.all_thresholds_results = []
        self.best_thresholds_results = []
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        self.results = {
            label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in range(self.label_length)
        }
        self.results_by_threshold = {
            label: [
                {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(len(self.thresholds[label]))
            ] for label in range(self.label_length)
        }
        self.all_thresholds_results = []
        self.best_thresholds_results = []
    
    def process(self, output, gt):
        output, gt = output.detach().cpu().numpy(), gt.detach().cpu().numpy()
        for i in range(len(output)):
            true_label = gt[i]
            for label in range(self.label_length):
                for t_idx, threshold in enumerate(self.thresholds[label]):
                    satisfited = output[i, label] > threshold

                    if true_label == label:
                        if satisfited:
                            self.results_by_threshold[label][t_idx]['TP'] += 1
                        else:
                            self.results_by_threshold[label][t_idx]['FN'] += 1
                    else:
                        if satisfited:
                            self.results_by_threshold[label][t_idx]['FP'] += 1
                        else:
                            self.results_by_threshold[label][t_idx]['TN'] += 1
    
    def evaluate(self, adapter_name="tip_adapter", is_searching=False):
        results = OrderedDict()
        best_thresholds = [{"F1": 0, "Threshold": 0, "Precision": 0, "Recall": 0} for _ in range(self.label_length)]

        for label in range(self.label_length): 
            for t_idx in range(len(self.thresholds[label])): 
                TP = self.results_by_threshold[label][t_idx]["TP"]
                FP = self.results_by_threshold[label][t_idx]["FP"]
                FN = self.results_by_threshold[label][t_idx]["FN"]
                TN = self.results_by_threshold[label][t_idx]["TN"]

                precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
                recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0
                f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

                self.all_thresholds_results.append({
                    "Class": self.label2name[label],
                    "Threshold": self.thresholds[label][t_idx],
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1
                })

                if f1 > best_thresholds[label]["F1"]:
                    best_thresholds[label] = {
                        "F1": f1,
                        "Threshold": self.thresholds[label][t_idx],
                        "Precision": precision,
                        "Recall": recall
                    }

        self.best_thresholds_results = [
            {
                "Label": self.label2name[label],
                "Threshold": best_thresholds[label]["Threshold"],
                "Precision": best_thresholds[label]["Precision"],
                "Recall": best_thresholds[label]["Recall"],
                "F1": best_thresholds[label]["F1"]
            }
            for label in range(self.label_length)
        ]   
        
        if is_searching:
            average_precision = round(sum([x["Precision"] for x in best_thresholds]) / self.label_length, 4)
            average_f1 = round(sum([x["F1"] for x in best_thresholds]) / self.label_length, 4)
            average_recall = round(sum([x["Recall"] for x in best_thresholds]) / self.label_length, 4)
        else:
            average_precision, average_f1, average_recall = self.savefiles(adapter_name)

        results["average_precision"] = average_precision
        results["average_recall"] = average_recall
        results["average_f1"] = average_f1

        return results

    def savefiles(self, adapter_name, shot_num=None):
        output_dir = self.cfg['OUTPUT_DIR']
        if shot_num is not None:
            output_dir = osp.join(output_dir, f"shot_{shot_num}")
            os.makedirs(output_dir, exist_ok=True)

        all_results_df = pd.DataFrame(self.all_thresholds_results)
        all_results_csv_path = osp.join(output_dir, f"all_evaluate_{adapter_name}.csv")
        all_results_df.to_csv(all_results_csv_path, index=False)

        best_thresholds_df = pd.DataFrame(self.best_thresholds_results)
        best_thresholds_csv_path = osp.join(output_dir, f"best_evaluate_{adapter_name}.csv")

        average_precision = round(best_thresholds_df["Precision"].mean(), 4)
        average_f1 = round(best_thresholds_df["F1"].mean(), 4)
        average_recall = round(best_thresholds_df["Recall"].mean(), 4)
        
        best_thresholds_df["average_precision"] = average_precision
        best_thresholds_df["average_recall"] = average_recall
        best_thresholds_df["average_f1"] = average_f1
        best_thresholds_df.to_csv(best_thresholds_csv_path, index=False)

        return average_precision, average_f1, average_recall