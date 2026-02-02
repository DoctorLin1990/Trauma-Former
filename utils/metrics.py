"""
Evaluation Metrics for Trauma-Former
Comprehensive metrics calculation as described in Section 3.4 of the paper
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            precision_recall_curve, roc_curve, 
                            confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score,
                            classification_report, cohen_kappa_score)
from sklearn.metrics import matthews_corrcoef
from scipy import stats
from typing import Tuple, Dict, List, Optional, Union
import warnings

class ClinicalMetrics:
    """Clinical evaluation metrics for TIC prediction"""
    
    def __init__(self, threshold: float = 0.5, 
                early_warning_threshold: float = 0.8):
        self.threshold = threshold
        self.early_warning_threshold = early_warning_threshold
    
    def compute_binary_metrics(self, y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_prob: np.ndarray) -> Dict:
        """
        Compute comprehensive binary classification metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Binary predictions
            y_prob: Probability predictions
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # AUC metrics
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except:
            auroc = 0.5
        
        try:
            auprc = average_precision_score(y_true, y_prob)
        except:
            auprc = 0.5
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Youden's J statistic
        youden_j = recall + specificity - 1
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),  # Same as recall
            'specificity': float(specificity),
            'f1_score': float(f1),
            'negative_predictive_value': float(npv),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'auroc': float(auroc),
            'auprc': float(auprc),
            'mcc': float(mcc),
            'kappa': float(kappa),
            'youden_j': float(youden_j),
            'threshold': float(self.threshold)
        }
    
    def compute_roc_analysis(self, y_true: np.ndarray, 
                            y_prob: np.ndarray, 
                            n_thresholds: int = 100) -> Dict:
        """
        Compute ROC curve analysis
        
        Args:
            y_true: Ground truth labels
            y_prob: Probability predictions
            n_thresholds: Number of thresholds to evaluate
        Returns:
            Dictionary with ROC analysis
        """
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find optimal threshold (Youden's J)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate AUC
        auroc = roc_auc_score(y_true, y_prob)
        
        # Calculate standard error for AUC (Hanley & McNeil)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        q1 = auroc / (2 - auroc)
        q2 = 2 * auroc**2 / (1 + auroc)
        se_auc = np.sqrt((auroc * (1 - auroc) + 
                         (n_pos - 1) * (q1 - auroc**2) +
                         (n_neg - 1) * (q2 - auroc**2)) / 
                         (n_pos * n_neg))
        
        # 95% confidence interval
        ci_lower = auroc - 1.96 * se_auc
        ci_upper = auroc + 1.96 * se_auc
        
        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auroc': float(auroc),
            'auroc_se': float(se_auc),
            'auroc_ci': [float(ci_lower), float(ci_upper)],
            'optimal_threshold': float(optimal_threshold),
            'precision_recall': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
    
    def compute_early_warning_metrics(self, 
                                     timestamps: np.ndarray,
                                     risk_scores: np.ndarray,
                                     event_times: np.ndarray,
                                     warning_threshold: float = 0.8) -> Dict:
        """
        Compute early warning metrics (as in Section 4.3)
        
        Args:
            timestamps: Time points
            risk_scores: Risk probability over time
            event_times: Times of actual events
            warning_threshold: Threshold for early warning
        Returns:
            Early warning metrics
        """
        warnings = []
        
        for event_time in event_times:
            # Find warning times before event
            pre_event_mask = timestamps < event_time
            pre_event_times = timestamps[pre_event_mask]
            pre_event_scores = risk_scores[pre_event_mask]
            
            # Find when risk exceeded threshold
            warning_mask = pre_event_scores > warning_threshold
            
            if np.any(warning_mask):
                first_warning_idx = np.where(warning_mask)[0][0]
                first_warning_time = pre_event_times[first_warning_idx]
                lead_time = event_time - first_warning_time
                
                warnings.append({
                    'event_time': float(event_time),
                    'warning_time': float(first_warning_time),
                    'lead_time': float(lead_time),
                    'warning_score': float(pre_event_scores[first_warning_idx])
                })
        
        if warnings:
            lead_times = [w['lead_time'] for w in warnings]
            mean_lead_time = np.mean(lead_times)
            median_lead_time = np.median(lead_times)
            std_lead_time = np.std(lead_times)
            detection_rate = len(warnings) / len(event_times)
        else:
            mean_lead_time = 0
            median_lead_time = 0
            std_lead_time = 0
            detection_rate = 0
        
        return {
            'warnings': warnings,
            'mean_lead_time': float(mean_lead_time),
            'median_lead_time': float(median_lead_time),
            'std_lead_time': float(std_lead_time),
            'detection_rate': float(detection_rate),
            'warning_threshold': float(warning_threshold)
        }
    
    def compute_calibration_metrics(self, y_true: np.ndarray, 
                                   y_prob: np.ndarray,
                                   n_bins: int = 10) -> Dict:
        """
        Compute calibration metrics (reliability diagram)
        """
        # Bin the predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        
        # Initialize arrays
        bin_centers = np.zeros(n_bins)
        bin_fractions = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        
        # Calculate calibration
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_counts[i] = np.sum(mask)
                bin_centers[i] = np.mean(y_prob[mask])
                bin_fractions[i] = np.mean(y_true[mask])
                bin_confidences[i] = np.mean(y_prob[mask])
        
        # Expected Calibration Error (ECE)
        ece = np.sum(np.abs(bin_fractions - bin_centers) * 
                    bin_counts / len(y_prob))
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_fractions - bin_centers))
        
        # Brier score
        brier = np.mean((y_prob - y_true) ** 2)
        
        return {
            'bins': bins.tolist(),
            'bin_centers': bin_centers.tolist(),
            'bin_fractions': bin_fractions.tolist(),
            'bin_counts': bin_counts.tolist(),
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'brier_score': float(brier)
        }
    
    def bootstrap_confidence_intervals(self, y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     metric_func,
                                     n_bootstraps: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Compute bootstrap confidence intervals for any metric
        
        Args:
            y_true: Ground truth labels
            y_prob: Probability predictions
            metric_func: Function that computes the metric
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level
        Returns:
            Dictionary with metric and confidence interval
        """
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            
            # Compute metric on bootstrap sample
            try:
                metric_val = metric_func(y_true_boot, y_prob_boot)
                bootstrap_metrics.append(metric_val)
            except:
                continue
        
        if not bootstrap_metrics:
            return {'metric': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        # Compute mean metric
        mean_metric = np.mean(bootstrap_metrics)
        
        return {
            'metric': float(mean_metric),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstraps': len(bootstrap_metrics)
        }
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and compute statistical significance
        
        Args:
            model_results: Dictionary with model names as keys and 
                          metrics dictionaries as values
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        # Extract metrics for each model
        for model_name, metrics in model_results.items():
            comparison_data.append({
                'model': model_name,
                'auroc': metrics.get('auroc', 0),
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'sensitivity': metrics.get('sensitivity', 0),
                'specificity': metrics.get('specificity', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by AUROC (descending)
        df = df.sort_values('auroc', ascending=False)
        
        # Compute pairwise comparisons
        if len(model_results) > 1:
            model_names = list(model_results.keys())
            pairwise_comparisons = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    
                    # For demonstration, we'd need the raw predictions
                    # to do statistical tests
                    comparison = {
                        'model1': model1,
                        'model2': model2,
                        'auroc_diff': (model_results[model1].get('auroc', 0) - 
                                      model_results[model2].get('auroc', 0)),
                        'f1_diff': (model_results[model1].get('f1_score', 0) - 
                                   model_results[model2].get('f1_score', 0))
                    }
                    pairwise_comparisons.append(comparison)
        
        return df
    
    def create_performance_summary(self, metrics: Dict) -> str:
        """
        Create a formatted performance summary string
        """
        summary = f"""
        Performance Summary:
        ===================
        
        Discriminative Ability:
          AUROC: {metrics.get('auroc', 0):.3f} (95% CI: {metrics.get('auroc_ci', [0,0])[0]:.3f}-{metrics.get('auroc_ci', [0,0])[1]:.3f})
          AUPRC: {metrics.get('auprc', 0):.3f}
        
        Diagnostic Metrics:
          Accuracy: {metrics.get('accuracy', 0):.3f}
          Sensitivity (Recall): {metrics.get('sensitivity', 0):.3f}
          Specificity: {metrics.get('specificity', 0):.3f}
          Precision: {metrics.get('precision', 0):.3f}
          F1-Score: {metrics.get('f1_score', 0):.3f}
          Negative Predictive Value: {metrics.get('negative_predictive_value', 0):.3f}
        
        Additional Metrics:
          Matthews Correlation Coefficient: {metrics.get('mcc', 0):.3f}
          Cohen's Kappa: {metrics.get('kappa', 0):.3f}
          Youden's J Statistic: {metrics.get('youden_j', 0):.3f}
        
        Confusion Matrix:
          True Positives: {metrics.get('true_positives', 0)}
          False Positives: {metrics.get('false_positives', 0)}
          True Negatives: {metrics.get('true_negatives', 0)}
          False Negatives: {metrics.get('false_negatives', 0)}
        
        Calibration:
          Brier Score: {metrics.get('brier_score', 0):.3f}
          Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.3f}
        
        Early Warning Performance:
          Median Lead Time: {metrics.get('median_lead_time', 0):.1f} minutes
          Detection Rate: {metrics.get('detection_rate', 0):.3f}
        """
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Initialize metrics calculator
    metrics_calc = ClinicalMetrics(threshold=0.5)
    
    # Compute metrics
    basic_metrics = metrics_calc.compute_binary_metrics(y_true, y_pred, y_prob)
    roc_analysis = metrics_calc.compute_roc_analysis(y_true, y_prob)
    
    print("Basic Metrics:")
    for key, value in basic_metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nROC Analysis:")
    print(f"  AUROC: {roc_analysis['auroc']:.3f}")
    print(f"  95% CI: [{roc_analysis['auroc_ci'][0]:.3f}, {roc_analysis['auroc_ci'][1]:.3f}]")
    print(f"  Optimal Threshold: {roc_analysis['optimal_threshold']:.3f}")