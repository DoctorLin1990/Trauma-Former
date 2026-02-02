"""
Attention Visualization for Trauma-Former
Visualization tools for model interpretability as referenced in Section 4.4
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class AttentionVisualizer:
    """Visualizes attention weights from Trauma-Former"""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or ['HR', 'SBP', 'DBP', 'SpO2']
        self.cmap = plt.cm.viridis
        
    def plot_attention_heatmap(self, attention_weights: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              title: str = "Attention Heatmap",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)):
        """
        Plot attention heatmap as in Figure 6C of the paper
        
        Args:
            attention_weights: [n_heads, seq_len, seq_len] or [seq_len, seq_len]
            timestamps: Time labels for x/y axes
            title: Plot title
            save_path: Path to save figure
        """
        if attention_weights.ndim == 3:
            # Multiple attention heads - average them
            attention_weights = attention_weights.mean(axis=0)
        
        seq_len = attention_weights.shape[0]
        
        if timestamps is None:
            timestamps = np.arange(seq_len)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Full attention heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(attention_weights, cmap=self.cmap, aspect='auto')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Time Step')
        ax1.set_title(f'{title} - Full Matrix')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Row-wise attention (average attention to each time step)
        ax2 = axes[0, 1]
        row_means = attention_weights.mean(axis=1)
        ax2.plot(timestamps, row_means, 'b-', linewidth=2)
        ax2.fill_between(timestamps, 0, row_means, alpha=0.3)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Average Attention Weight')
        ax2.set_title('Row-wise Average Attention')
        ax2.grid(True, alpha=0.3)
        
        # 3. Column-wise attention (average attention from each time step)
        ax3 = axes[1, 0]
        col_means = attention_weights.mean(axis=0)
        ax3.plot(timestamps, col_means, 'r-', linewidth=2)
        ax3.fill_between(timestamps, 0, col_means, alpha=0.3)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Average Attention Weight')
        ax3.set_title('Column-wise Average Attention')
        ax3.grid(True, alpha=0.3)
        
        # 4. Attention diagonal (self-attention)
        ax4 = axes[1, 1]
        diagonal = np.diag(attention_weights)
        ax4.plot(timestamps, diagonal, 'g-', linewidth=2)
        ax4.fill_between(timestamps, 0, diagonal, alpha=0.3)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Self-Attention Weight')
        ax4.set_title('Self-Attention (Diagonal)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        return fig
    
    def plot_feature_attention(self, attention_weights: np.ndarray,
                              feature_assignments: List[int],
                              title: str = "Feature-wise Attention",
                              save_path: Optional[str] = None):
        """
        Plot attention weights grouped by features
        
        Args:
            attention_weights: [seq_len, seq_len]
            feature_assignments: List mapping each time step to feature index
            title: Plot title
        """
        num_features = len(set(feature_assignments))
        feature_names = self.feature_names[:num_features]
        
        # Group attention by feature
        feature_attention = np.zeros((num_features, num_features))
        feature_counts = np.zeros((num_features, num_features))
        
        seq_len = attention_weights.shape[0]
        for i in range(seq_len):
            for j in range(seq_len):
                src_feat = feature_assignments[i]
                tgt_feat = feature_assignments[j]
                feature_attention[src_feat, tgt_feat] += attention_weights[i, j]
                feature_counts[src_feat, tgt_feat] += 1
        
        # Average
        feature_attention = np.divide(feature_attention, feature_counts, 
                                     where=feature_counts != 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(feature_attention, cmap=self.cmap)
        
        # Set ticks
        ax.set_xticks(np.arange(num_features))
        ax.set_yticks(np.arange(num_features))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(num_features):
            for j in range(num_features):
                text = ax.text(j, i, f'{feature_attention[i, j]:.3f}',
                              ha="center", va="center", color="w" if feature_attention[i, j] > 0.5 else "black")
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, feature_attention
    
    def plot_attention_temporal_pattern(self, attention_weights: np.ndarray,
                                       vital_signs: np.ndarray,
                                       risk_scores: np.ndarray,
                                       timestamps: np.ndarray,
                                       save_path: Optional[str] = None):
        """
        Plot attention weights with vital signs and risk scores (Figure 6 style)
        
        Args:
            attention_weights: [seq_len, seq_len]
            vital_signs: [seq_len, num_features]
            risk_scores: [seq_len] risk probability over time
            timestamps: Time points
        """
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Vital Signs", "Risk Score", "Attention Weights", 
                          "Feature-wise Attention"),
            vertical_spacing=0.08,
            row_heights=[0.25, 0.15, 0.35, 0.25]
        )
        
        # 1. Vital signs
        for i in range(vital_signs.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=vital_signs[:, i],
                    mode='lines',
                    name=self.feature_names[i],
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Risk score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=risk_scores,
                mode='lines',
                name='Risk Score',
                line=dict(color='red', width=3),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", 
                     annotation_text="Alert Threshold", 
                     annotation_position="top right",
                     row=2, col=1)
        
        # 3. Attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=attention_weights,
                x=timestamps,
                y=timestamps,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Attention")
            ),
            row=3, col=1
        )
        
        # 4. Feature-wise attention (average)
        feature_attention = attention_weights.mean(axis=1)
        if vital_signs.shape[1] == 4:  # Assuming 4 features
            # Assuming equal distribution of features
            steps_per_feature = len(timestamps) // 4
            feature_means = []
            for i in range(4):
                start = i * steps_per_feature
                end = (i + 1) * steps_per_feature
                feature_means.append(feature_attention[start:end].mean())
            
            fig.add_trace(
                go.Bar(
                    x=self.feature_names,
                    y=feature_means,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f'{v:.3f}' for v in feature_means],
                    textposition='auto'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Trauma-Former Attention Analysis",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_xaxes(title_text="Features", row=4, col=1)
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Risk Probability", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Average Attention", row=4, col=1)
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            fig.write_image(save_path)
            print(f"Attention visualization saved to {save_path}")
        
        return fig
    
    def plot_multihead_attention(self, attention_weights: np.ndarray,
                                n_rows: int = 2,
                                n_cols: int = 2,
                                save_path: Optional[str] = None):
        """
        Plot attention weights for multiple heads separately
        """
        n_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten()
        
        for i in range(min(n_heads, len(axes))):
            ax = axes[i]
            im = ax.imshow(attention_weights[i], cmap=self.cmap, aspect='auto')
            ax.set_title(f'Attention Head {i+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Time Step')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_heads, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_attention_correlation(self, attention_weights: np.ndarray,
                                     vital_signs: np.ndarray) -> Dict:
        """
        Compute correlation between attention weights and vital signs
        as in Section 4.4 (Spearman's ρ = 0.68)
        """
        from scipy import stats
        
        results = {}
        
        # Average attention to each time step
        avg_attention = attention_weights.mean(axis=1)
        
        # Compute correlations with each vital sign
        for i, feat_name in enumerate(self.feature_names):
            vital_signal = vital_signs[:, i]
            
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(avg_attention, vital_signal)
            
            # Spearman correlation (non-parametric)
            spearman_corr, spearman_p = stats.spearmanr(avg_attention, vital_signal)
            
            results[feat_name] = {
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p)
            }
        
        return results
    
    def visualize_attention_evolution(self, attention_history: List[np.ndarray],
                                     timestamps: List[np.ndarray],
                                     save_path: Optional[str] = None):
        """
        Visualize how attention evolves over multiple inferences
        """
        n_frames = len(attention_history)
        
        fig, axes = plt.subplots(1, min(3, n_frames), figsize=(15, 5))
        
        if n_frames == 1:
            axes = [axes]
        
        # Plot attention at different time points
        for i, (attention, ts) in enumerate(zip(attention_history[:len(axes)], 
                                                timestamps[:len(axes)])):
            ax = axes[i]
            im = ax.imshow(attention, cmap=self.cmap, aspect='auto')
            ax.set_title(f'Time: {ts[-1]:.1f}s')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Time Step')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_attention_data(self, attention_weights: np.ndarray,
                             vital_signs: np.ndarray,
                             timestamps: np.ndarray,
                             output_path: str):
        """
        Export attention data for further analysis
        """
        # Flatten attention matrix
        seq_len = attention_weights.shape[0]
        rows, cols = np.indices((seq_len, seq_len))
        
        data = {
            'source_time': timestamps[rows.flatten()],
            'target_time': timestamps[cols.flatten()],
            'attention_weight': attention_weights.flatten(),
            'source_feature': np.repeat(self.feature_names, seq_len // len(self.feature_names))[:seq_len][rows.flatten()],
            'target_feature': np.repeat(self.feature_names, seq_len // len(self.feature_names))[:seq_len][cols.flatten()]
        }
        
        df = pd.DataFrame(data)
        
        # Add vital sign values
        for i, feat_name in enumerate(self.feature_names):
            df[f'source_{feat_name}'] = vital_signs[rows.flatten(), i]
            df[f'target_{feat_name}'] = vital_signs[cols.flatten(), i]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Attention data exported to {output_path}")
        
        return df
    
    @staticmethod
    def create_case_study_figure(vital_signs: np.ndarray,
                                 risk_scores: np.ndarray,
                                 attention_weights: np.ndarray,
                                 timestamps: np.ndarray,
                                 feature_names: List[str],
                                 save_path: str):
        """
        Create comprehensive case study figure similar to Figure 6 in the paper
        """
        visualizer = AttentionVisualizer(feature_names)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Vital signs (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i in range(vital_signs.shape[1]):
            ax1.plot(timestamps, vital_signs[:, i], 
                    label=feature_names[i], 
                    color=colors[i],
                    linewidth=2)
        ax1.set_ylabel('Vital Signs')
        ax1.set_xlabel('Time (seconds)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Physiological Timeline')
        
        # 2. Risk score (second row, left)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(timestamps, risk_scores, 'r-', linewidth=3)
        ax2.fill_between(timestamps, 0, risk_scores, alpha=0.3, color='red')
        ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
                   label='Alert Threshold')
        ax2.set_ylabel('Risk Probability')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Dynamic Risk Prediction')
        
        # 3. Attention heatmap (second row, middle and right)
        ax3 = fig.add_subplot(gs[1, 1:])
        im = ax3.imshow(attention_weights.mean(axis=0) if attention_weights.ndim == 3 else attention_weights, 
                       cmap='viridis', aspect='auto')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Time Step')
        ax3.set_title('Attention Weight Heatmap')
        plt.colorbar(im, ax=ax3)
        
        # 4. Feature-wise attention (third row)
        ax4 = fig.add_subplot(gs[2, :])
        feature_attention = []
        if vital_signs.shape[1] == len(feature_names):
            for i in range(vital_signs.shape[1]):
                # Simple correlation between attention and vital sign
                avg_attention = attention_weights.mean(axis=1) if attention_weights.ndim == 3 else attention_weights.mean(axis=1)
                correlation = np.corrcoef(avg_attention, vital_signs[:, i])[0, 1]
                feature_attention.append(abs(correlation))
        
        bars = ax4.bar(feature_names, feature_attention, 
                      color=colors[:len(feature_names)])
        ax4.set_ylabel('Attention Correlation')
        ax4.set_title('Feature-wise Attention Importance')
        
        # Add value labels on bars
        for bar, val in zip(bars, feature_attention):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 5. Early warning analysis (fourth row)
        ax5 = fig.add_subplot(gs[3, :])
        
        # Simulate warning times
        warning_threshold = 0.8
        warning_indices = np.where(risk_scores > warning_threshold)[0]
        
        if len(warning_indices) > 0:
            first_warning = warning_indices[0]
            warning_time = timestamps[first_warning]
            total_time = timestamps[-1]
            
            ax5.axvline(x=warning_time, color='red', linestyle='-', 
                       linewidth=3, label=f'First Alert: {warning_time:.1f}s')
            ax5.axvspan(warning_time, total_time, alpha=0.2, color='red',
                       label=f'Warning Duration: {total_time-warning_time:.1f}s')
        
        ax5.plot(timestamps, risk_scores, 'k-', linewidth=2)
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Risk Probability')
        ax5.set_ylim([0, 1])
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Early Warning Timeline')
        
        plt.suptitle('Trauma-Former Digital Twin Case Study', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Case study figure saved to {save_path}")
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    seq_len = 30
    timestamps = np.arange(seq_len)
    
    # Sample vital signs
    vital_signs = np.column_stack([
        np.random.normal(90, 10, seq_len),  # HR
        np.random.normal(120, 15, seq_len), # SBP
        np.random.normal(80, 10, seq_len),  # DBP
        np.random.normal(98, 2, seq_len)    # SpO2
    ])
    
    # Sample risk scores (increasing trend)
    risk_scores = np.linspace(0.1, 0.9, seq_len) + np.random.normal(0, 0.1, seq_len)
    risk_scores = np.clip(risk_scores, 0, 1)
    
    # Sample attention weights
    attention_weights = np.random.rand(4, seq_len, seq_len)  # 4 attention heads
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(['HR', 'SBP', 'DBP', 'SpO2'])
    
    # Plot attention heatmap
    visualizer.plot_attention_heatmap(
        attention_weights,
        timestamps=timestamps,
        title="Trauma-Former Attention",
        save_path="attention_heatmap.png"
    )
    
    # Compute correlations
    correlations = visualizer.compute_attention_correlation(
        attention_weights.mean(axis=0),
        vital_signs
    )
    
    print("Attention-Vital Sign Correlations:")
    for feat, stats in correlations.items():
        print(f"  {feat}: Spearman ρ = {stats['spearman_correlation']:.3f} "
              f"(p = {stats['spearman_p_value']:.4f})")
    
    # Create comprehensive case study figure
    visualizer.create_case_study_figure(
        vital_signs=vital_signs,
        risk_scores=risk_scores,
        attention_weights=attention_weights,
        timestamps=timestamps,
        feature_names=['HR', 'SBP', 'DBP', 'SpO2'],
        save_path="case_study_figure.png"
    )