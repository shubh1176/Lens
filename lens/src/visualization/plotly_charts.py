import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import numpy as np

class LensVisualizer:
    """Visualization component for Lens explanations."""
    
    @staticmethod
    def feature_importance_plot(importance_dict: Dict[str, float]) -> go.Figure:
        """Create a bar plot of feature importance."""
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=values,
                text=np.round(values, 3),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def prediction_explanation_plot(explanation: Dict[str, Any]) -> go.Figure:
        """Create a waterfall plot for individual prediction explanation."""
        features = []
        contributions = []
        
        for feat, value in explanation.items():
            features.append(feat)
            contributions.append(value)
            
        fig = go.Figure(go.Waterfall(
            name="Prediction Explanation",
            orientation="v",
            measure=["relative"] * len(features),
            x=features,
            y=contributions,
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        
        fig.update_layout(
            title='Feature Contributions to Prediction',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def counterfactual_plot(original: np.ndarray, 
                           counterfactual: np.ndarray, 
                           feature_names: List[str]) -> go.Figure:
        """Create a comparison plot between original and counterfactual."""
        fig = go.Figure()
        
        # Original values
        fig.add_trace(go.Bar(
            name='Original',
            x=feature_names,
            y=original,
            marker_color='blue'
        ))
        
        # Counterfactual values
        fig.add_trace(go.Bar(
            name='Counterfactual',
            x=feature_names,
            y=counterfactual,
            marker_color='red'
        ))
        
        # Changes
        changes = counterfactual - original
        fig.add_trace(go.Bar(
            name='Changes',
            x=feature_names,
            y=changes,
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Counterfactual Comparison',
            barmode='group',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def error_analysis_plot(error_clusters: Dict[str, Any], 
                          X_incorrect: np.ndarray) -> go.Figure:
        """Create visualization for error analysis."""
        fig = go.Figure()
        
        # Plot error clusters if they exist
        if 'cluster_labels' in error_clusters:
            for i in range(len(error_clusters['centroids'])):
                mask = error_clusters['cluster_labels'] == i
                cluster_points = X_incorrect[mask]
                
                fig.add_trace(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(size=8)
                ))
                
                # Plot centroids
                fig.add_trace(go.Scatter(
                    x=[error_clusters['centroids'][i, 0]],
                    y=[error_clusters['centroids'][i, 1]],
                    mode='markers',
                    name=f'Centroid {i}',
                    marker=dict(
                        symbol='star',
                        size=15,
                        line=dict(width=2)
                    )
                ))
                
        fig.update_layout(
            title='Error Cluster Analysis',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def fairness_metrics_plot(fairness_results: Dict[str, Any]) -> go.Figure:
        """Create visualization for fairness metrics."""
        fig = go.Figure()
        
        for sensitive_feature, metrics in fairness_results.items():
            group_metrics = metrics['group_metrics']
            
            # Plot accuracy comparison
            accuracies = [m['accuracy'] for m in group_metrics.values()]
            groups = list(group_metrics.keys())
            
            fig.add_trace(go.Bar(
                name=f'{sensitive_feature} - Accuracy',
                x=groups,
                y=accuracies,
                text=np.round(accuracies, 3),
                textposition='auto',
            ))
            
            # Plot selection rates
            selection_rates = [m['selection_rate'] for m in group_metrics.values()]
            
            fig.add_trace(go.Bar(
                name=f'{sensitive_feature} - Selection Rate',
                x=groups,
                y=selection_rates,
                text=np.round(selection_rates, 3),
                textposition='auto',
            ))
            
        fig.update_layout(
            title='Fairness Metrics Across Groups',
            barmode='group',
            template='plotly_white',
            xaxis_title='Groups',
            yaxis_title='Metric Value',
            showlegend=True
        )
        
        return fig
        
    @staticmethod
    def fairness_summary_plot(fairness_results: Dict[str, Any]) -> go.Figure:
        """Create summary visualization for fairness metrics."""
        fig = go.Figure()
        
        features = list(fairness_results.keys())
        disparate_impact = [metrics['disparate_impact'] for metrics in fairness_results.values()]
        equal_opportunity = [metrics['equal_opportunity'] for metrics in fairness_results.values()]
        demographic_parity = [metrics['demographic_parity'] for metrics in fairness_results.values()]
        
        fig.add_trace(go.Bar(
            name='Disparate Impact',
            x=features,
            y=disparate_impact,
            text=np.round(disparate_impact, 3),
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Equal Opportunity Diff',
            x=features,
            y=equal_opportunity,
            text=np.round(equal_opportunity, 3),
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Demographic Parity Diff',
            x=features,
            y=demographic_parity,
            text=np.round(demographic_parity, 3),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Fairness Metrics Summary',
            barmode='group',
            template='plotly_white',
            xaxis_title='Sensitive Features',
            yaxis_title='Metric Value',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def drift_analysis_plot(drift_results: Dict[str, Any]) -> go.Figure:
        """Create visualization for drift analysis results."""
        fig = go.Figure()
        
        # Feature drift visualization
        feature_names = list(drift_results['feature_drift'].keys())
        drift_statistics = [d['statistic'] for d in drift_results['feature_drift'].values()]
        p_values = [d['p_value'] for d in drift_results['feature_drift'].values()]
        
        # Bar plot for drift statistics
        fig.add_trace(go.Bar(
            name='Drift Statistic',
            x=feature_names,
            y=drift_statistics,
            text=np.round(drift_statistics, 3),
            textposition='auto',
        ))
        
        # Add significance threshold line
        fig.add_hline(
            y=0.1,  # typical threshold for KS test
            line_dash="dash",
            line_color="red",
            annotation_text="Significance Threshold"
        )
        
        fig.update_layout(
            title='Feature Drift Analysis',
            xaxis_title='Features',
            yaxis_title='Drift Statistic',
            template='plotly_white',
            showlegend=True
        )
        
        return fig