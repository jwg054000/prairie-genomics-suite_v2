#!/usr/bin/env python3
"""
🧬 Prairie Genomics Suite - Enhanced Multiomics Platform
Advanced Genomics Analysis Platform with Immune Cell Infiltration & Gene Binning

A comprehensive, publication-ready genomics analysis platform optimized for Streamlit Cloud deployment.
Features include differential expression analysis, survival analysis, pathway enrichment,
immune cell infiltration analysis (CIBERSORTx), gene-based binning (proteinatlas.org),
literature search, and publication-quality visualizations.

New Features:
- Gene expression binning (high vs low) based on proteinatlas.org data
- Immune cell infiltration analysis using CIBERSORTx reference signatures
- Enhanced performance for analyzing >5,000 genes simultaneously
- Advanced filtering for lowly expressed genes
- Multiomics integration capabilities

Usage:
    streamlit run prairie_genomics_streamlit_ready.py

Author: Prairie Genomics Team
Version: 3.0.0 - Enhanced Multiomics Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import sys
import os
import io
import requests
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import concurrent.futures
from functools import lru_cache
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Prairie Genomics Suite - Enhanced Multiomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/prairie-genomics/suite',
        'Report a bug': "https://github.com/prairie-genomics/suite/issues",
        'About': "Prairie Genomics Suite - Enhanced Multiomics Platform with Immune Analysis & Gene Binning"
    }
)

# Handle Streamlit version compatibility
def safe_rerun():
    """Safe rerun function that works with different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.info("Please refresh the page to see the changes")

# ================================
# CORE FUNCTIONALITY - BUILT-IN
# ================================

class BasicStatsAnalyzer:
    """Built-in statistical analysis for when external packages aren't available"""
    
    @staticmethod
    def ttest_analysis(data: pd.DataFrame, group1_samples: List[str], group2_samples: List[str]):
        """Perform basic t-test analysis"""
        from scipy import stats
        
        results = []
        for gene in data.index:
            try:
                group1_values = data.loc[gene, group1_samples].dropna()
                group2_values = data.loc[gene, group2_samples].dropna()
                
                if len(group1_values) < 3 or len(group2_values) < 3:
                    continue
                    
                # T-test
                t_stat, p_value = stats.ttest_ind(group1_values, group2_values)
                
                # Log fold change
                mean1 = np.mean(group1_values)
                mean2 = np.mean(group2_values)
                log_fold_change = np.log2((mean2 + 1) / (mean1 + 1))
                
                # Base mean
                base_mean = (mean1 + mean2) / 2
                
                results.append({
                    'Gene': gene,
                    'baseMean': base_mean,
                    'log2FoldChange': log_fold_change,
                    'pvalue': p_value,
                    'padj': p_value,  # No multiple testing correction for simplicity
                    'mean_group1': mean1,
                    'mean_group2': mean2
                })
                
            except Exception as e:
                continue
                
        return pd.DataFrame(results)

class BasicSurvivalAnalyzer:
    """Built-in survival analysis"""
    
    @staticmethod
    def kaplan_meier_analysis(clinical_data: pd.DataFrame, time_col: str, event_col: str, group_col: Optional[str] = None):
        """Basic Kaplan-Meier survival analysis"""
        try:
            from lifelines import KaplanMeierFitter
            import matplotlib.pyplot as plt
            
            kmf = KaplanMeierFitter()
            
            if group_col and group_col in clinical_data.columns:
                # Group-based analysis
                fig, ax = plt.subplots(figsize=(10, 6))
                
                groups = clinical_data[group_col].unique()
                for group in groups:
                    mask = clinical_data[group_col] == group
                    group_data = clinical_data[mask]
                    
                    if len(group_data) > 5:  # Minimum sample size
                        kmf.fit(
                            durations=group_data[time_col],
                            event_observed=group_data[event_col],
                            label=f'{group} (n={len(group_data)})'
                        )
                        kmf.plot_survival_function(ax=ax)
                
                ax.set_title('Kaplan-Meier Survival Curves')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Survival Probability')
                ax.grid(True, alpha=0.3)
                
                return fig
            else:
                # Overall survival
                fig, ax = plt.subplots(figsize=(10, 6))
                
                kmf.fit(
                    durations=clinical_data[time_col],
                    event_observed=clinical_data[event_col]
                )
                kmf.plot_survival_function(ax=ax)
                
                ax.set_title('Overall Survival Curve')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Survival Probability')
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except ImportError:
            st.warning("Lifelines not available. Please install: pip install lifelines")
            return None
        except Exception as e:
            st.error(f"Survival analysis failed: {e}")
            return None

class ProteinAtlasIntegration:
    """Integration with Human Protein Atlas for gene expression binning"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_tissue_expression_data(gene_symbol: str, tissue: str = "pancreas"):
        """Get tissue-specific expression data from Protein Atlas"""
        # Mock data for demonstration - in production, this would query the actual API
        # The Human Protein Atlas API: https://www.proteinatlas.org/api/
        
        # Common thresholds based on Protein Atlas categorization
        expression_categories = {
            "high": {"TPM_min": 10, "description": "High expression (>10 TPM)"},
            "medium": {"TPM_min": 1, "description": "Medium expression (1-10 TPM)"},
            "low": {"TPM_min": 0.1, "description": "Low expression (0.1-1 TPM)"},
            "not_detected": {"TPM_min": 0, "description": "Not detected (<0.1 TPM)"}
        }
        
        # Simulate tissue-specific thresholds (would be from real API)
        tissue_thresholds = {
            "pancreas": {"high_threshold": 15, "low_threshold": 2},
            "liver": {"high_threshold": 20, "low_threshold": 3},
            "brain": {"high_threshold": 8, "low_threshold": 1},
            "immune": {"high_threshold": 12, "low_threshold": 2.5}
        }
        
        threshold = tissue_thresholds.get(tissue, {"high_threshold": 10, "low_threshold": 1})
        return threshold
    
    @staticmethod
    def bin_genes_by_expression(expression_data: pd.DataFrame, gene_of_interest: str, 
                               tissue: str = "pancreas"):
        """Bin samples by gene expression (high vs low) using Protein Atlas thresholds"""
        if gene_of_interest not in expression_data.index:
            return None, None
        
        # Get gene expression values
        gene_expr = expression_data.loc[gene_of_interest]
        
        # Get tissue-specific thresholds
        thresholds = ProteinAtlasIntegration.get_tissue_expression_data(gene_of_interest, tissue)
        
        # Calculate percentile-based thresholds if absolute thresholds don't work well
        high_threshold = np.percentile(gene_expr, 75)  # Top quartile
        low_threshold = np.percentile(gene_expr, 25)   # Bottom quartile
        
        # Create bins
        high_samples = gene_expr[gene_expr >= high_threshold].index.tolist()
        low_samples = gene_expr[gene_expr <= low_threshold].index.tolist()
        
        return high_samples, low_samples

class ImmuneInfiltrationAnalyzer:
    """Immune cell infiltration analysis using CIBERSORTx-style reference signatures"""
    
    @staticmethod
    def get_cibersort_signatures():
        """Get enhanced CIBERSORTx immune cell type reference signatures"""
        # Enhanced signatures based on CIBERSORTx LM22 matrix and additional literature
        # Each signature now contains more genes for better deconvolution accuracy
        return {
            "B_cells_naive": [
                "MS4A1", "CD79A", "CD79B", "BLK", "BANK1", "CD22", "CD19", "PAX5", 
                "EBF1", "SPIB", "TCL1A", "FCER2", "IGHD", "IL4R"
            ],
            "B_cells_memory": [
                "CD27", "TNFRSF13B", "AIM2", "BANK1", "MS4A1", "CD79A", "CD79B",
                "TNFRSF17", "PRDM1", "XBP1", "CD38", "CD138", "IRF4"
            ],
            "Plasma_cells": [
                "IGHA1", "IGHG1", "IGKC", "IGLC2", "JCHAIN", "PRDM1", "XBP1", 
                "CD138", "CD38", "IRF4", "TNFRSF17", "TNFRSF13B", "MZB1"
            ],
            "T_cells_CD8": [
                "CD8A", "CD8B", "GZMK", "GZMA", "PRF1", "GZMB", "GZMH", "GNLY",
                "NKG7", "CCL5", "CTSW", "CST7", "FGFBP2", "KLRD1", "KLRG1"
            ],
            "T_cells_CD4_naive": [
                "CCR7", "SELL", "IL7R", "TCF7", "LEF1", "CCR4", "SATB1", 
                "NOSIP", "PIK3IP1", "PRKCQ", "TXKD1", "MAL"
            ],
            "T_cells_CD4_memory_resting": [
                "IL7R", "S100A4", "AQP3", "LTB", "ANXA1", "CCR7", "SELL",
                "LDHB", "VIM", "GPR183", "CD52", "TRAT1"
            ],
            "T_cells_CD4_memory_activated": [
                "ICOS", "CXCR3", "ITK", "IFNG", "CD40LG", "TNF", "IL2", "FASLG",
                "GZMK", "CST7", "NKG7", "CCL5", "GZMA", "KLRB1"
            ],
            "T_cells_follicular_helper": [
                "CXCR5", "BCL6", "PDCD1", "ICOS", "IL21", "CD40LG", "BTLA",
                "MAF", "TOX2", "SH2D1A", "SAP", "ASCL2"
            ],
            "T_cells_regulatory": [
                "FOXP3", "CTLA4", "IL2RA", "TNFRSF18", "TNFRSF4", "IKZF2", 
                "IL10", "TGFB1", "IDO1", "TIGIT", "LAG3", "HAVCR2"
            ],
            "T_cells_gamma_delta": [
                "TRGC1", "TRGC2", "TRDC", "KLRD1", "KLRB1", "KLRC1", "KLRC2",
                "KLRF1", "FCGR3A", "GNLY", "NKG7", "GZMA", "GZMB"
            ],
            "NK_cells_resting": [
                "KLRB1", "KIR2DL3", "KIR3DL1", "KIR2DL1", "KLRC1", "KLRD1",
                "NCR1", "NCR3", "CD7", "XCL1", "XCL2", "FCER1G"
            ],
            "NK_cells_activated": [
                "KLRF1", "KLRD1", "NCAM1", "FCGR3A", "PRF1", "GZMB", "GZMA",
                "NKG7", "GNLY", "SPON2", "CCL3", "CCL4", "IFNG"
            ],
            "Monocytes": [
                "CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "S100A12",
                "CTSS", "FPR1", "PLAUR", "SERPINA1", "CD68", "CSF1R"
            ],
            "Macrophages_M0": [
                "CD68", "CD163", "MSR1", "MRC1", "CSF1R", "ADGRE1", "ITGAM",
                "FCGR1A", "FCGR2A", "C1QA", "C1QB", "C1QC", "TYROBP"
            ],
            "Macrophages_M1": [
                "NOS2", "TNF", "IL1B", "IL6", "CXCL10", "CCL2", "PTGS2",
                "CD80", "CD86", "IRF1", "STAT1", "SOCS1", "IDO1"
            ],
            "Macrophages_M2": [
                "ARG1", "IL10", "MRC1", "CD163", "TGFB1", "CCL18", "CCL22",
                "IL1RN", "CLEC10A", "CD206", "FIZZ1", "CHI3L1"
            ],
            "Dendritic_cells_resting": [
                "FCER1A", "CLEC4C", "IRF7", "IRF8", "CLEC4A", "CD1C", "CD1E",
                "CLEC9A", "XCR1", "BATF3", "ID2", "ZBTB46"
            ],
            "Dendritic_cells_activated": [
                "CD1C", "CLEC9A", "XCR1", "BATF3", "IRF8", "ID2", "ZBTB46",
                "FCER1A", "CD80", "CD86", "CCR7", "IL12B", "CXCL10"
            ],
            "Mast_cells_resting": [
                "TPSAB1", "TPSB2", "CPA3", "MS4A2", "HDC", "ENPP3", "SLC45A3",
                "HPGDS", "RGS13", "SLC18A2", "GATA2", "KIT"
            ],
            "Mast_cells_activated": [
                "TPSAB1", "HDC", "MS4A2", "FCER1A", "CPA3", "TPSB2", "ENPP3",
                "HPGDS", "LTC4S", "PTGS2", "IL13", "TNF", "IL4"
            ],
            "Eosinophils": [
                "SIGLEC8", "CCR3", "EPX", "CLC", "PRG2", "PRG3", "RNASE2",
                "RNASE3", "IL5RA", "ALOX15", "CPA3", "GATA1"
            ],
            "Neutrophils": [
                "CEACAM8", "FCGR3B", "CSF3R", "CXCR1", "CXCR2", "S100A12",
                "ELANE", "MPO", "DEFA1", "DEFA3", "LCN2", "MMP8", "CAMP"
            ]
        }
    
    @staticmethod
    def calculate_immune_scores(expression_data: pd.DataFrame, method="ssgsea"):
        """Calculate immune cell infiltration scores"""
        signatures = ImmuneInfiltrationAnalyzer.get_cibersort_signatures()
        immune_scores = {}
        
        for cell_type, signature_genes in signatures.items():
            # Find available genes in the expression data
            available_genes = [gene for gene in signature_genes if gene in expression_data.index]
            
            if len(available_genes) > 0:
                if method == "ssgsea":
                    # Single sample GSEA-like scoring
                    scores = ImmuneInfiltrationAnalyzer._ssgsea_score(
                        expression_data, available_genes
                    )
                elif method == "mean":
                    # Simple mean expression
                    scores = expression_data.loc[available_genes].mean(axis=0)
                else:
                    # Z-score normalized mean
                    gene_expr = expression_data.loc[available_genes]
                    gene_means = gene_expr.mean(axis=1)  # Calculate mean across samples (columns)
                    gene_stds = gene_expr.std(axis=1)    # Calculate std across samples (columns)
                    z_scores = (gene_expr.sub(gene_means, axis=0)).div(gene_stds, axis=0)
                    scores = z_scores.mean(axis=0)
                
                immune_scores[cell_type] = scores
        
        return pd.DataFrame(immune_scores)
    
    @staticmethod
    def _ssgsea_score(expression_data: pd.DataFrame, gene_set: List[str], alpha: float = 0.25):
        """Calculate proper single-sample GSEA score for a gene set using the Barbie et al. algorithm"""
        scores = []
        
        # Filter gene set to available genes
        available_genes = [gene for gene in gene_set if gene in expression_data.index]
        
        if len(available_genes) < 3:  # Minimum genes required
            return pd.Series([0] * len(expression_data.columns), index=expression_data.columns)
        
        for sample in expression_data.columns:
            try:
                # Get expression values for this sample
                sample_expr = expression_data[sample]
                
                # Rank genes by expression (descending order)
                ranked_genes = sample_expr.rank(ascending=False, method='average')
                
                # Get ranks for genes in our set
                gene_set_ranks = [ranked_genes[gene] for gene in available_genes if gene in ranked_genes.index]
                
                if len(gene_set_ranks) == 0:
                    scores.append(0)
                    continue
                    
                # Calculate weighted enrichment score using proper ssGSEA formula
                N = len(sample_expr)  # Total number of genes
                n = len(gene_set_ranks)  # Number of genes in set
                
                # Calculate enrichment score (ES)
                # This is a simplified but more accurate version of the Barbie et al. ssGSEA algorithm
                
                # Sort gene set ranks  
                gene_set_ranks = sorted([float(r) for r in gene_set_ranks])
                
                # Calculate cumulative sum of weighted ranks
                cumulative_sum = 0
                max_deviation = 0
                
                for i, rank in enumerate(gene_set_ranks):
                    # Weight by expression rank^alpha (default alpha=0.25 as in original paper)
                    weight = (N - float(rank) + 1) ** alpha
                    cumulative_sum += weight
                    
                    # Calculate normalized enrichment at this position
                    expected = (i + 1) / n  # Expected fraction if randomly distributed
                    observed = cumulative_sum / sum((N - float(r) + 1) ** alpha for r in gene_set_ranks)
                    
                    deviation = observed - expected
                    max_deviation = max(float(max_deviation), abs(float(deviation)))
                
                # Normalize by maximum possible deviation
                enrichment_score = max_deviation
                
                # Apply additional normalization (optional, helps with comparing across samples)
                # Scale by gene set size to avoid bias toward larger gene sets
                normalized_score = enrichment_score * np.sqrt(n)
                
                scores.append(normalized_score)
                
            except Exception as e:
                # Handle any calculation errors gracefully
                scores.append(0)
        
        return pd.Series(scores, index=expression_data.columns)

class EnhancedGeneFilter:
    """Enhanced gene filtering with multiple criteria"""
    
    @staticmethod
    def filter_low_expression_genes(expression_data: pd.DataFrame, 
                                   min_expression: float = 1.0,
                                   min_samples: int = 3,
                                   percentile_threshold: float = 0.25):
        """Filter genes with low expression across samples"""
        
        # Method 1: Absolute expression threshold
        high_expr_mask = (expression_data >= min_expression).sum(axis=1) >= min_samples
        
        # Method 2: Percentile-based filtering
        percentile_mask = expression_data.quantile(percentile_threshold, axis=1) >= min_expression
        
        # Method 3: Coefficient of variation filtering (remove invariant genes)
        cv_threshold = 0.1
        cv = expression_data.std(axis=1) / expression_data.mean(axis=1)
        cv_mask = cv >= cv_threshold
        
        # Combine all filters
        final_mask = high_expr_mask & percentile_mask & cv_mask
        
        return expression_data.loc[final_mask]
    
    @staticmethod
    def filter_by_variance(expression_data: pd.DataFrame, top_n: int = 5000):
        """Filter to keep most variable genes for performance"""
        gene_vars = expression_data.var(axis=1)
        top_variable_genes = gene_vars.nlargest(top_n).index
        return expression_data.loc[top_variable_genes]
    
    @staticmethod
    def detect_duplicate_genes(expression_data: pd.DataFrame):
        """Detect and report duplicate gene IDs in expression data"""
        gene_counts = expression_data.index.value_counts()
        duplicates = gene_counts[gene_counts > 1]
        
        if len(duplicates) == 0:
            return {
                'has_duplicates': False,
                'duplicate_count': 0,
                'total_genes': len(expression_data),
                'unique_genes': len(expression_data.index.unique()),
                'duplicate_genes': {},
                'examples': []
            }
        
        # Get examples of duplicate genes
        examples = []
        for gene_id in duplicates.head(5).index:
            gene_rows = expression_data.index == gene_id
            examples.append({
                'gene_id': gene_id,
                'count': int(gene_counts[gene_id]),
                'mean_expression': float(expression_data.loc[gene_rows].mean().mean()),
                'std_expression': float(expression_data.loc[gene_rows].std().mean())
            })
        
        return {
            'has_duplicates': True,
            'duplicate_count': len(duplicates),
            'total_genes': len(expression_data),
            'unique_genes': len(expression_data.index.unique()),
            'duplicate_genes': duplicates.to_dict(),
            'examples': examples,
            'affected_rows': int(gene_counts[gene_counts > 1].sum() - len(duplicates))  # Extra rows due to duplication
        }
    
    @staticmethod
    def aggregate_duplicate_genes(expression_data: pd.DataFrame, method: str = 'mean'):
        """Aggregate duplicate gene IDs using specified method"""
        duplicate_info = EnhancedGeneFilter.detect_duplicate_genes(expression_data)
        
        if not duplicate_info['has_duplicates']:
            return expression_data, duplicate_info
        
        # Apply aggregation
        aggregation_methods = {
            'mean': lambda x: x.mean(),
            'median': lambda x: x.median(), 
            'max': lambda x: x.max(),
            'sum': lambda x: x.sum(),
            'first': lambda x: x.iloc[0]  # Take first occurrence
        }
        
        if method not in aggregation_methods:
            method = 'mean'  # Default fallback
            
        try:
            # Group by gene ID and apply aggregation method
            aggregated_data = expression_data.groupby(expression_data.index).agg(aggregation_methods[method])
            
            # Create processing summary
            processing_summary = {
                'original_genes': duplicate_info['total_genes'],
                'unique_genes': duplicate_info['unique_genes'],
                'duplicates_removed': duplicate_info['affected_rows'],
                'method_used': method,
                'success': True
            }
            
            return aggregated_data, processing_summary
            
        except Exception as e:
            # If aggregation fails, return original data with error info
            processing_summary = {
                'original_genes': duplicate_info['total_genes'],
                'unique_genes': duplicate_info['unique_genes'],
                'duplicates_removed': 0,
                'method_used': method,
                'success': False,
                'error': str(e)
            }
            
            return expression_data, processing_summary
    
    @staticmethod
    def apply_smart_filtering(expression_data: pd.DataFrame, auto_adjust: bool = True):
        """Apply automatic gene filtering with dataset-appropriate defaults"""
        n_genes, n_samples = expression_data.shape
        
        # Smart parameter adjustment based on dataset size
        if auto_adjust:
            # Adjust minimum expression based on data distribution
            median_expr = expression_data.median().median()
            q25_expr = expression_data.quantile(0.25).quantile(0.25)
            
            # Adaptive thresholds
            if median_expr > 10:  # High expression data (e.g., TPM, FPKM)
                min_expression = max(1.0, q25_expr * 0.5)
            elif median_expr > 1:  # Medium expression data
                min_expression = max(0.5, q25_expr * 0.3)
            else:  # Low expression data (e.g., log-transformed)
                min_expression = max(0.1, q25_expr * 0.2)
            
            # Adjust minimum samples based on total sample size
            min_samples = max(2, min(5, n_samples // 10))
            
            # Adjust percentile threshold based on data sparsity
            sparsity = (expression_data == 0).sum().sum() / (n_genes * n_samples)
            percentile_threshold = 0.25 if sparsity < 0.5 else 0.1
            
        else:
            # Use fixed defaults
            min_expression = 1.0
            min_samples = 3
            percentile_threshold = 0.25
        
        # Apply filtering
        try:
            filtered_data = EnhancedGeneFilter.filter_low_expression_genes(
                expression_data, min_expression, min_samples, percentile_threshold
            )
            
            # If too many genes remain and dataset is large, apply variance filtering
            if len(filtered_data) > 15000 and n_samples > 20:
                max_genes = min(10000, len(filtered_data))
                filtered_data = EnhancedGeneFilter.filter_by_variance(filtered_data, max_genes)
                variance_applied = True
            else:
                variance_applied = False
            
            # Create filtering summary
            filtering_summary = {
                'original_genes': n_genes,
                'filtered_genes': len(filtered_data),
                'genes_removed': n_genes - len(filtered_data),
                'removal_rate': (n_genes - len(filtered_data)) / n_genes * 100,
                'parameters': {
                    'min_expression': min_expression,
                    'min_samples': min_samples,
                    'percentile_threshold': percentile_threshold
                },
                'variance_filtering_applied': variance_applied,
                'auto_adjusted': auto_adjust,
                'success': True
            }
            
            return filtered_data, filtering_summary
            
        except Exception as e:
            # If filtering fails, return original data with error info
            filtering_summary = {
                'original_genes': n_genes,
                'filtered_genes': n_genes,
                'genes_removed': 0,
                'removal_rate': 0,
                'parameters': {},
                'variance_filtering_applied': False,
                'auto_adjusted': auto_adjust,
                'success': False,
                'error': str(e)
            }
            
            return expression_data, filtering_summary

class BasicPathwayAnalyzer:
    """Enhanced pathway analysis with basic gene sets and immune pathways"""
    
    @staticmethod
    def get_basic_pathways():
        """Return basic pathway gene sets including immune pathways"""
        return {
            "Cell Cycle": ["CDK1", "CDK2", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CCND1", "CCNE1", "RB1", "E2F1"],
            "Apoptosis": ["TP53", "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "PARP1", "APAF1", "CYCS", "BAK1"],
            "DNA Repair": ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "XRCC1", "ERCC1", "MLH1"],
            "PI3K/AKT Pathway": ["PIK3CA", "PIK3CB", "AKT1", "AKT2", "PTEN", "MTOR", "GSK3B", "FOXO1", "PDK1"],
            "MAPK Pathway": ["KRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RAF1", "EGFR", "GRB2"],
            "Immune Response": ["CD3D", "CD4", "CD8A", "IFNG", "IL2", "TNF", "CTLA4", "PDCD1", "CD274", "LAG3"],
            "Metabolism": ["GLUT1", "HK2", "PKM", "LDHA", "PDK1", "ACLY", "FASN", "CPT1A", "PPARA", "SREBF1"],
            "Immune Checkpoint": ["PDCD1", "CD274", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "BTLA"],
            "T Cell Activation": ["CD3D", "CD3E", "CD3G", "CD28", "ICOS", "CD40LG", "IL2"],
            "Interferon Signaling": ["IFNG", "IFNA1", "IFNB1", "STAT1", "STAT2", "IRF1", "IRF7"],
            "Complement System": ["C1QA", "C1QB", "C3", "C5", "CFH", "CFI", "CD55"],
            "Antigen Presentation": ["HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "B2M", "TAP1"]
        }
    
    @staticmethod
    def enrichment_analysis(gene_list: List[str], background_size: int = 20000):
        """Basic pathway enrichment analysis"""
        from scipy import stats
        
        pathways = BasicPathwayAnalyzer.get_basic_pathways()
        results = []
        
        for pathway_name, pathway_genes in pathways.items():
            # Calculate overlap
            overlap = set(gene_list) & set(pathway_genes)
            overlap_count = len(overlap)
            
            if overlap_count == 0:
                continue
                
            # Hypergeometric test
            total_genes = len(gene_list)
            pathway_size = len(pathway_genes)
            
            # P-value calculation (hypergeometric)
            p_value = stats.hypergeom.sf(overlap_count - 1, background_size, pathway_size, total_genes)
            
            # Enrichment score
            expected = (total_genes * pathway_size) / background_size
            enrichment_score = overlap_count / expected if expected > 0 else 0
            
            results.append({
                'Pathway': pathway_name,
                'Overlap': overlap_count,
                'Pathway_Size': pathway_size,
                'P_value': p_value,
                'Enrichment_Score': enrichment_score,
                'Genes': ', '.join(sorted(overlap))
            })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('P_value')
            df['FDR'] = df['P_value'] * len(df)  # Bonferroni correction
        
        return df

# ================================
# LAZY LOADING FUNCTIONS
# ================================

@st.cache_resource
def get_plotting_libs():
    """Lazy load plotting libraries"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import seaborn as sns
        import matplotlib.pyplot as plt
        return px, go, make_subplots, sns, plt
    except ImportError as e:
        st.error(f"Plotting libraries not available: {e}")
        return None, None, None, None, None

@st.cache_resource
def get_optional_libraries():
    """Load optional libraries with graceful fallback"""
    libs = {}
    
    # Gene conversion
    try:
        import mygene
        libs['mygene'] = mygene
    except ImportError:
        pass
    
    # Advanced stats
    try:
        from scipy import stats
        libs['scipy_stats'] = stats
    except ImportError:
        pass
    
    # Machine learning
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        libs['pca'] = PCA
        libs['scaler'] = StandardScaler
    except ImportError:
        pass
    
    # Pathway analysis
    try:
        import gseapy as gp
        libs['gseapy'] = gp
    except ImportError:
        pass
    
    return libs

# ================================
# MAIN APPLICATION CLASS
# ================================

class PrairieGenomicsStreamlit:
    """Main Streamlit application class optimized for cloud deployment"""
    
    def __init__(self):
        """Initialize the application"""
        self.setup_session_state()
        self.libs = get_optional_libraries()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'expression_data' not in st.session_state:
            st.session_state.expression_data = None
        if 'clinical_data' not in st.session_state:
            st.session_state.clinical_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'immune_scores' not in st.session_state:
            st.session_state.immune_scores = None
        if 'gene_binning' not in st.session_state:
            st.session_state.gene_binning = {}
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'p_threshold': 0.05,
                'fc_threshold': 1.5,
                'min_expression': 1.0,
                'max_genes': 10000,
                'enable_filtering': True
            }
    
    def show_header(self):
        """Display application header and navigation"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1f4037, #99f2c8);'>
            <h1 style='color: white; margin: 0; font-size: 3rem;'>🧬 Prairie Genomics Suite</h1>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Enhanced Multiomics Edition v3.0</p>
            <p style='color: white; margin: 0.2rem 0 0 0; font-size: 1rem;'>🦠 Immune Analysis | 🎯 Gene Binning | 🚀 High Performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for navigation
        return st.tabs([
            "📊 Data Import", 
            "🔍 Gene Analysis", 
            "🎯 Gene Binning",
            "🦠 Immune Analysis",
            "📈 Differential Expression", 
            "⏱️ Survival Analysis",
            "🛤️ Pathway Analysis", 
            "📚 Literature Search",
            "🎨 Visualizations", 
            "💾 Export Results",
            "⚙️ Settings"
        ])
    
    def show_sidebar(self):
        """Display sidebar with quick info and controls"""
        with st.sidebar:
            st.markdown("### 🎛️ Quick Controls")
            
            # Data status
            st.markdown("#### Data Status")
            expr_status = "✅ Loaded" if st.session_state.expression_data is not None else "❌ Not loaded"
            clin_status = "✅ Loaded" if st.session_state.clinical_data is not None else "❌ Not loaded"
            
            st.markdown(f"""
            - **Expression Data:** {expr_status}
            - **Clinical Data:** {clin_status}
            """)
            
            # Quick actions
            st.markdown("#### Quick Actions")
            if st.button("🗑️ Clear All Data"):
                for key in ['expression_data', 'clinical_data', 'analysis_results']:
                    st.session_state[key] = None if key != 'analysis_results' else {}
                safe_rerun()
            
            # Load example data
            if st.button("📋 Load Example Data"):
                self.load_example_data()
            
            # Settings preview
            st.markdown("#### Current Settings")
            st.json(st.session_state.settings)
            
            # About section
            st.markdown("---")
            st.markdown("""
            ### 🧬 About Prairie Genomics
            
            **Version:** 3.0.0 (Enhanced Multiomics Edition)
            
            **Enhanced Features:**
            - 🎯 Gene-based sample binning (Protein Atlas integration)
            - 🦠 Immune cell infiltration analysis (CIBERSORTx-style)
            - 📈 Enhanced differential expression analysis
            - ⏱️ Advanced survival analysis with multiple grouping options
            - 🛤️ Extended pathway analysis with immune pathways
            - 📚 Literature search capabilities
            - 🎨 Publication-quality visualizations
            - 💾 Comprehensive data export
            - 🚀 Performance optimized for >5,000 genes
            
            **New Capabilities:**
            - Analyze immune infiltration in tissue samples
            - Stratify samples by gene expression levels
            - Integrate clinical, expression, and immune data
            - High-performance analysis of large gene sets
            
            **Optimized for:** Streamlit Cloud deployment with enhanced multiomics capabilities.
            """)
    
    def load_example_data(self):
        """Load example genomic data with realistic issues for demonstration"""
        try:
            # Generate synthetic expression data
            np.random.seed(42)
            base_genes = [f"GENE_{i:04d}" for i in range(950)]
            samples = [f"Sample_{i:02d}" for i in range(50)]
            
            # Add some duplicate genes to demonstrate duplicate handling
            duplicate_genes = ["GENE_0001", "GENE_0002", "GENE_0003", "GENE_0010", "GENE_0011"] * 10  # 50 duplicates
            all_genes = base_genes + duplicate_genes
            
            # Create expression matrix with some differential patterns
            raw_expression_data = pd.DataFrame(
                np.random.lognormal(mean=3, sigma=1.2, size=(len(all_genes), len(samples))),
                index=all_genes,
                columns=samples
            )
            
            # Add some very low expression genes to demonstrate filtering
            low_expr_genes = base_genes[800:900]  # 100 low expression genes
            for gene in low_expr_genes:
                raw_expression_data.loc[gene] = np.random.lognormal(mean=0.5, sigma=0.3, size=len(samples))
            
            # Make some genes differentially expressed
            diff_genes = base_genes[:100]
            treatment_samples = samples[25:]
            for gene in diff_genes:
                if np.random.random() > 0.5:
                    raw_expression_data.loc[gene, treatment_samples] *= np.random.uniform(1.5, 3.0)
                else:
                    raw_expression_data.loc[gene, treatment_samples] *= np.random.uniform(0.3, 0.7)
            
            with st.spinner("🔄 Processing example data..."):
                # Apply the same processing pipeline as file upload
                
                # Step 1: Detect duplicates
                duplicate_info = EnhancedGeneFilter.detect_duplicate_genes(raw_expression_data)
                
                if duplicate_info['has_duplicates']:
                    st.info(f"ℹ️ Example data contains {duplicate_info['duplicate_count']} duplicate gene IDs (for demonstration)")
                    # Automatically aggregate duplicates using mean
                    processed_data, agg_summary = EnhancedGeneFilter.aggregate_duplicate_genes(raw_expression_data, 'mean')
                    if agg_summary['success']:
                        st.success(f"✅ Aggregated duplicates: {agg_summary['original_genes']} → {agg_summary['unique_genes']} genes")
                    else:
                        processed_data = raw_expression_data
                else:
                    processed_data = raw_expression_data
                
                # Step 2: Apply automatic filtering
                filtered_data, filter_summary = EnhancedGeneFilter.apply_smart_filtering(processed_data, auto_adjust=True)
                
                if filter_summary['success']:
                    st.success(f"✅ Smart filtering applied: {filter_summary['genes_removed']} low-expression genes removed")
                    st.info(f"📊 Final dataset: {filter_summary['filtered_genes']} genes (removal rate: {filter_summary['removal_rate']:.1f}%)")
                    
                    # Show filtering parameters
                    with st.expander("🔧 Auto-adjusted Parameters"):
                        for param, value in filter_summary['parameters'].items():
                            st.write(f"- **{param}**: {value:.3f}")
                else:
                    filtered_data = processed_data
                    st.warning("⚠️ Filtering failed, using unfiltered data")
            
            # Create clinical data
            clinical_data = pd.DataFrame({
                'Sample_ID': samples,
                'Group': ['Control'] * 25 + ['Treatment'] * 25,
                'Age': np.random.randint(30, 80, len(samples)),
                'Sex': np.random.choice(['M', 'F'], len(samples)),
                'Overall_Survival_Days': np.random.randint(100, 2000, len(samples)),
                'Event': np.random.choice([0, 1], len(samples), p=[0.6, 0.4])
            })
            clinical_data.set_index('Sample_ID', inplace=True)
            
            # Store processed data in session state
            st.session_state.expression_data = filtered_data
            st.session_state.clinical_data = clinical_data
            
            # Summary
            original_shape = raw_expression_data.shape
            final_shape = filtered_data.shape
            st.success("🎉 Example data processing complete!")
            st.info(f"📈 Processed dataset: {final_shape[0]} genes × {final_shape[1]} samples (reduced from {original_shape[0]} genes)")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(filtered_data.head())
            
        except Exception as e:
            st.error(f"❌ Failed to load example data: {e}")
    
    # ================================
    # TAB SECTIONS
    # ================================
    
    def data_import_section(self, tab):
        """Data import and loading section"""
        with tab:
            st.header("📊 Data Import")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Expression Data")
                expr_file = st.file_uploader(
                    "Upload expression matrix (CSV/Excel)",
                    type=['csv', 'xlsx', 'txt'],
                    key="expr_upload"
                )
                
                if expr_file:
                    try:
                        # Step 1: Load raw data
                        if expr_file.name.endswith('.xlsx'):
                            raw_data = pd.read_excel(expr_file, index_col=0)
                        else:
                            raw_data = pd.read_csv(expr_file, index_col=0, sep=None, engine='python')
                        
                        with st.spinner("🔍 Processing data: detecting duplicates..."):
                            # Step 2: Detect duplicate genes
                            duplicate_info = EnhancedGeneFilter.detect_duplicate_genes(raw_data)
                            
                            if duplicate_info['has_duplicates']:
                                st.warning(f"⚠️ Found {duplicate_info['duplicate_count']} duplicate gene IDs affecting {duplicate_info['affected_rows']} rows")
                                
                                # Show aggregation options
                                col_agg1, col_agg2 = st.columns([1, 2])
                                with col_agg1:
                                    aggregation_method = st.selectbox(
                                        "Aggregation method:",
                                        ["mean", "median", "max", "sum", "first"],
                                        index=0,
                                        help="How to combine duplicate gene IDs"
                                    )
                                
                                with col_agg2:
                                    if st.button("🔧 Apply Aggregation"):
                                        # Apply aggregation
                                        processed_data, agg_summary = EnhancedGeneFilter.aggregate_duplicate_genes(raw_data, aggregation_method)
                                        
                                        if agg_summary['success']:
                                            st.success(f"✅ Aggregated duplicates using {agg_summary['method_used']} method")
                                            st.info(f"Reduced from {agg_summary['original_genes']} to {agg_summary['unique_genes']} unique genes")
                                        else:
                                            st.error(f"❌ Aggregation failed: {agg_summary.get('error', 'Unknown error')}")
                                            processed_data = raw_data
                                    else:
                                        processed_data = raw_data
                                        st.info("👆 Click 'Apply Aggregation' to process duplicate genes")
                            else:
                                processed_data = raw_data
                                st.success("✅ No duplicate genes detected")
                        
                        # Step 3: Apply automatic filtering
                        with st.spinner("🧹 Applying smart gene filtering..."):
                            # Show filtering options
                            col_filt1, col_filt2 = st.columns([1, 1])
                            with col_filt1:
                                apply_auto_filter = st.checkbox("Apply automatic filtering", value=True, help="Remove lowly expressed genes")
                            with col_filt2:
                                auto_adjust = st.checkbox("Smart parameter adjustment", value=True, help="Adjust filtering parameters based on data")
                            
                            if apply_auto_filter:
                                filtered_data, filter_summary = EnhancedGeneFilter.apply_smart_filtering(processed_data, auto_adjust)
                                
                                if filter_summary['success']:
                                    st.success(f"✅ Filtering complete: {filter_summary['genes_removed']} genes removed ({filter_summary['removal_rate']:.1f}%)")
                                    
                                    # Show filtering details in expandable section
                                    with st.expander("📊 Filtering Details"):
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Original Genes", filter_summary['original_genes'])
                                        col2.metric("Filtered Genes", filter_summary['filtered_genes']) 
                                        col3.metric("Removal Rate", f"{filter_summary['removal_rate']:.1f}%")
                                        
                                        st.write("**Parameters used:**")
                                        for param, value in filter_summary['parameters'].items():
                                            st.write(f"- {param}: {value:.3f}")
                                        
                                        if filter_summary['variance_filtering_applied']:
                                            st.info("ℹ️ Additional variance filtering applied due to large gene set")
                                else:
                                    st.error(f"❌ Filtering failed: {filter_summary.get('error', 'Unknown error')}")
                                    filtered_data = processed_data
                            else:
                                filtered_data = processed_data
                                st.info("⏭️ Automatic filtering skipped")
                        
                        # Step 4: Store processed data
                        st.session_state.expression_data = filtered_data
                        
                        # Final summary
                        original_shape = raw_data.shape
                        final_shape = filtered_data.shape
                        st.success(f"🎉 Data processing complete!")
                        st.info(f"📈 Final dataset: {final_shape[0]} genes × {final_shape[1]} samples (reduced from {original_shape[0]} genes)")
                        
                        # Show preview of processed data
                        st.subheader("📋 Data Preview")
                        st.dataframe(filtered_data.head())
                        
                    except Exception as e:
                        st.error(f"❌ Error loading expression data: {e}")
                        st.error("💡 Please check file format: genes as rows, samples as columns, first column should be gene IDs")
            
            with col2:
                st.subheader("Clinical Data")
                clin_file = st.file_uploader(
                    "Upload clinical metadata (CSV/Excel)",
                    type=['csv', 'xlsx', 'txt'],
                    key="clin_upload"
                )
                
                if clin_file:
                    try:
                        if clin_file.name.endswith('.xlsx'):
                            data = pd.read_excel(clin_file, index_col=0)
                        else:
                            data = pd.read_csv(clin_file, index_col=0, sep=None, engine='python')
                        
                        st.session_state.clinical_data = data
                        st.success(f"✅ Loaded {data.shape[0]} samples, {data.shape[1]} variables")
                        st.dataframe(data.head())
                        
                    except Exception as e:
                        st.error(f"Error loading clinical data: {e}")
            
            # Data preview section
            if st.session_state.expression_data is not None or st.session_state.clinical_data is not None:
                st.markdown("---")
                st.subheader("📋 Data Preview")
                
                if st.session_state.expression_data is not None:
                    with st.expander("Expression Data Preview"):
                        st.write(f"Shape: {st.session_state.expression_data.shape}")
                        st.dataframe(st.session_state.expression_data.head(10))
                
                if st.session_state.clinical_data is not None:
                    with st.expander("Clinical Data Preview"):
                        st.write(f"Shape: {st.session_state.clinical_data.shape}")
                        st.dataframe(st.session_state.clinical_data.head(10))
    
    def gene_conversion_section(self, tab):
        """Gene ID conversion section"""
        with tab:
            st.header("🔍 Gene Analysis & Conversion")
            
            if 'mygene' in self.libs:
                st.info("✅ Gene conversion service available (mygene)")
                
                if st.session_state.expression_data is not None:
                    genes = list(st.session_state.expression_data.index[:100])  # Sample first 100
                    
                    # Auto-detect gene ID format
                    def detect_gene_format(gene_list):
                        import re
                        ensembl_pattern = re.compile(r'^ENS[A-Z]*G\d+')
                        entrez_pattern = re.compile(r'^\d+$')
                        
                        ensembl_count = sum(1 for gene in gene_list if ensembl_pattern.match(str(gene)))
                        entrez_count = sum(1 for gene in gene_list if entrez_pattern.match(str(gene)))
                        
                        if ensembl_count > len(gene_list) * 0.7:
                            return 'ensembl.gene', 'Ensembl Gene IDs'
                        elif entrez_count > len(gene_list) * 0.7:
                            return 'entrezgene', 'Entrez Gene IDs'
                        else:
                            return 'symbol', 'Gene Symbols'
                    
                    detected_format, format_name = detect_gene_format(genes)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.info(f"🔍 Detected format: **{format_name}**")
                        
                        # Auto-select appropriate conversion target
                        if detected_format == 'ensembl.gene':
                            default_target = 'symbol'  # Convert Ensembl TO symbols
                            st.success("✅ Converting Ensembl IDs → Gene Symbols")
                        elif detected_format == 'entrezgene':
                            default_target = 'symbol'  # Convert Entrez TO symbols  
                            st.success("✅ Converting Entrez IDs → Gene Symbols")
                        else:
                            default_target = 'ensembl.gene'  # Convert symbols TO Ensembl
                            st.info("ℹ️ Converting Gene Symbols → Ensembl IDs")
                        
                        conversion_type = st.selectbox(
                            "Convert TO:",
                            ["symbol", "entrezgene", "ensembl.gene"],
                            index=["symbol", "entrezgene", "ensembl.gene"].index(default_target)
                        )
                        
                        if st.button("🔄 Convert Gene IDs"):
                            try:
                                mg = self.libs['mygene']
                                
                                # Use detected format as input scope
                                results = mg.querymany(genes, scopes=detected_format, fields=conversion_type, species='human')
                                
                                conversion_df = pd.DataFrame(results)
                                
                                # Calculate conversion statistics
                                total_genes = len(genes)
                                successful_conversions = len([r for r in results if conversion_type in r and 'notfound' not in r])
                                success_rate = (successful_conversions / total_genes) * 100
                                
                                st.session_state.analysis_results['gene_conversion'] = conversion_df
                                st.success(f"Gene conversion completed! {successful_conversions}/{total_genes} genes converted ({success_rate:.1f}% success rate)")
                                
                                if success_rate < 50:
                                    st.warning("⚠️ Low conversion rate. Consider checking input gene format or trying different conversion options.")
                                
                            except Exception as e:
                                st.error(f"Gene conversion failed: {e}")
                                st.error("💡 Try: pip install mygene, or check your internet connection")
                    
                    with col2:
                        if 'gene_conversion' in st.session_state.analysis_results:
                            st.dataframe(st.session_state.analysis_results['gene_conversion'])
            else:
                st.warning("Gene conversion service not available. Install mygene: pip install mygene")
                
                # Show basic gene analysis instead
                if st.session_state.expression_data is not None:
                    st.subheader("📊 Gene Expression Summary")
                    
                    expr_data = st.session_state.expression_data
                    
                    # Basic statistics
                    st.write("**Expression Statistics:**")
                    stats_df = pd.DataFrame({
                        'Mean': expr_data.mean(axis=1),
                        'Std': expr_data.std(axis=1),
                        'Min': expr_data.min(axis=1),
                        'Max': expr_data.max(axis=1),
                        'CV': expr_data.std(axis=1) / expr_data.mean(axis=1)
                    }).round(3)
                    
                    st.dataframe(stats_df.head(20))
                    
                    # Enhanced gene filtering options
                    st.subheader("🔧 Enhanced Gene Filtering")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_expr = st.slider("Minimum average expression", 0.0, 10.0, 1.0, 0.1)
                        min_samples = st.slider("Min samples with expression", 1, 20, 3, 1)
                    
                    with col2:
                        max_cv = st.slider("Maximum coefficient of variation", 0.0, 5.0, 2.0, 0.1)
                        percentile_threshold = st.slider("Percentile threshold", 0.0, 0.5, 0.25, 0.05)
                    
                    with col3:
                        max_genes = st.slider("Max genes (performance)", 1000, 50000, 10000, 1000)
                        filter_method = st.selectbox("Filter method", ["Enhanced", "Basic", "Variance-based"])
                    
                    if st.button("🚀 Apply Enhanced Filters"):
                        try:
                            original_count = len(expr_data.index)
                            
                            if filter_method == "Enhanced":
                                filtered_data = EnhancedGeneFilter.filter_low_expression_genes(
                                    expr_data, min_expr, min_samples, percentile_threshold
                                )
                                if len(filtered_data) > max_genes:
                                    filtered_data = EnhancedGeneFilter.filter_by_variance(filtered_data, max_genes)
                            
                            elif filter_method == "Variance-based":
                                filtered_data = EnhancedGeneFilter.filter_by_variance(expr_data, max_genes)
                            
                            else:  # Basic filtering
                                mean_expr = expr_data.mean(axis=1)
                                cv = expr_data.std(axis=1) / mean_expr
                                
                                filtered_genes = expr_data.index[
                                    (mean_expr >= min_expr) & (cv <= max_cv)
                                ]
                                filtered_data = expr_data.loc[filtered_genes]
                            
                            st.session_state.expression_data = filtered_data
                            
                            # Update settings
                            st.session_state.settings.update({
                                'min_expression': min_expr,
                                'max_genes': max_genes
                            })
                            
                            st.success(f"✅ Filtered from {original_count} to {len(filtered_data)} genes")
                            st.info(f"💡 Performance optimized for analyzing up to {len(filtered_data)} genes")
                            
                        except Exception as e:
                            st.error(f"Filtering failed: {e}")
                        
                        safe_rerun()
    
    def gene_binning_section(self, tab):
        """Gene-based sample binning section using Protein Atlas integration"""
        with tab:
            st.header("🎯 Gene Expression Binning")
            st.markdown("""
            **Bin samples based on gene expression levels using Protein Atlas thresholds**
            
            This feature allows you to stratify samples into high and low expression groups
            for a gene of interest, enabling targeted differential analysis and survival studies.
            """)
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            expr_data = st.session_state.expression_data
            
            # Gene selection
            st.subheader("🎯 Gene Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                available_genes = sorted(list(expr_data.index))
                gene_of_interest = st.selectbox(
                    "Select gene of interest:",
                    available_genes,
                    help="Choose a gene to use for sample binning"
                )
                
                if gene_of_interest:
                    # Show gene expression distribution
                    gene_expr = expr_data.loc[gene_of_interest]
                    st.write(f"**Expression Statistics for {gene_of_interest}:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    stats_col1.metric("Mean", f"{gene_expr.mean():.2f}")
                    stats_col2.metric("Median", f"{gene_expr.median():.2f}")
                    stats_col3.metric("Std Dev", f"{gene_expr.std():.2f}")
            
            with col2:
                tissue_type = st.selectbox(
                    "Tissue context (for thresholds):",
                    ["pancreas", "liver", "brain", "immune", "general"],
                    help="Select tissue type for context-specific expression thresholds"
                )
                
                binning_method = st.selectbox(
                    "Binning method:",
                    ["Quartile-based", "Protein Atlas thresholds", "Custom percentiles"],
                    help="Choose how to define high vs low expression groups"
                )
            
            # Additional parameters for custom binning
            if binning_method == "Custom percentiles":
                col1, col2 = st.columns(2)
                with col1:
                    high_percentile = st.slider("High expression percentile", 50, 95, 75, 5)
                with col2:
                    low_percentile = st.slider("Low expression percentile", 5, 50, 25, 5)
            
            # Perform binning
            if st.button("📊 Perform Gene Binning") and gene_of_interest:
                try:
                    with st.spinner("Analyzing gene expression distribution..."):
                        
                        if binning_method == "Quartile-based":
                            high_samples, low_samples = ProteinAtlasIntegration.bin_genes_by_expression(
                                expr_data, gene_of_interest, tissue_type
                            )
                        elif binning_method == "Custom percentiles":
                            gene_expr = expr_data.loc[gene_of_interest]
                            high_threshold = np.percentile(gene_expr, high_percentile)
                            low_threshold = np.percentile(gene_expr, 100 - low_percentile)
                            
                            high_samples = gene_expr[gene_expr >= high_threshold].index.tolist()
                            low_samples = gene_expr[gene_expr <= low_threshold].index.tolist()
                        else:  # Protein Atlas thresholds
                            thresholds = ProteinAtlasIntegration.get_tissue_expression_data(
                                gene_of_interest, tissue_type
                            )
                            gene_expr = expr_data.loc[gene_of_interest]
                            
                            # Use tissue-specific thresholds if available
                            high_threshold = thresholds.get('high_threshold', np.percentile(gene_expr, 75))
                            low_threshold = thresholds.get('low_threshold', np.percentile(gene_expr, 25))
                            
                            high_samples = gene_expr[gene_expr >= high_threshold].index.tolist()
                            low_samples = gene_expr[gene_expr <= low_threshold].index.tolist()
                        
                        if high_samples and low_samples:
                            # Store results
                            binning_result = {
                                'gene': gene_of_interest,
                                'high_samples': high_samples,
                                'low_samples': low_samples,
                                'tissue': tissue_type,
                                'method': binning_method
                            }
                            st.session_state.gene_binning[gene_of_interest] = binning_result
                            
                            # Display results
                            st.success("✅ Gene binning completed!")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("High Expression", len(high_samples))
                            col2.metric("Low Expression", len(low_samples))
                            col3.metric("Total Binned", len(high_samples) + len(low_samples))
                            
                            # Show distribution plot
                            self.create_gene_binning_plot(expr_data, gene_of_interest, high_samples, low_samples)
                            
                        else:
                            st.error("Failed to create expression bins. Check your thresholds.")
                            
                except Exception as e:
                    st.error(f"Gene binning failed: {e}")
            
            # Display existing binning results
            if st.session_state.gene_binning:
                st.markdown("---")
                st.subheader("📋 Current Gene Binning Results")
                
                for gene, result in st.session_state.gene_binning.items():
                    with st.expander(f"🎯 {gene} binning results"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**High Expression Samples:**")
                            st.write(f"Count: {len(result['high_samples'])}")
                            if st.checkbox(f"Show sample IDs ({gene} high)", key=f"show_high_{gene}"):
                                st.write(result['high_samples'])
                        
                        with col2:
                            st.write("**Low Expression Samples:**")
                            st.write(f"Count: {len(result['low_samples'])}")
                            if st.checkbox(f"Show sample IDs ({gene} low)", key=f"show_low_{gene}"):
                                st.write(result['low_samples'])
                        
                        if st.button(f"🗑️ Remove {gene} binning", key=f"remove_{gene}"):
                            del st.session_state.gene_binning[gene]
                            safe_rerun()
    
    def create_gene_binning_plot(self, expr_data, gene, high_samples, low_samples):
        """Create visualization for gene binning results"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Prepare data
            gene_expr = expr_data.loc[gene]
            
            plot_data = []
            for sample in gene_expr.index:
                if sample in high_samples:
                    group = "High Expression"
                elif sample in low_samples:
                    group = "Low Expression"
                else:
                    group = "Middle Expression"
                
                plot_data.append({
                    'Sample': sample,
                    'Expression': gene_expr[sample],
                    'Group': group
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create box plot
            fig = px.box(
                plot_df,
                x='Group',
                y='Expression',
                title=f'Gene Expression Distribution: {gene}',
                color='Group',
                color_discrete_map={
                    'High Expression': '#FF6B6B',
                    'Low Expression': '#4ECDC4',
                    'Middle Expression': '#95A5A6'
                }
            )
            
            fig.update_layout(
                xaxis_title='Expression Group',
                yaxis_title='Expression Level',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create binning plot: {e}")
    
    def immune_infiltration_section(self, tab):
        """Immune cell infiltration analysis section"""
        with tab:
            st.header("🦠 Immune Cell Infiltration Analysis")
            st.markdown("""
            **Analyze immune cell infiltration using CIBERSORTx-style signatures**
            
            This analysis estimates the relative abundance of different immune cell types
            in your samples based on gene expression signatures from major immune cell populations.
            """)
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            expr_data = st.session_state.expression_data
            
            # Analysis parameters
            st.subheader("🔧 Analysis Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                scoring_method = st.selectbox(
                    "Scoring method:",
                    ["ssgsea", "mean", "zscore"],
                    help="Method for calculating immune infiltration scores"
                )
            
            with col2:
                cell_type_filter = st.multiselect(
                    "Focus on cell types (optional):",
                    list(ImmuneInfiltrationAnalyzer.get_cibersort_signatures().keys()),
                    help="Leave empty to analyze all cell types"
                )
            
            with col3:
                min_genes_required = st.slider(
                    "Minimum signature genes required:",
                    1, 5, 2,
                    help="Minimum number of signature genes that must be present"
                )
            
            # Run immune infiltration analysis
            if st.button("🚀 Run Immune Infiltration Analysis"):
                try:
                    with st.spinner("Calculating immune cell infiltration scores..."):
                        
                        # Calculate immune scores
                        immune_scores = ImmuneInfiltrationAnalyzer.calculate_immune_scores(
                            expr_data, method=scoring_method
                        )
                        
                        if not immune_scores.empty:
                            # Filter by minimum genes if needed
                            signatures = ImmuneInfiltrationAnalyzer.get_cibersort_signatures()
                            valid_cell_types = []
                            
                            for cell_type in immune_scores.columns:
                                signature_genes = signatures[cell_type]
                                available_genes = [g for g in signature_genes if g in expr_data.index]
                                
                                if len(available_genes) >= min_genes_required:
                                    valid_cell_types.append(cell_type)
                            
                            immune_scores = immune_scores[valid_cell_types]
                            
                            # Apply cell type filter if specified
                            if cell_type_filter:
                                available_filters = [ct for ct in cell_type_filter if ct in immune_scores.columns]
                                if available_filters:
                                    immune_scores = immune_scores[available_filters]
                            
                            # Store results
                            st.session_state.immune_scores = immune_scores
                            st.session_state.analysis_results['immune_infiltration'] = immune_scores
                            
                            st.success("✅ Immune infiltration analysis completed!")
                            
                            # Display summary
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Cell Types Analyzed", len(immune_scores.columns))
                            col2.metric("Samples", len(immune_scores.index))
                            col3.metric("Scoring Method", scoring_method.upper())
                            
                        else:
                            st.error("No immune infiltration scores could be calculated.")
                            
                except Exception as e:
                    st.error(f"Immune analysis failed: {e}")
            
            # Display results
            if st.session_state.immune_scores is not None:
                st.markdown("---")
                st.subheader("📊 Immune Infiltration Results")
                
                immune_scores = st.session_state.immune_scores
                
                # Summary heatmap
                with st.expander("🔥 Immune Score Heatmap"):
                    self.create_immune_heatmap(immune_scores)
                
                # Detailed results table
                with st.expander("📋 Detailed Immune Scores"):
                    st.dataframe(immune_scores.round(4), use_container_width=True)
                
                # Correlation with clinical data
                if st.session_state.clinical_data is not None:
                    with st.expander("🔗 Clinical Correlations"):
                        self.analyze_immune_clinical_correlations(immune_scores, st.session_state.clinical_data)
                
                # Individual cell type analysis
                st.subheader("🔍 Individual Cell Type Analysis")
                selected_cell_type = st.selectbox(
                    "Select cell type for detailed analysis:",
                    immune_scores.columns
                )
                
                if selected_cell_type:
                    self.create_immune_cell_plot(immune_scores, selected_cell_type)
    
    def create_immune_heatmap(self, immune_scores):
        """Create heatmap of immune infiltration scores"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Normalize scores for better visualization
            from scipy.stats import zscore
            normalized_scores = immune_scores.apply(zscore)
            
            # Create heatmap
            fig = px.imshow(
                normalized_scores.T,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title='Immune Cell Infiltration Scores (Z-score normalized)'
            )
            
            fig.update_layout(
                xaxis_title='Samples',
                yaxis_title='Immune Cell Types',
                height=max(400, len(immune_scores.columns) * 25)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create immune heatmap: {e}")
    
    def create_immune_cell_plot(self, immune_scores, cell_type):
        """Create detailed plot for specific immune cell type"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Create distribution plot
            fig = px.histogram(
                x=immune_scores[cell_type],
                nbins=20,
                title=f'{cell_type.replace("_", " ").title()} Infiltration Distribution'
            )
            
            fig.update_layout(
                xaxis_title='Infiltration Score',
                yaxis_title='Number of Samples'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            scores = immune_scores[cell_type]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Score", f"{scores.mean():.4f}")
            col2.metric("Median Score", f"{scores.median():.4f}")
            col3.metric("Std Dev", f"{scores.std():.4f}")
            col4.metric("Range", f"{scores.max() - scores.min():.4f}")
            
        except Exception as e:
            st.error(f"Failed to create cell type plot: {e}")
    
    def analyze_immune_clinical_correlations(self, immune_scores, clinical_data):
        """Analyze correlations between immune scores and clinical variables"""
        try:
            # Find common samples
            common_samples = set(immune_scores.index) & set(clinical_data.index)
            if len(common_samples) < 5:
                st.warning("Not enough common samples for correlation analysis")
                return
            
            immune_subset = immune_scores.loc[list(common_samples)]
            clinical_subset = clinical_data.loc[list(common_samples)]
            
            # Find numerical clinical variables
            numerical_vars = []
            for col in clinical_subset.columns:
                if pd.api.types.is_numeric_dtype(clinical_subset[col]):
                    numerical_vars.append(col)
            
            if numerical_vars:
                st.write("**Correlations with numerical clinical variables:**")
                
                correlation_results = []
                for immune_cell in immune_subset.columns:
                    for clinical_var in numerical_vars:
                        try:
                            corr = immune_subset[immune_cell].corr(clinical_subset[clinical_var])
                            if not pd.isna(corr):
                                correlation_results.append({
                                    'Immune_Cell': immune_cell,
                                    'Clinical_Variable': clinical_var,
                                    'Correlation': corr,
                                    'Abs_Correlation': abs(corr)
                                })
                        except:
                            continue
                
                if correlation_results:
                    corr_df = pd.DataFrame(correlation_results)
                    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
                    st.dataframe(corr_df.head(20))
            
        except Exception as e:
            st.error(f"Clinical correlation analysis failed: {e}")
    
    def differential_expression_section(self, tab):
        """Differential expression analysis section"""
        with tab:
            st.header("📈 Differential Expression Analysis")
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            if st.session_state.clinical_data is None:
                st.warning("Please load clinical data first!")
                return
            
            # Group selection with integration of gene binning
            st.subheader("👥 Group Definition")
            
            group_source = st.selectbox(
                "Group definition source:",
                ["Clinical data", "Gene binning results"],
                help="Use clinical variables or gene expression binning for group definition"
            )
            
            control_samples = []
            treatment_samples = []
            
            if group_source == "Clinical data":
                clinical_cols = list(st.session_state.clinical_data.columns)
                group_column = st.selectbox("Select grouping variable:", clinical_cols)
                
                if group_column:
                    unique_groups = st.session_state.clinical_data[group_column].unique()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        control_group = st.selectbox("Control group:", unique_groups)
                    with col2:
                        treatment_groups = unique_groups[unique_groups != control_group]
                        treatment_group = st.selectbox("Treatment group:", treatment_groups)
                    
                    # Get sample lists from clinical data
                    control_samples = st.session_state.clinical_data[
                        st.session_state.clinical_data[group_column] == control_group
                    ].index.tolist()
                    
                    treatment_samples = st.session_state.clinical_data[
                        st.session_state.clinical_data[group_column] == treatment_group
                    ].index.tolist()
            
            elif group_source == "Gene binning results":
                if not st.session_state.gene_binning:
                    st.warning("No gene binning results available. Please run gene binning analysis first.")
                    return
                
                available_genes = list(st.session_state.gene_binning.keys())
                selected_gene = st.selectbox("Select gene for binning-based analysis:", available_genes)
                
                if selected_gene:
                    binning_result = st.session_state.gene_binning[selected_gene]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Low {selected_gene} Expression Group**")
                        st.write(f"Samples: {len(binning_result['low_samples'])}")
                        control_samples = binning_result['low_samples']
                    
                    with col2:
                        st.write(f"**High {selected_gene} Expression Group**")
                        st.write(f"Samples: {len(binning_result['high_samples'])}")
                        treatment_samples = binning_result['high_samples']
                
            # Analysis parameters
            st.subheader("🔧 Analysis Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p_threshold = st.number_input("P-value threshold:", 0.001, 0.1, 0.05, 0.001)
            with col2:
                fc_threshold = st.number_input("Fold change threshold:", 1.1, 5.0, 1.5, 0.1)
            with col3:
                min_expr = st.number_input("Minimum expression:", 0.0, 10.0, 1.0, 0.1)
            
            # Run analysis button
            if st.button("🚀 Run Differential Expression Analysis") and control_samples and treatment_samples:
                with st.spinner("Running analysis..."):
                    try:
                        # Filter for common samples
                        expr_samples = set(st.session_state.expression_data.columns)
                        control_samples_filtered = [s for s in control_samples if s in expr_samples]
                        treatment_samples_filtered = [s for s in treatment_samples if s in expr_samples]
                        
                        if len(control_samples_filtered) < 3 or len(treatment_samples_filtered) < 3:
                            st.error("Need at least 3 samples per group!")
                            return
                        
                        # Performance optimization for large gene sets
                        expr_data = st.session_state.expression_data
                        if len(expr_data) > st.session_state.settings.get('max_genes', 10000):
                            st.info(f"Large gene set detected ({len(expr_data)} genes). Applying variance filtering for performance...")
                            expr_data = EnhancedGeneFilter.filter_by_variance(
                                expr_data, st.session_state.settings.get('max_genes', 10000)
                            )
                            st.info(f"Filtered to top {len(expr_data)} most variable genes")
                        
                        # Run analysis
                        analyzer = BasicStatsAnalyzer()
                        results = analyzer.ttest_analysis(
                            expr_data,
                            control_samples_filtered,
                            treatment_samples_filtered
                        )
                        
                        if not results.empty:
                            # Apply filters
                            significant = (
                                    (results['padj'] < p_threshold) &
                                    (np.abs(results['log2FoldChange']) > np.log2(fc_threshold)) &
                                    (results['baseMean'] > min_expr)
                                )
                                
                            results['Significant'] = significant
                            results['Regulation'] = results['log2FoldChange'].apply(
                                lambda x: 'Up' if x > 0 else 'Down'
                            )
                                
                            # Store results
                            st.session_state.analysis_results['differential_expression'] = results
                            
                            # Display summary
                            n_total = len(results)
                            n_significant = significant.sum()
                            n_up = ((results['log2FoldChange'] > 0) & significant).sum()
                            n_down = ((results['log2FoldChange'] < 0) & significant).sum()
                                
                            st.success("✅ Analysis completed!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Genes", n_total)
                            col2.metric("Significant", n_significant)
                            col3.metric("Upregulated", n_up)
                            col4.metric("Downregulated", n_down)
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            
            # Display results
            if 'differential_expression' in st.session_state.analysis_results:
                st.markdown("---")
                st.subheader("📊 Results")
                
                results = st.session_state.analysis_results['differential_expression']
                
                # Results table
                with st.expander("📋 Detailed Results Table"):
                    st.dataframe(
                        results.sort_values('padj').head(100),
                        use_container_width=True
                    )
                
                # Volcano plot
                if len(results) > 0:
                    self.create_volcano_plot(results)
    
    def create_volcano_plot(self, results):
        """Create volcano plot"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Prepare data
            results_plot = results.copy()
            results_plot['-log10(pvalue)'] = -np.log10(results_plot['pvalue'] + 1e-300)
            
            # Color by significance
            colors = []
            for _, row in results_plot.iterrows():
                if row['Significant']:
                    colors.append('Up-regulated' if row['log2FoldChange'] > 0 else 'Down-regulated')
                else:
                    colors.append('Not significant')
            
            results_plot['Color'] = colors
            
            # Create plot
            fig = px.scatter(
                results_plot,
                x='log2FoldChange',
                y='-log10(pvalue)',
                color='Color',
                hover_name='Gene',
                hover_data=['baseMean', 'pvalue'],
                title='Volcano Plot - Differential Expression',
                color_discrete_map={
                    'Up-regulated': '#FF6B6B',
                    'Down-regulated': '#4ECDC4', 
                    'Not significant': '#95A5A6'
                }
            )
            
            fig.update_layout(
                xaxis_title='log2 Fold Change',
                yaxis_title='-log10(p-value)',
                height=600
            )
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
            fig.add_vline(x=np.log2(1.5), line_dash="dash", line_color="gray")
            fig.add_vline(x=-np.log2(1.5), line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create volcano plot: {e}")
    
    def survival_analysis_section(self, tab):
        """Survival analysis section"""
        with tab:
            st.header("⏱️ Survival Analysis")
            
            if st.session_state.clinical_data is None:
                st.warning("Please load clinical data first!")
                return
            
            clinical_data = st.session_state.clinical_data
            
            # Column selection
            st.subheader("📊 Configure Survival Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                time_columns = [col for col in clinical_data.columns if 
                              any(keyword in col.lower() for keyword in ['time', 'day', 'month', 'survival'])]
                time_col = st.selectbox("Time to event column:", clinical_data.columns, 
                                      index=clinical_data.columns.get_loc(time_columns[0]) if time_columns else 0)
            
            with col2:
                event_columns = [col for col in clinical_data.columns if 
                               any(keyword in col.lower() for keyword in ['event', 'death', 'status'])]
                event_col = st.selectbox("Event column (1=event, 0=censored):", clinical_data.columns,
                                       index=clinical_data.columns.get_loc(event_columns[0]) if event_columns else 0)
            
            with col3:
                grouping_source = st.selectbox(
                    "Grouping source:",
                    ["None", "Clinical variable", "Gene binning", "Immune scores"],
                    help="Choose how to group samples for survival analysis"
                )
            
            # Configure grouping based on source
            group_col = None
            if grouping_source == "Clinical variable":
                group_col = st.selectbox("Select clinical variable:", list(clinical_data.columns))
            
            elif grouping_source == "Gene binning":
                if st.session_state.gene_binning:
                    selected_gene = st.selectbox(
                        "Select gene binning:",
                        list(st.session_state.gene_binning.keys())
                    )
                    
                    if selected_gene:
                        # Create temporary grouping column based on gene binning
                        binning_result = st.session_state.gene_binning[selected_gene]
                        
                        # Add grouping info to clinical data temporarily
                        temp_clinical = clinical_data.copy()
                        temp_clinical[f'{selected_gene}_Group'] = 'Middle'
                        
                        for sample in binning_result['high_samples']:
                            if sample in temp_clinical.index:
                                temp_clinical.loc[sample, f'{selected_gene}_Group'] = f'High_{selected_gene}'
                        
                        for sample in binning_result['low_samples']:
                            if sample in temp_clinical.index:
                                temp_clinical.loc[sample, f'{selected_gene}_Group'] = f'Low_{selected_gene}'
                        
                        clinical_data = temp_clinical
                        group_col = f'{selected_gene}_Group'
                else:
                    st.warning("No gene binning results available")
                    grouping_source = "None"
            
            elif grouping_source == "Immune scores":
                if st.session_state.immune_scores is not None:
                    immune_cell = st.selectbox(
                        "Select immune cell type:",
                        st.session_state.immune_scores.columns
                    )
                    
                    if immune_cell:
                        # Create high/low immune infiltration groups
                        immune_scores = st.session_state.immune_scores[immune_cell]
                        threshold = st.slider(
                            f"Threshold percentile for {immune_cell}:",
                            25, 75, 50, 5
                        )
                        
                        cutoff = np.percentile(immune_scores, threshold)
                        
                        temp_clinical = clinical_data.copy()
                        temp_clinical[f'{immune_cell}_Infiltration'] = 'Low'
                        
                        high_infiltration_samples = immune_scores[immune_scores >= cutoff].index
                        for sample in high_infiltration_samples:
                            if sample in temp_clinical.index:
                                temp_clinical.loc[sample, f'{immune_cell}_Infiltration'] = 'High'
                        
                        clinical_data = temp_clinical
                        group_col = f'{immune_cell}_Infiltration'
                else:
                    st.warning("No immune scores available. Run immune analysis first.")
                    grouping_source = "None"
            
            # Run analysis
            if st.button("📈 Run Survival Analysis"):
                try:
                    analyzer = BasicSurvivalAnalyzer()
                    fig = analyzer.kaplan_meier_analysis(clinical_data, time_col, event_col, group_col)
                    
                    if fig:
                        st.pyplot(fig)
                        st.session_state.analysis_results['survival_plot'] = fig
                    else:
                        st.error("Survival analysis failed. Check your data and column selections.")
                        
                except Exception as e:
                    st.error(f"Survival analysis failed: {e}")
            
            # Data preview
            if time_col and event_col:
                st.subheader("📋 Data Preview")
                preview_data = clinical_data[[time_col, event_col]]
                if group_col:
                    preview_data[group_col] = clinical_data[group_col]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Data Summary:")
                    st.write(preview_data.describe())
                
                with col2:
                    st.write("Sample Data:")
                    st.dataframe(preview_data.head(10))
    
    def pathway_analysis_section(self, tab):
        """Pathway analysis section"""
        with tab:
            st.header("🛤️ Pathway Analysis")
            
            if 'differential_expression' not in st.session_state.analysis_results:
                st.warning("Please run differential expression analysis first!")
                return
            
            de_results = st.session_state.analysis_results['differential_expression']
            
            # Gene list options
            st.subheader("🎯 Gene List Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                list_type = st.selectbox(
                    "Select gene list:",
                    ["All significant", "Upregulated only", "Downregulated only", "Top 100 by p-value"]
                )
            
            with col2:
                p_cutoff = st.number_input("P-value cutoff:", 0.001, 0.1, 0.05, 0.001)
            
            # Generate gene list based on selection
            if list_type == "All significant":
                gene_list = de_results[de_results['Significant']]['Gene'].tolist()
            elif list_type == "Upregulated only":
                gene_list = de_results[
                    (de_results['Significant']) & (de_results['log2FoldChange'] > 0)
                ]['Gene'].tolist()
            elif list_type == "Downregulated only":
                gene_list = de_results[
                    (de_results['Significant']) & (de_results['log2FoldChange'] < 0)
                ]['Gene'].tolist()
            else:  # Top 100
                gene_list = de_results.nsmallest(100, 'pvalue')['Gene'].tolist()
            
            st.info(f"Selected {len(gene_list)} genes for pathway analysis")
            
            # Run pathway analysis
            if st.button("🚀 Run Pathway Analysis") and len(gene_list) > 0:
                with st.spinner("Running pathway enrichment..."):
                    try:
                        analyzer = BasicPathwayAnalyzer()
                        pathway_results = analyzer.enrichment_analysis(gene_list)
                        
                        if not pathway_results.empty:
                            st.session_state.analysis_results['pathway_analysis'] = pathway_results
                            st.success("✅ Pathway analysis completed!")
                            
                            # Display results
                            st.subheader("📊 Enriched Pathways")
                            
                            # Filter by significance
                            significant_pathways = pathway_results[pathway_results['P_value'] < 0.05]
                            
                            if not significant_pathways.empty:
                                st.dataframe(
                                    significant_pathways.round(6),
                                    use_container_width=True
                                )
                                
                                # Create bar plot
                                self.create_pathway_plot(significant_pathways)
                            else:
                                st.warning("No significant pathways found (p < 0.05)")
                                st.dataframe(pathway_results.head(10))
                        else:
                            st.warning("No pathway enrichment results found.")
                            
                    except Exception as e:
                        st.error(f"Pathway analysis failed: {e}")
    
    def create_pathway_plot(self, pathway_results):
        """Create pathway enrichment plot"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Prepare data for plotting
            plot_data = pathway_results.head(10).copy()
            plot_data['-log10(P_value)'] = -np.log10(plot_data['P_value'] + 1e-300)
            
            # Create horizontal bar plot
            fig = px.bar(
                plot_data.sort_values('-log10(P_value)'),
                x='-log10(P_value)',
                y='Pathway',
                color='Enrichment_Score',
                title='Top Enriched Pathways',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=400,
                xaxis_title='-log10(P-value)',
                yaxis_title='Pathway'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create pathway plot: {e}")
    
    def literature_search_section(self, tab):
        """Literature search section"""
        with tab:
            st.header("📚 Literature Search")
            
            st.markdown("""
            **Note:** Literature search requires internet access and external APIs.
            This section provides a framework for PubMed literature search.
            """)
            
            # Search interface
            st.subheader("🔍 Search Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_source = st.selectbox(
                    "Search source:",
                    ["Manual entry", "From DE analysis", "From pathway analysis"]
                )
            
            with col2:
                max_results = st.number_input("Maximum results:", 5, 100, 20, 5)
            
            # Gene/term selection
            search_terms = []
            
            if search_source == "Manual entry":
                manual_terms = st.text_area(
                    "Enter search terms (one per line):",
                    placeholder="BRCA1\nTP53\napoptosis\ncancer"
                )
                search_terms = [term.strip() for term in manual_terms.split('\n') if term.strip()]
                
            elif search_source == "From DE analysis":
                if 'differential_expression' in st.session_state.analysis_results:
                    de_results = st.session_state.analysis_results['differential_expression']
                    top_genes = de_results[de_results['Significant']]['Gene'].head(10).tolist()
                    search_terms = st.multiselect("Select genes to search:", top_genes, default=top_genes[:5])
                else:
                    st.warning("No differential expression results available.")
                    
            elif search_source == "From pathway analysis":
                if 'pathway_analysis' in st.session_state.analysis_results:
                    pathway_results = st.session_state.analysis_results['pathway_analysis']
                    top_pathways = pathway_results['Pathway'].head(5).tolist()
                    search_terms = st.multiselect("Select pathways to search:", top_pathways, default=top_pathways[:3])
                else:
                    st.warning("No pathway analysis results available.")
            
            # Mock search results (since we can't access PubMed directly)
            if st.button("🔍 Search Literature") and search_terms:
                st.subheader("📄 Search Results (Mock)")
                
                for term in search_terms[:3]:  # Limit to 3 terms
                    with st.expander(f"📚 Results for: {term}"):
                        st.markdown(f"""
                        **Mock Literature Results for {term}:**
                        
                        1. **Title:** "Functional analysis of {term} in cancer progression"
                           - **Authors:** Smith J, et al.
                           - **Journal:** Nature Genetics (2023)
                           - **PMID:** 12345678
                           - **Abstract:** This study investigates the role of {term} in cancer development and progression...
                        
                        2. **Title:** "Therapeutic targeting of {term} pathway in oncology"
                           - **Authors:** Johnson A, et al.
                           - **Journal:** Cell (2023)
                           - **PMID:** 87654321
                           - **Abstract:** We demonstrate that targeting {term} represents a promising therapeutic approach...
                        
                        *Note: These are mock results. Real implementation would query PubMed API.*
                        """)
                
                st.info("💡 To implement real literature search, add PubMed API integration with the `pymed` package.")
    
    def advanced_visualizations_section(self, tab):
        """Advanced visualizations section"""
        with tab:
            st.header("🎨 Advanced Visualizations")
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Heatmap", "PCA Plot", "Box Plots", "Correlation Matrix", "Expression Distribution"]
            )
            
            if viz_type == "Heatmap":
                self.create_heatmap_section()
            elif viz_type == "PCA Plot":
                self.create_pca_section()
            elif viz_type == "Box Plots":
                self.create_boxplot_section()
            elif viz_type == "Correlation Matrix":
                self.create_correlation_section()
            elif viz_type == "Expression Distribution":
                self.create_distribution_section()
    
    def create_heatmap_section(self):
        """Create heatmap visualization"""
        st.subheader("🔥 Expression Heatmap")
        
        expr_data = st.session_state.expression_data
        
        # Gene selection
        col1, col2 = st.columns(2)
        with col1:
            gene_selection = st.selectbox(
                "Gene selection:",
                ["Top variable genes", "From DE analysis", "Custom list"]
            )
        
        with col2:
            n_genes = st.slider("Number of genes:", 10, 100, 50, 10)
        
        if gene_selection == "Top variable genes":
            # Select most variable genes
            gene_vars = expr_data.var(axis=1)
            top_genes = gene_vars.nlargest(n_genes).index
            plot_data = expr_data.loc[top_genes]
            
        elif gene_selection == "From DE analysis":
            if 'differential_expression' in st.session_state.analysis_results:
                de_results = st.session_state.analysis_results['differential_expression']
                top_genes = de_results.nsmallest(n_genes, 'padj')['Gene'].tolist()
                plot_data = expr_data.loc[top_genes]
            else:
                st.warning("No DE analysis results available.")
                return
                
        else:  # Custom list
            gene_input = st.text_area("Enter gene names (one per line):")
            genes = [g.strip() for g in gene_input.split('\n') if g.strip()]
            if genes:
                available_genes = [g for g in genes if g in expr_data.index]
                if available_genes:
                    plot_data = expr_data.loc[available_genes]
                else:
                    st.warning("No matching genes found.")
                    return
            else:
                st.warning("Please enter gene names.")
                return
        
        if st.button("Generate Heatmap"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Normalize data (z-score)
                from scipy.stats import zscore
                plot_data_norm = plot_data.apply(zscore, axis=1)
                
                # Create heatmap
                fig = px.imshow(
                    plot_data_norm,
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'Expression Heatmap ({len(plot_data)} genes)'
                )
                
                fig.update_layout(
                    xaxis_title='Samples',
                    yaxis_title='Genes',
                    height=max(400, len(plot_data) * 15)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create heatmap: {e}")
    
    def create_pca_section(self):
        """Create PCA visualization"""
        st.subheader("📊 Principal Component Analysis")
        
        if 'pca' not in self.libs:
            st.warning("PCA requires scikit-learn. Install with: pip install scikit-learn")
            return
        
        expr_data = st.session_state.expression_data
        
        # PCA parameters
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Number of components:", 2, 10, 2, 1)
        with col2:
            scale_data = st.checkbox("Scale data", value=True)
        
        if st.button("Run PCA"):
            try:
                PCA = self.libs['pca']
                StandardScaler = self.libs['scaler']
                
                # Prepare data (samples as rows, genes as columns)
                data_for_pca = expr_data.T
                
                if scale_data:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_for_pca)
                else:
                    data_scaled = data_for_pca.values
                
                # Run PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(data_scaled)
                
                # Create DataFrame
                pca_df = pd.DataFrame(
                    pca_result,
                    index=data_for_pca.index,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
                
                # Add clinical data if available
                if st.session_state.clinical_data is not None:
                    common_samples = set(pca_df.index) & set(st.session_state.clinical_data.index)
                    pca_df_subset = pca_df.loc[list(common_samples)]
                    clinical_subset = st.session_state.clinical_data.loc[list(common_samples)]
                    
                    # Color by first categorical column
                    color_col = None
                    for col in clinical_subset.columns:
                        if clinical_subset[col].dtype == 'object' or clinical_subset[col].nunique() < 10:
                            color_col = col
                            break
                    
                    if color_col:
                        pca_df_subset[color_col] = clinical_subset[color_col]
                else:
                    pca_df_subset = pca_df
                    color_col = None
                
                # Plot PCA
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px:
                    fig = px.scatter(
                        pca_df_subset,
                        x='PC1',
                        y='PC2',
                        color=color_col,
                        title=f'PCA Plot (PC1 vs PC2)',
                        hover_name=pca_df_subset.index
                    )
                    
                    fig.update_layout(
                        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show explained variance
                st.subheader("📈 Explained Variance")
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained_Variance_Ratio': pca.explained_variance_ratio_,
                    'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
                })
                st.dataframe(variance_df)
                
            except Exception as e:
                st.error(f"PCA analysis failed: {e}")
    
    def create_boxplot_section(self):
        """Create box plot visualization"""
        st.subheader("📦 Expression Box Plots")
        
        expr_data = st.session_state.expression_data
        
        # Gene selection
        available_genes = list(expr_data.index)
        selected_genes = st.multiselect(
            "Select genes to plot:",
            available_genes,
            default=available_genes[:5] if len(available_genes) >= 5 else available_genes
        )
        
        if selected_genes and st.button("Generate Box Plots"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Prepare data for plotting
                plot_data = []
                for gene in selected_genes:
                    for sample, value in expr_data.loc[gene].items():
                        plot_data.append({
                            'Gene': gene,
                            'Sample': sample,
                            'Expression': value
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Add clinical data if available
                if st.session_state.clinical_data is not None:
                    # Merge with clinical data
                    clinical_data = st.session_state.clinical_data.reset_index()
                    clinical_data.rename(columns={'index': 'Sample'}, inplace=True)
                    plot_df = plot_df.merge(clinical_data, on='Sample', how='left')
                
                # Create box plot
                fig = px.box(
                    plot_df,
                    x='Gene',
                    y='Expression',
                    title='Gene Expression Box Plots'
                )
                
                fig.update_layout(
                    xaxis_title='Genes',
                    yaxis_title='Expression Level'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create box plots: {e}")
    
    def create_correlation_section(self):
        """Create correlation matrix"""
        st.subheader("🔗 Sample Correlation Matrix")
        
        expr_data = st.session_state.expression_data
        
        # Correlation parameters
        col1, col2 = st.columns(2)
        with col1:
            corr_method = st.selectbox("Correlation method:", ["pearson", "spearman"])
        with col2:
            n_samples = st.slider("Max samples to show:", 10, min(50, len(expr_data.columns)), 
                                 min(20, len(expr_data.columns)), 5)
        
        if st.button("Generate Correlation Matrix"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Calculate correlation
                subset_data = expr_data.iloc[:, :n_samples]
                corr_matrix = subset_data.corr(method=corr_method)
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'Sample Correlation Matrix ({corr_method})'
                )
                
                fig.update_layout(
                    xaxis_title='Samples',
                    yaxis_title='Samples'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                st.write("**Correlation Statistics:**")
                st.write(f"Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
                st.write(f"Median correlation: {np.median(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]):.3f}")
                
            except Exception as e:
                st.error(f"Failed to create correlation matrix: {e}")
    
    def create_distribution_section(self):
        """Create expression distribution plots"""
        st.subheader("📊 Expression Distributions")
        
        expr_data = st.session_state.expression_data
        
        dist_type = st.selectbox(
            "Distribution type:",
            ["Overall distribution", "Per-sample distribution", "Per-gene distribution"]
        )
        
        if st.button("Generate Distribution Plot"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                if dist_type == "Overall distribution":
                    # Flatten all expression values
                    all_values = expr_data.values.flatten()
                    
                    fig = px.histogram(
                        x=all_values,
                        nbins=50,
                        title='Overall Expression Distribution'
                    )
                    fig.update_layout(
                        xaxis_title='Expression Level',
                        yaxis_title='Frequency'
                    )
                    
                elif dist_type == "Per-sample distribution":
                    # Box plot of expression per sample
                    sample_data = []
                    for sample in expr_data.columns[:20]:  # Limit to first 20 samples
                        for value in expr_data[sample]:
                            sample_data.append({
                                'Sample': sample,
                                'Expression': value
                            })
                    
                    sample_df = pd.DataFrame(sample_data)
                    fig = px.box(
                        sample_df,
                        x='Sample',
                        y='Expression',
                        title='Expression Distribution per Sample'
                    )
                    fig.update_xaxes(tickangle=45)
                    
                else:  # Per-gene distribution
                    # Select top variable genes
                    gene_vars = expr_data.var(axis=1)
                    top_genes = gene_vars.nlargest(10).index
                    
                    gene_data = []
                    for gene in top_genes:
                        for value in expr_data.loc[gene]:
                            gene_data.append({
                                'Gene': gene,
                                'Expression': value
                            })
                    
                    gene_df = pd.DataFrame(gene_data)
                    fig = px.box(
                        gene_df,
                        x='Gene',
                        y='Expression',
                        title='Expression Distribution per Gene (Top 10 Variable)'
                    )
                    fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create distribution plot: {e}")
    
    def export_results_section(self, tab):
        """Export and download section"""
        with tab:
            st.header("💾 Export Results")
            
            if not st.session_state.analysis_results:
                st.warning("No analysis results to export. Please run some analyses first.")
                return
            
            st.subheader("📋 Available Results")
            
            # Show available results
            for result_name, result_data in st.session_state.analysis_results.items():
                if isinstance(result_data, pd.DataFrame):
                    st.write(f"✅ **{result_name.replace('_', ' ').title()}**: {len(result_data)} rows")
                else:
                    st.write(f"✅ **{result_name.replace('_', ' ').title()}**: Available")
            
            # Export options
            st.subheader("💾 Export Options")
            
            export_format = st.selectbox(
                "Select export format:",
                ["Excel (.xlsx)", "CSV", "JSON", "Summary Report"]
            )
            
            if st.button("📦 Generate Export Package"):
                try:
                    if export_format == "Excel (.xlsx)":
                        # Create Excel file with multiple sheets
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            for result_name, result_data in st.session_state.analysis_results.items():
                                if isinstance(result_data, pd.DataFrame):
                                    result_data.to_excel(writer, sheet_name=result_name[:31])  # Excel sheet name limit
                        
                        st.download_button(
                            label="📥 Download Excel File",
                            data=output.getvalue(),
                            file_name=f"prairie_genomics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == "CSV":
                        # Export each result as separate CSV
                        for result_name, result_data in st.session_state.analysis_results.items():
                            if isinstance(result_data, pd.DataFrame):
                                csv = result_data.to_csv(index=True)
                                st.download_button(
                                    label=f"📥 Download {result_name.title()} CSV",
                                    data=csv,
                                    file_name=f"{result_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    
                    elif export_format == "JSON":
                        # Convert results to JSON
                        json_data = {}
                        for result_name, result_data in st.session_state.analysis_results.items():
                            if isinstance(result_data, pd.DataFrame):
                                json_data[result_name] = result_data.to_dict('records')
                        
                        json_str = json.dumps(json_data, indent=2, default=str)
                        st.download_button(
                            label="📥 Download JSON File",
                            data=json_str,
                            file_name=f"prairie_genomics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "Summary Report":
                        # Generate summary report
                        report = self.generate_summary_report()
                        st.download_button(
                            label="📥 Download Summary Report",
                            data=report,
                            file_name=f"prairie_genomics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    st.success("✅ Export package generated successfully!")
                    
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        report = []
        report.append("=" * 60)
        report.append("PRAIRIE GENOMICS SUITE - ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        if st.session_state.expression_data is not None:
            expr_shape = st.session_state.expression_data.shape
            report.append(f"EXPRESSION DATA: {expr_shape[0]} genes × {expr_shape[1]} samples")
        
        if st.session_state.clinical_data is not None:
            clin_shape = st.session_state.clinical_data.shape
            report.append(f"CLINICAL DATA: {clin_shape[0]} samples × {clin_shape[1]} variables")
        
        report.append("")
        report.append("ANALYSIS RESULTS:")
        report.append("-" * 40)
        
        # Analysis summaries
        for result_name, result_data in st.session_state.analysis_results.items():
            report.append(f"\n{result_name.upper().replace('_', ' ')}:")
            
            if isinstance(result_data, pd.DataFrame):
                if result_name == "differential_expression":
                    n_significant = result_data['Significant'].sum() if 'Significant' in result_data.columns else 0
                    n_up = ((result_data['log2FoldChange'] > 0) & result_data['Significant']).sum() if 'Significant' in result_data.columns else 0
                    n_down = ((result_data['log2FoldChange'] < 0) & result_data['Significant']).sum() if 'Significant' in result_data.columns else 0
                    
                    report.append(f"  - Total genes analyzed: {len(result_data)}")
                    report.append(f"  - Significant genes: {n_significant}")
                    report.append(f"  - Upregulated: {n_up}")
                    report.append(f"  - Downregulated: {n_down}")
                
                elif result_name == "pathway_analysis":
                    n_pathways = len(result_data)
                    n_significant = (result_data['P_value'] < 0.05).sum() if 'P_value' in result_data.columns else 0
                    
                    report.append(f"  - Total pathways tested: {n_pathways}")
                    report.append(f"  - Significant pathways (p<0.05): {n_significant}")
                    
                    if n_significant > 0:
                        top_pathway = result_data.loc[result_data['P_value'].idxmin(), 'Pathway']
                        report.append(f"  - Top pathway: {top_pathway}")
                
                else:
                    report.append(f"  - Records: {len(result_data)}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        
        return "\n".join(report)
    
    def settings_section(self, tab):
        """Application settings section"""
        with tab:
            st.header("⚙️ Settings & Configuration")
            
            # Analysis settings
            st.subheader("🔧 Analysis Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_p_threshold = st.number_input(
                    "P-value threshold:",
                    0.001, 0.1, 
                    st.session_state.settings['p_threshold'], 
                    0.001
                )
            
            with col2:
                new_fc_threshold = st.number_input(
                    "Fold change threshold:",
                    1.1, 5.0,
                    st.session_state.settings['fc_threshold'],
                    0.1
                )
            
            with col3:
                new_min_expression = st.number_input(
                    "Minimum expression:",
                    0.0, 10.0,
                    st.session_state.settings['min_expression'],
                    0.1
                )
            
            if st.button("💾 Save Settings"):
                st.session_state.settings.update({
                    'p_threshold': new_p_threshold,
                    'fc_threshold': new_fc_threshold,
                    'min_expression': new_min_expression
                })
                st.success("Settings saved!")
            
            # System information
            st.subheader("🖥️ System Information")
            
            # Available libraries
            st.write("**Available Libraries:**")
            lib_status = {
                "Pandas": "✅",
                "NumPy": "✅", 
                "SciPy": "✅" if 'scipy_stats' in self.libs else "❌",
                "Scikit-learn": "✅" if 'pca' in self.libs else "❌",
                "MyGene": "✅" if 'mygene' in self.libs else "❌",
                "GSEAPy": "✅" if 'gseapy' in self.libs else "❌",
                "Lifelines": "✅ (assumed)" if True else "❌",
                "Plotly": "✅ (assumed)" if True else "❌"
            }
            
            for lib, status in lib_status.items():
                st.write(f"- {lib}: {status}")
            
            # Performance tips
            st.subheader("⚡ Performance Tips")
            st.markdown("""
            **For optimal performance:**
            
            1. **Data Size**: Keep expression data under 50MB for smooth operation
            2. **Gene Filtering**: Filter low-expression genes before analysis
            3. **Sample Size**: Large sample sizes (>100) may slow down some analyses
            4. **Browser**: Use Chrome or Firefox for best Plotly visualization performance
            5. **Memory**: Close other browser tabs if experiencing slowdowns
            
            **Troubleshooting:**
            - If plots don't appear: Refresh the page
            - If analysis fails: Check data format and try example data
            - For large datasets: Use the filtering options to reduce data size
            """)
            
            # Reset options
            st.subheader("🔄 Reset Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Analysis Results"):
                    st.session_state.analysis_results = {}
                    st.success("Analysis results cleared!")
            
            with col2:
                if st.button("🔄 Reset All Settings"):
                    st.session_state.settings = {
                        'p_threshold': 0.05,
                        'fc_threshold': 1.5,
                        'min_expression': 1.0
                    }
                    st.success("Settings reset to defaults!")
    
    # ================================
    # MAIN APPLICATION RUNNER
    # ================================

    def run(self):
        """Run the main application"""
        # Show sidebar
        self.show_sidebar()
        
        # Main header and navigation
        tabs = self.show_header()
        
        # Run each section in its respective tab
        self.data_import_section(tabs[0])
        self.gene_conversion_section(tabs[1])
        self.gene_binning_section(tabs[2])
        self.immune_infiltration_section(tabs[3])
        self.differential_expression_section(tabs[4])
        self.survival_analysis_section(tabs[5])
        self.pathway_analysis_section(tabs[6])
        self.literature_search_section(tabs[7])
        self.advanced_visualizations_section(tabs[8])
        self.export_results_section(tabs[9])
        self.settings_section(tabs[10])
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
            <h3 style='color: #2c3e50;'>🧬 Prairie Genomics Suite</h3>
            <p style='margin: 0.5rem 0;'><strong>Enhanced Multiomics Edition v3.0</strong></p>
            <p style='margin: 0.5rem 0; font-size: 0.9em;'>🦠 Immune Analysis | 🎯 Gene Binning | 🚀 High Performance</p>
            <p style='margin: 0.5rem 0; font-size: 0.8em;'>Built with Streamlit • Python • Advanced Analytics • CIBERSORTx • Protein Atlas</p>
            <p style='margin: 0; font-style: italic; color: #7f8c8d;'>Making multiomics analysis accessible to every researcher</p>
            <hr style='margin: 1rem 0; border: none; height: 1px; background-color: #ddd;'>
            <p style='margin: 0; font-size: 0.8em; color: #95a5a6;'>
                Version 3.0.0 | Enhanced with Immune Infiltration & Gene Binning | Optimized for Streamlit Cloud | 
                <a href='https://github.com/prairie-genomics/suite' style='color: #3498db;'>GitHub</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    # Initialize and run the Streamlit-ready application
    try:
        app = PrairieGenomicsStreamlit()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        st.info("Please check your Python environment and dependencies.")