"""Helper for saving pipeline results to CSV"""
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import csv


class ResultSaver:
    """Saves NORMA pipeline results to CSV files."""
    
    @staticmethod
    def save_sentence_results(
        results: List[Tuple[str, bool, Optional[List[str]]]],
        output_file: str,
        mode: str,
        ground_truth: Optional[Dict[str, List[str]]] = None,
        classification_results: Optional[List[Tuple[str, bool]]] = None,
        mapping_results: Optional[List[Tuple[str, Optional[List[str]]]]] = None
    ):
        """
        Save sentence-level results to CSV.
        
        Args:
            results: List of (sentence, is_rule, predicates)
            output_file: Output path (without extension)
            mode: Processing mode (for column selection)
            ground_truth: Optional {sentence: predicates} dict
            classification_results: Optional classification results to save
            mapping_results: Optional mapping results to save
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Main results CSV
        csv_path = output_path.with_suffix('.csv')
        
        # Determine columns
        cols = ['sentence']
        if mode == 'sentence_filtered':
            cols.append('is_rule')
        cols.append('norma_mapped')
        if ground_truth is not None:
            cols.append('gt')
        
        # Write main CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            
            for sentence, is_rule, predicates in results:
                row = {
                    'sentence': sentence,
                    'norma_mapped': ResultSaver._format_predicates(predicates)
                }
                if mode == 'sentence_filtered':
                    row['is_rule'] = is_rule
                if ground_truth is not None:
                    row['gt'] = ResultSaver._format_predicates(ground_truth.get(sentence))
                
                writer.writerow(row)
        
        print(f"\n✓ Saved: {csv_path}")
        
        # Save intermediate results if provided
        if classification_results:
            cls_path = output_path.parent / f"{output_path.stem}_classification.csv"
            ResultSaver._save_simple_csv(
                cls_path,
                ['sentence', 'is_rule'],
                [(s, r) for s, r in classification_results]
            )
        
        if mapping_results:
            map_path = output_path.parent / f"{output_path.stem}_mapping.csv"
            ResultSaver._save_simple_csv(
                map_path,
                ['sentence', 'predicates'],
                [(s, ResultSaver._format_predicates(p)) for s, p in mapping_results]
            )
    
    @staticmethod
    def save_document_results(
        all_dcs: List[List[str]],
        output_file: str
    ):
        """
        Save document-level results to CSV.
        
        Args:
            all_dcs: List of denial constraints
            output_file: Output path (without extension)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_path.with_suffix('.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['dc'])
            writer.writeheader()
            for dc in all_dcs:
                writer.writerow({'dc': ResultSaver._format_predicates(dc)})
        
        print(f"\n✓ Saved: {csv_path}")
    
    @staticmethod
    def _save_simple_csv(path: Path, cols: List[str], rows: List[Tuple]):
        """Helper to save simple CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"✓ Saved: {path}")
    
    @staticmethod
    def _format_predicates(predicates: Optional[List[str]]) -> str:
        """Format predicates list as string."""
        if not predicates:
            return ""
        return " AND ".join(predicates)