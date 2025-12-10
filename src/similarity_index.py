# import necessary libraries
import tensorflow_hub as hub
from typing import List, Tuple
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import Annotation, load_runs_from_csv

# Global variable to cache the loaded model
_model = None


def get_embedding_model():
    """
    Load and return the universal sentence encoder model.
    The model is cached after first load to avoid reloading.
    """
    global _model
    if _model is None:
        _model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2?tfhub-redirect=true")
    return _model


def generate_embeddings(sentences):
    """
    Generate embeddings for a list of sentences using the universal sentence encoder.
    """
    # Load the model (cached after first call)
    embed = get_embedding_model()
    
    # Generate embeddings
    embeddings = embed(sentences)
    
    return embeddings


def iou(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
    """
    Calculate Intersection over Union (IOU) for two time intervals.
    """
    a_start, a_end = interval_a
    b_start, b_end = interval_b
    
    # Calculate intersection
    intersection_start = max(a_start, b_start)
    intersection_end = min(a_end, b_end)
    
    if intersection_start >= intersection_end:
        # No overlap
        return 0.0
    
    intersection = intersection_end - intersection_start
    
    # Calculate union
    union_start = min(a_start, b_start)
    union_end = max(a_end, b_end)
    union = union_end - union_start
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def match_annotations(annotations_a: List[Annotation], annotations_b: List[Annotation]) -> List[Tuple[Annotation, Annotation]]:
    """
    Greedy match annotations from two lists of annotations based on IOU.
    Matches overlapping intervals greedily by selecting the best IOU match.
    """
    # Sort annotations by start time
    sorted_a = sorted(annotations_a, key=lambda x: x.get_start_time())
    sorted_b = sorted(annotations_b, key=lambda x: x.get_start_time())
    
    matches = []
    used_b = set()  # Track which annotations from b have been matched
    
    # For each annotation in a, find the best match in b
    for ann_a in sorted_a:
        best_match = None
        best_iou = 0.0
        best_idx = -1
        
        interval_a = ann_a.get_interval()
        
        # Find the best overlapping match in b
        for idx, ann_b in enumerate(sorted_b):
            if idx in used_b:
                continue
            
            interval_b = ann_b.get_interval()
            
            # Check if intervals overlap
            if interval_a[1] < interval_b[0]:
                # a ends before b starts, no need to check further (sorted order)
                break
            
            if interval_b[1] < interval_a[0]:
                # b ends before a starts, skip this b
                continue
            
            # Calculate IOU for overlapping intervals
            overlap_iou = iou(interval_a, interval_b)
            if overlap_iou > best_iou:
                best_iou = overlap_iou
                best_match = ann_b
                best_idx = idx
        
        # If we found a match, add it and mark b as used
        if best_match is not None and best_iou > 0:
            matches.append((ann_a, best_match))
            used_b.add(best_idx)
    
    return matches


def main():
    """
    Test the match_annotations function using example_annotation.csv.
    """
    csv_path = "data/example_annotation.csv"
    runs = load_runs_from_csv(csv_path)
    
    print(f"Loaded {len(runs)} run(s) from {csv_path}\n")
    
    for run in runs:
        run_id = run.get_run_id()
        annotators = run.get_annotators()
        
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Annotators: {annotators}")
        print(f"{'='*60}\n")
        
        # Match annotations between all pairs of annotators
        for i, annotator_a in enumerate(annotators):
            for annotator_b in annotators[i+1:]:
                annotations_a = run.get_annotations(annotator_a)
                annotations_b = run.get_annotations(annotator_b)
                
                print(f"Matching annotations between '{annotator_a}' and '{annotator_b}':")
                print(f"  '{annotator_a}': {len(annotations_a)} annotation(s)")
                print(f"  '{annotator_b}': {len(annotations_b)} annotation(s)")
                
                matches = match_annotations(annotations_a, annotations_b)
                
                print(f"  Found {len(matches)} match(es):\n")
                
                for ann_a, ann_b in matches:
                    interval_a = ann_a.get_interval()
                    interval_b = ann_b.get_interval()
                    overlap_iou = iou(interval_a, interval_b)
                    
                    print(f"    Match (IOU: {overlap_iou:.3f}):")
                    print(f"      '{annotator_a}': [{interval_a[0]:.1f}, {interval_a[1]:.1f}] '{ann_a.get_label()}'")
                    print(f"      '{annotator_b}': [{interval_b[0]:.1f}, {interval_b[1]:.1f}] '{ann_b.get_label()}'")
                    print()
                
                if len(matches) == 0:
                    print("    No matches found.\n")
                print()


if __name__ == "__main__":
    main()