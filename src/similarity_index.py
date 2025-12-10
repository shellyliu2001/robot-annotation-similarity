# import necessary libraries
import tensorflow_hub as hub
from typing import List, Tuple
import sys
from pathlib import Path
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import Annotation, Run, load_runs_from_csv

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


def generate_embeddings(label_a: str, label_b: str):
    """
    Generate embeddings for a list of sentences using the universal sentence encoder.
    """
    # Load the model (cached after first call)
    embed = get_embedding_model()
    
    # Generate embeddings
    embeddings = embed([label_a, label_b])
    
    return embeddings[0], embeddings[1]


def get_embedding_similarity(label_a: str, label_b: str) -> float:
    """
    Calculate the cosine similarity between two sentences using their embeddings.
    
    Args:
        label_a: First sentence/label
        label_b: Second sentence/label
    
    Returns:
        Cosine similarity score between -1 and 1 (typically between 0 and 1 for normalized embeddings).
    """
    # Get embeddings for both labels
    embedding_a, embedding_b = generate_embeddings(label_a, label_b)
    
    # Convert to numpy arrays if they're tensorflow tensors
    if hasattr(embedding_a, 'numpy'):
        embedding_a = embedding_a.numpy()
    if hasattr(embedding_b, 'numpy'):
        embedding_b = embedding_b.numpy()
    
    # Calculate cosine similarity: dot product / (norm_a * norm_b)
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    cosine_similarity = dot_product / (norm_a * norm_b)
    return float(cosine_similarity)


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


def calculate_similarity_index(annotation_a: Annotation, annotation_b: Annotation) -> float:
    """
    Calculate similarity index for two annotations by averaging IOU and cosine similarity.
    """
    # Calculate IOU based on time intervals
    interval_a = annotation_a.get_interval()
    interval_b = annotation_b.get_interval()
    iou_score = iou(interval_a, interval_b)
    
    # Calculate embedding similarity based on labels
    embedding_sim = get_embedding_similarity(annotation_a.get_label(), annotation_b.get_label())
    
    # Return average of IOU and embedding similarity
    return (iou_score + embedding_sim) / 2.0


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


def calculate_run_similarity_index(run: Run) -> float:
    """
    Calculate similarity index for a run by matching annotations between all annotator pairs,
    computing similarity indices for each match, and taking a weighted average weighted by IOU.
    
    Args:
        run: Run object containing annotations from multiple annotators
    
    Returns:
        Weighted average similarity index for the run, between 0 and 1.
        Returns 0.0 if there are no matches or fewer than 2 annotators.
    """
    annotators = run.get_annotators()
    run_id = run.get_run_id()
    
    print(f"\nCalculating similarity index for Run {run_id}")
    print(f"Annotators: {annotators}")
    
    # Need at least 2 annotators to calculate similarity
    if len(annotators) < 2:
        print(f"  Not enough annotators (need at least 2, found {len(annotators)})")
        return 0.0
    
    all_similarity_indices = []
    all_ious = []
    
    # Match annotations between all pairs of annotators
    for i, annotator_a in enumerate(annotators):
        for annotator_b in annotators[i+1:]:
            annotations_a = run.get_annotations(annotator_a)
            annotations_b = run.get_annotations(annotator_b)
            
            print(f"\n  Matching '{annotator_a}' vs '{annotator_b}':")
            
            # Get matches between this pair of annotators
            matches = match_annotations(annotations_a, annotations_b)
            
            print(f"    Found {len(matches)} match(es)")
            
            # For each match, calculate similarity index and IOU
            for ann_a, ann_b in matches:
                # Calculate similarity index
                sim_index = calculate_similarity_index(ann_a, ann_b)
                
                # Calculate IOU for weighting
                interval_a = ann_a.get_interval()
                interval_b = ann_b.get_interval()
                iou_score = iou(interval_a, interval_b)
                
                print(f"      Match: Sim Index={sim_index:.3f}, IOU={iou_score:.3f}, "
                      f"Labels: '{ann_a.get_label()}' <-> '{ann_b.get_label()}'")
                
                all_similarity_indices.append(sim_index)
                all_ious.append(iou_score)
    
    # If no matches found, return 0.0
    if len(all_similarity_indices) == 0:
        print(f"  No matches found for Run {run_id}")
        return 0.0
    
    # Calculate weighted average: sum(similarity_index * iou) / sum(iou)
    total_weighted_sum = sum(sim * iou_val for sim, iou_val in zip(all_similarity_indices, all_ious))
    total_iou_sum = sum(all_ious)
    
    if total_iou_sum == 0:
        print(f"  Total IOU sum is 0 for Run {run_id}")
        return 0.0
    
    weighted_avg = total_weighted_sum / total_iou_sum
    print(f"\n  Run {run_id} Similarity Index: {weighted_avg:.3f} "
          f"(weighted average of {len(all_similarity_indices)} matches)")
    
    return weighted_avg


def main():
    """
    Calculate similarity index for each run using example_annotation.csv.
    """
    csv_path = "data/example_annotation.csv"
    runs = load_runs_from_csv(csv_path)
    
    print(f"Loaded {len(runs)} run(s) from {csv_path}\n")
    
    for run in runs:
        calculate_run_similarity_index(run)


if __name__ == "__main__":
    main()