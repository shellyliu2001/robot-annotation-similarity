from typing import Dict, List, Optional, Tuple
import pandas as pd


class Annotation:
    def __init__(self, run_id: str, start_time: float, end_time: float, 
                 label: str, annotator_id: str):
        """
        Initialize an Annotation object.
        """
        self._run_id = run_id
        self._start_time = start_time
        self._end_time = end_time
        self._label = label
        self._annotator_id = annotator_id
    
    def get_run_id(self) -> str:
        return self._run_id
    
    def get_start_time(self) -> float:
        return self._start_time
    
    def get_end_time(self) -> float:
        return self._end_time
    
    def get_label(self) -> str:
        return self._label
    
    def get_annotator_id(self) -> str:
        return self._annotator_id

    def get_duration(self) -> float:
        return self._end_time - self._start_time

    def get_interval(self) -> Tuple[float, float]:
        return (self._start_time, self._end_time)


class Run:
    def __init__(self, run_id: str, annotations: Optional[List[Annotation]] = None):
        """
        Initialize a Run object.
        """
        self._run_id = run_id
        self._annotations = {}
        # Build dictionary from list of annotations
        if annotations is not None:
            for annotation in annotations:
                annotator_id = annotation.get_annotator_id()
                if annotator_id not in self._annotations:
                    self._annotations[annotator_id] = []
                self._annotations[annotator_id].append(annotation)
        # Extract unique annotator IDs from the annotations dictionary
        self._annotators = list(self._annotations.keys())
    
    def get_run_id(self) -> str:
        return self._run_id
    
    def get_annotators(self) -> List[str]:
        return self._annotators.copy()  # Return a copy to prevent external modification
    
    def get_all_annotations(self) -> Dict[str, List[Annotation]]:
        return self._annotations.copy()  # Return a copy to prevent external modification
    
    def add_annotation(self, annotation: Annotation):
        annotator_id = annotation.get_annotator_id()
        if annotator_id not in self._annotations:
            self._annotations[annotator_id] = []
            self._annotators.append(annotator_id)
        self._annotations[annotator_id].append(annotation)

    def get_annotations(self, annotator_id: str) -> List[Annotation]:
        return self._annotations[annotator_id]


def load_runs_from_dataframe(df: pd.DataFrame) -> List[Run]:
    """
    Load Run objects from a pandas DataFrame.
    """
    runs = []
    
    # Group by run_id
    for run_id, group in df.groupby('run_id'):
        annotations = []
        
        # Create Annotation objects for each row
        for _, row in group.iterrows():
            annotation = Annotation(
                run_id=str(row['run_id']),
                start_time=float(row['start_time']),
                end_time=float(row['end_time']),
                label=str(row['label']),
                annotator_id=str(row['annotator_id'])
            )
            annotations.append(annotation)
        
        # Create Run object with list of annotations
        run = Run(run_id=str(run_id), annotations=annotations)
        runs.append(run)
    
    return runs


def load_runs_from_csv(csv_path: str) -> List[Run]:
    """
    Load Run objects from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return load_runs_from_dataframe(df)


def main():
    """
    Test function to load runs from example_annotation.csv and display the results.
    """
    csv_path = "data/example_annotation.csv"
    runs = load_runs_from_csv(csv_path)
    
    print(f"Loaded {len(runs)} run(s) from {csv_path}\n")
    
    for run in runs:
        print(f"Run ID: {run.get_run_id()}")
        print(f"Annotators: {run.get_annotators()}")
        
        for annotator_id in run.get_annotators():
            annotations = run.get_annotations(annotator_id)
            print(f"\n  Annotator '{annotator_id}' has {len(annotations)} annotation(s):")
            for ann in annotations:
                print(f"    - [{ann.get_start_time():.1f}, {ann.get_end_time():.1f}] "
                      f"'{ann.get_label()}' (duration: {ann.get_duration():.1f}s)")
        print()


if __name__ == "__main__":
    main()

