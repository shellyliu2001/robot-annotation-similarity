# Robot Policy Run Annotation Similarity Index
Improve the reliability of human-provided annotations for robot policy runs by constructing and applying a similarity index over timestamped event labels. 

## Data
The dataset consists of timestamped event annotations produced by human evaluators watching videos of robot policy runs. Each annotation corresponds to a single event observed in a single run and is supplied independently by each annotator. The data looks like the following:

| Field          | Type        | Description                                                             |
| -------------- | ----------- | ----------------------------------------------------------------------- |
| `run_id`       | string      | Unique identifier for the robot policy execution.                       |
| `annotator_id` | string      | Identifier for the annotator who created the label.                     |
| `start_time`   | float       | Start timestamp of the labeled event (in seconds).                      |
| `end_time`     | float       | End timestamp of the labeled event (must be greater than `start_time`). |
| `label`        | string      | Event label (e.g., `"pick_up_shirt"`, `"place_on_table"`).              |

## Method
### Greedy Annotation Matching
We use a simple greedy matching algorithm to match two annotations together to compute their similarity based on their timestamp overlap.

With two lists of intervals:
* Annotator A: $A_1, A_2, ... A_n$ where $A_i$ = [start_time, end_time]
* Annotator B: $B_1, B_2, ... B_m$ where $B_i$ = [start_time, end_time]

1. Sort A and B by start time
2. Use two pointers to sweep through the two interval lists
3. Match interval i with interval j if intervals overlap by more than a certain amount of times (normalized using IOU scores)

### IoU Annotation Timestamp
IOU(Intersection over Union) scores are used in the greedy matching algorithm and the similarity scores calculation. For two intervals $[start_i, end_i], [start_j, end_j]$,  the IOU score is defined as: 
$$IOU = \frac{overlap}{len_i + len_j - overlap}$$

### Sentence Embeddings
Sentence embeddings are calculated using the [Universal Sentence Encoder model](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder) from Google. Their similarities are then calculated using their cosine similarity
$$\text{cos}(u, v) = \frac{u \cdot v}{\|u\| \, \|v\|}$$

### Similarity Score
The similarity score of a single pair of annotations is simply the average between their IOU score and the cosine similarity.

For an entire run, the similarity index is the weighted average of the similarity scores by the union of the two intervals.


## Examples
The following example can be obtained from running with the input in the data folder.

### Run 1
#### Matches

| Interval       | Label (Annotator A)              | Interval       | Label (Annotator B)              | Similarity Index |
| -------------- | ---------------------------------| -------------- | ---------------------------------|------------------|
| 0 - 15         | pick up shirt                    | 2 - 10         | right arm pick up shirt          | 0.621            |
| 20 - 25        | shake shirt                      | 22 - 30        | unfold shirt                     | 0.484            |
| 30 - 35        | put shirt down                   | 30 - 36        | lay shirt flat on table          | 0.684            |
| 36 - 45        | fold shirt in half               | 37 - 45        | fold halfway                     | 0.769            |

Overall Similarity Score: 0.630

### Run 2
Annotator A
| Interval       | Label (Annotator A)              | Interval       | Label (Annotator B)              | Similarity Index |
| -------------- | ---------------------------------| -------------- | ---------------------------------|------------------|
| 0 - 15         | pick up box                      | 0 - 15         | pick up box                      | 1.000            |
| 16 - 20        | reorient box                     | 15 - 19        | turn box around                  | 0.609            |

Overall Similarity Score: 0.902


## Quickstart

Install dependencies
```bash
pip install -r requirements.txt
```
This snippet generates the example results from above:
```python
from src.data import load_runs_from_csv
from src.similarity_index import calculate_run_similarity_index
csv_path = "data/example_annotation.csv" # Replace with your own annotations dataset
runs = load_runs_from_csv(csv_path)
for run in runs:
    calculate_run_similarity_index(run)
```

