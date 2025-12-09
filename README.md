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
IOU(Intersection over Union) scores are used in the greedy matching algorithm and the similarity scores calculation. For two intervals $[start_i, end_i], [start_j, end_j]$,  the IOU score is defined as: $$len_i​=end_i−start_i,lenj_j=end_j−start_j$$ $$IOU = \frac{overlap}{len_i + len_j - overlap}$$

### Sentence Embeddings

### Similarity Score


## Examples


## Quickstart

## Jupyter Notebook

