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



## Examples


## Quickstart

## Jupyter Notebook

