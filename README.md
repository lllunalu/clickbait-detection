# clickbait-detection
This repository is for MSE 641 final project.

We build models for two subtasks: spoiler type classification and spoiler text generation, based on clickbait posts and linked articles.


## Repository Structure

```
├── README.md
├── requirements.txt # Python dependencies
├── clickbait_detection_final.ipynb # Main notebook (Google Colab-compatible)
├── Task_1.py # Script for spoiler type classification (Task 1)
├── Task_2.py # Script for spoiler generation (Task 2)
├── data/
+ │   ├── train.jsonl
+ │   ├── val.jsonl
+ │   └── test.jsonl
├── results/
+ │ ├── task1_output.csv
+ │ └── task2_output.csv
```
## How to Run

> **Environment**: Python 3.8+, GPU recommended  
> You can run this in Google Colab or locally (if **GPU** is available).

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Task 1 (spoiler type classification)
```bash
python Task_1.py
```
> Output: `results/task1_output.csv`


### 3. Run Task 2 (spoiler generation using T5)
```bash
python Task_2.py
```
> Output: `results/task2_output.csv`

### To Run in Google Colab

You can also run the full project notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nPJNEgjRoyX0vhZvC4L_BJ1jYhgLHroy?usp=sharing)

Recommended: Use **A100 GPU** for faster training  
Estimated runtime: **~30 minutes** for the entire project


## Task Descriptions

### Task 1: Spoiler Type Classification
**Goal**: Classify what type of spoiler is needed for each clickbait post.

**Output Types**:
- `phrase` - Short phrase spoiler
- `passage` - Longer passage spoiler  
- `multi` - Multiple non-consecutive pieces of text

**Baseline Strategy**: The naive baseline predicts `"passage"` for all clickbait posts.

**Expected Output Format**:
```csv
id,spoilerType
0,passage
1,passage
2,passage
```

### Task 2: Spoiler Generation
**Goal**: Generate text that "spoils" the clickbait by revealing what the linked article is actually about.

**Baseline Strategy**: The naive baseline uses the title of the linked webpage (`targetTitle`) as the spoiler text.

**Expected Output Format**:
```csv
id,spoiler
0,"All the scenes are actually in the movie"
1,"4.47 billion years ago"
```

## Data Format

The input data (`test.jsonl`, `train.jsonl`, `val.jsonl`) contains JSON objects with the following key fields:

- `id`: Unique identifier for each post
- `postText`: The clickbait post text
- `targetTitle`: Title of the linked webpage
- `targetParagraphs`: Main content paragraphs from the linked page
- `targetUrl`: URL of the linked webpage

**Training/Validation data also includes**:
- `tags`: Ground truth spoiler type (`"phrase"`, `"passage"`, or `"multi"`)
- `spoiler`: Human-extracted spoiler text
- `humanSpoiler`: Human-generated spoiler text