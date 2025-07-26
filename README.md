# clickbait-detection
This repository is for MSE 641 final project.

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── Task_1.py
└── Task_2.py
```

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