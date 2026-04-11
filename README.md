# CS455LegalCaseFinder

## Team
- Nicholas Lin
- Alexandra Ramey

## GitHub Handle
- nicklin877

## Project Summary
This project is a Python-based legal precedent retrieval system for landmark U.S. Supreme Court cases. It reads a spreadsheet of curated cases and returns the most relevant matches for a case name, legal topic, factual issue, or plain-language query. The project models retrieval as a search problem over structured case records and ranks candidate results with a heuristic scoring function.

## Expected Dataset Format
The dataset should be an Excel or CSV file with these columns:

- `Case`
- `Year`
- `Topic`
- `Ruling`
- `Summary`

Example row:

`Marbury v. Madison | 1803 | Separation of Powers | 4 to 0 | Established the principle of judicial review.`

## Files
- `legal_case_finder.py` - main Python program
- `requirements.txt` - Python dependencies
- `LICENSE` - project license

## Installation
Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Program

### Interactive mode
```bash
python legal_case_finder.py caselist.xlsx
```

### Using interactive mode
- keyword or keyword, keyword, etc.
- phrase
- top [number of results]

### One query from the command line
```bash
python legal_case_finder.py caselist.xlsx --query "free speech in schools"
```

### Change number of returned results
```bash
python legal_case_finder.py caselist.xlsx --query "privacy abortion" --top-k 3
```

## Evaluation
Create a benchmark spreadsheet with at least these columns:

- `Query`
- `ExpectedCase`

Then run:

```bash
python legal_case_finder.py caselist.xlsx --evaluate benchmark_queries.xlsx --top-k 3 --save-evaluation evaluation_details.csv
```

The program reports:
- Top-1 accuracy
- Top-k accuracy
- Mean (Average) Reciprocol Rank (MRR)

## Notes
- Search is case-insensitive.
- The system supports partial keyword matching and multi-word phrase matching.
- This project is a retrieval aid only and does not provide legal advice.
