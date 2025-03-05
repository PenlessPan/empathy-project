# Empathy Analysis in Nursing Interactions

A computational approach to estimating empathy in nurse-patient interactions using Natural Language Processing.

## Project Overview

This project aims to estimate the empathy levels of nursing students during simulated patient interactions. It processes audio recordings from 37 sessions where nursing students interact with a realistic patient simulator. The project compares multiple NLP approaches to determine the most effective method for automated empathy detection in healthcare conversations.

## Methods Implemented

Five different approaches to empathy detection are compared:

1. **AlephBERT (Hebrew-specialized)**: Hebrew BERT model on original transcripts
2. **Multilingual Hebrew**: XLM-RoBERTa on Hebrew transcripts
3. **Multilingual English**: XLM-RoBERTa on translated English transcripts
4. **English Specialized**: RoBERTa-large on translated English transcripts
5. **LLM Empathy Judge**: GPT-4 to directly score empathy levels

## Results

![Average Errors](https://raw.githubusercontent.com/PenlessPan/empathy-project/main/images/average_errors.png)

The LLM-based approach (with +1 score adjustment) performed best, followed by AlephBERT. All methods showed reasonable correlation with human-rated empathy scores.

![Correlation Heatmap](https://raw.githubusercontent.com/PenlessPan/empathy-project/main/images/correlation_heatmap.png)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Whisper
- pyannote.audio
- Pandas, NumPy, Matplotlib, Seaborn
- OpenAI API (for LLM method)
- Selenium (for translation)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/PenlessPan/empathy-project.git
   cd empathy-project
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up API credentials:
   - Create a `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here # if you want to use the LLM judge
     HF_TOKEN=your_huggingface_token_here
     ```

### Usage

#### Processing New Data

1. Place your audio files in a directory structure like:
   ```
   movies/
   ├── 101/
   │   ├── 101_ceiling.mp4
   │   ├── 101_face.mp4
   │   └── 101_top.mp4
   └── ...
   ```

2. Run the processing pipeline:
   ```
   # Extract audio
   python processing_scripts/audio-extraction.py

   # Process speech
   python processing_scripts/speech-processing.py

   # Label speakers
   python processing_scripts/label-speakers.py

   # Create dialog
   python processing_scripts/dialog-creator.py

   # Translate (if needed)
   python processing_scripts/translate_selenium_google.py
   ```

#### Running Empathy Analysis

Choose one or more analysis methods:

```
# Hebrew analysis with AlephBERT
python scoring_scripts/alephbert-empathy.py

# Using LLM for scoring
python scoring_scripts/llm-empathy-judge.py

# Other methods similarly...
```

#### Analyzing Results

Run the analysis notebook to compare methods:

```
jupyter notebook data_analysis_colab.ipynb
```

can be used directly in colab.

## Directory Structure

- **final_scores/**: Results from each method
- **processing_scripts/**: Data preparation pipeline
- **scoring_scripts/**: Empathy analysis implementations
- **data_analysis_colab.py**: Results analysis and visualization

## Implementation Notes

- The pipeline is modular - you can use individual components separately
- Processing large audio files requires significant computational resources
- The LLM approach requires an OpenAI API key with GPT-4 access
- Experiments can be run on a subset of data to reduce costs/time

## License

This project is distributed under the MIT License.
