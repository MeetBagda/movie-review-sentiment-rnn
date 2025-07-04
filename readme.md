# Movie Review Sentiment Analysis with RNN

A deep learning project that performs sentiment analysis on movie reviews using Recurrent Neural Networks (RNN) built with PyTorch. The model classifies IMDB movie reviews as either positive or negative sentiment.

## 🎯 Project Overview

This project implements a sentiment analysis system that:
- Processes text data from the IMDB movie reviews dataset
- Uses RNN architecture for sequence modeling
- Achieves binary classification (positive/negative sentiment)
- Includes data preprocessing, model training, and evaluation

## 🗂️ Dataset

The project uses the **IMDB Movie Reviews Dataset** which contains:
- 50,000 movie reviews
- Binary sentiment labels (positive/negative)
- Balanced dataset with equal positive and negative reviews

## 🏗️ Model Architecture

The sentiment analysis model consists of:

1. **Embedding Layer**: Converts words to dense vector representations
2. **RNN Layer**: Processes sequential information in the text
3. **Fully Connected Layer**: Maps RNN output to sentiment classes

### Model Parameters:
- Vocabulary size: Dynamic based on dataset
- Embedding dimension: 128
- Hidden size: 128
- Output classes: 2 (positive/negative)

## 🛠️ Technologies Used

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **scikit-learn** - Data preprocessing and evaluation

## 📁 Project Structure

```
Movie Review Sentiment Analysis with RNN/
├── Movie_Review_Sentiment_Analysis.ipynb  # Main notebook with implementation
├── IMDB-Dataset.csv                       # Dataset file
├── requirements.txt                       # Python dependencies
├── readme.md                              # Project documentation
├── venv/                                  # Virtual environment
└── .gitignore                            # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Movie Review Sentiment Analysis with RNN"
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Alternatively, you can install packages individually:
```bash
pip install torch pandas numpy matplotlib scikit-learn jupyter
```

### Running the Project

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Movie_Review_Sentiment_Analysis.ipynb`

3. Run all cells to:
   - Load and preprocess the data
   - Train the RNN model
   - Evaluate model performance
   - Visualize training progress

## 📊 Model Performance

The model training includes:
- **Training/Test Split**: 80/20
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss

Performance metrics tracked:
- Training loss over epochs
- Test accuracy
- Real-time training progress

## 🔍 Key Features

- **Text Preprocessing**: Converts text to lowercase and tokenizes
- **Vocabulary Building**: Creates word-to-index mapping
- **Sequence Padding**: Handles variable-length reviews
- **GPU Support**: Automatically detects and uses CUDA if available
- **Progress Tracking**: Real-time training updates
- **Visualization**: Loss curves and training metrics

## 📈 Results

The model provides:
- Training progress visualization
- Final accuracy on test set
- Loss curves over training epochs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 🙏 Acknowledgments

- IMDB for providing the movie reviews dataset
- PyTorch team for the excellent deep learning framework
- The open-source community for various tools and libraries used

## 📧 Contact

For questions or suggestions, please open an issue in this repository.

---

**Note**: This project is designed for educational purposes and demonstrates the application of RNN for sentiment analysis tasks.
