# AI Offensive Coordinator: NFL Play Prediction  

**This system will use Python and C++ to predict NFL offensive plays using data-driven modeling (in python) and high-performance data preprocessing (in C++)**  

---  

## Overview  

AI Offensive Coordinator (AIOC) is a personal project designed to explore data science and machine learning in C++ and Python through the lens of American Football.   

The goal is to predict specific offensive play types (ex: inside run, deep pass, etc.), based on situational factors such as down, distance, field position, score differential, and team tendencies.  

This project highlights:  
- Performance-focused data preprocessing in C++  
- Machine Learning modeling in Python  
- Integration via Pybind and mixed-language software design  

---  

## Project Architecture  

AI_Offensive_Coordinator/  
│  
├── cpp/                              # C++ source code  
│   ├── include/                      # Header files  
│   ├── play_features.cpp             # Feature engineering logic  
│   ├── CMakeLists.txt  
│   └── build/                        # Compiled binaries (gitignored)  
│  
├── python/                           # Python ML pipeline  
│   │  
│   ├── data/                         # Raw and cleaned datasets  
│   │  
│   ├── notebooks/                    # Research and experimentation  
│   │   ├── 01_exploration.ipynb  
│   │   └── 02_modeling.ipynb  
│   │  
│   ├── configs/                      # Configurable pipeline parameters  
│   │   ├── model_config.yaml  
│   │   └── pipeline_config.yaml  
│   │  
│   ├── ingestion/                    # Data acquisition  
│   │   └── load_data.py  
│   │  
│   ├── preprocessing/                # Data cleaning and preprocessing  
│   │   └── clean_data.py  
│   │  
│   ├── features/                     # Feature engineering logic  
│   │   └── build_features.py  
│   │  
│   ├── models/                       # Model training and inference  
│   │   ├── train_model.py  
│   │   └── predict.py  
│   │  
│   ├── evaluation/                   # Metrics and benchmarking  
│   │   ├── metrics.py  
│   │   └── benchmark.py  
│   │  
│   ├── tests/                        # Unit and integration testing  
│   │   ├── test_load_data.py  
│   │   ├── test_clean_data.py  
│   │   ├── test_build_features.py  
│   │   ├── test_train_model.py  
│   │   └── test_pipeline.py  
│   │  
│   ├── artifacts/                    # Saved models and outputs  
│   │   ├── trained_models/  
│   │   └── metrics/  
│   │  
│   ├── run_pipeline.py               # End-to-end pipeline entrypoint  
│   │  
│   └── requirements.txt  
│  
└── README.md  

---  

## Using This System

### 1. Clone the Repository    
```bash
git clone https://github.com/<your-username>/gridiron-analytics.git
cd gridiron-analytics 
```

### 2. Set Up Python Environment  
```bash
python -m venv venv
source venv\\Scripts\\activate
pip install -r requirements.txt
```  
nex
### 3. Build the C++ Module  
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```  

### 4. Test Python Integration  
```python
import play_features
data = play_features.extract_features("python/data/plays_2023.csv")
```    

## Data Source  

This project uses open football data from:  
- **NFLverse / nfl_data_py** https://github.com/nflverse/nfl_data_py  

## Machine Learning Pipeline  

The AI Offensive Coordinator system is designed as a modular end-to-end machine learning pipeline that separates data ingestion, preprocessing, feature engineering, modeling, and evaluation into distinct stages.  

This architecture mirrors real-world production ML systems by emphasizing reproducibility, modularity, and clear separation of responsibilities between Python and C++.  

### 1. Data Acquisition (Python)  
Historical NFL play-by-play data is fetched using the nflverse ecosystem (`nfl_data_py`).  

This stage is responsible for:  
- Downloading raw play-by-play datasets  
- Managing local dataset storage  
- Validating required columns and schema consistency  
- Preparing raw data for downstream preprocessing  

### 2. Data Cleaning & Preprocessing (Python → C++)  
Raw NFL data contains missing values, inconsistent fields, and sequential game-state information that must be standardized before modeling.  

Initial cleaning and validation are handled in Python, while performance-critical feature extraction and sequential processing logic are delegated to C++.  

The C++ preprocessing layer is designed to support:  
- Efficient computation of rolling historical statistics    
- Possession-aware feature generation  
- Fast transformation of large play-by-play datasets  
- Low-overhead feature extraction for future real-time inference scenarios  

Python and C++ are integrated using Pybind11.  

### 3. Feature Engineering (C++)  
The feature engineering pipeline transforms cleaned play-by-play data into structured model-ready inputs.  

Features include:  
- Down and distance  
- Field position  
- Score differential  
- Time remaining  
- Offensive team tendencies  
- Historical play-calling patterns  

The output of this stage is a structured feature matrix used for model training and inference.  

### 4. Model Training (Python)  
Python is used for model experimentation, training, and evaluation using libraries such as XGBoost and Scikit-learn.  

This stage is responsible for:  
- Training predictive play-calling models  
- Hyperparameter experimentation  
- Cross-validation  
- Model serialization and artifact generation  

Model parameters and pipeline behavior are controlled through configurable YAML files to support reproducible experimentation.  

### 5. Inference & Play Prediction (Python)  
Once trained, the system can generate offensive play predictions for a given game situation.  

The inference pipeline:  
- Accepts situational football inputs  
- Generates model-ready features  
- Loads trained model artifacts  
- Outputs predicted offensive play probabilities  

Future versions of the project may support simulated real-time game-state inference.  

### 6. Evaluation & Benchmarking (Python)  
Model performance is evaluated using both statistical metrics and football-specific contextual analysis.  

Evaluation includes:  
- Classification accuracy  
- Precision and recall  
- Confusion matrices  
- Team tendency visualization  
- Feature importance analysis  
- Benchmark comparisons between preprocessing implementations  

### 7. Pipeline Orchestration  
The complete system is executed through a unified pipeline entrypoint:  

```bash
python run_pipeline.py
```

## Author  
**Tyler Kinn**  
Undergraduate Student | University of Texas at Austin  

## License  
MIT License © 2025 Tyler Kinn  

## Acknowledgments  
- nflverse community for data access
- Pybind11 and CMake for seamless integration  
- Inspiration from the NFL Big Data Bowl  