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
├── cpp/ # C++ source code  
│ ├── include/ # Header files  
│ ├── play_features.cpp  
│ ├── CMakeLists.txt  
│ └── build/ # Compiled binaries (gitignored)  
│  
├── python/ # Python data science modules  
│ ├── data/ # Cleaned or raw datasets  
│ ├── notebooks/ # Jupyter notebooks  
│ │ ├── 01_exploration.ipynb  
│ │ └── 02_modeling.ipynb  
│ ├── train_model.py  
│ ├── benchmark.py   
│ └── requirements.txt  
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
1. Data Acquisition: Fetch data via nfl_data_py.  
2. Feature Engineering (C++): Compute situational and historical features for each play.  
3. Model Training (Python):  
- Train a random forest model.  
- Predict Play Type.  
4. Evaluation: Measure predictive accuracy and visualize tendencies by team.  

## Author  
**Tyler Kinn**  
Undergraduate Student | University of Texas at Austin  

## License  
MIT License © 2025 Tyler Kinn  

## Acknowledgments  
- nflverse community for data access
- Pybind11 and CMake for seamless integration  
- Inspiration from the NFL Big Data Bowl  