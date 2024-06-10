# ml_course_final_project
Final project for ML course.


# Description
This project is a final project for the ML course conducted by Prof.Ying in PKU. The project is based on a [kaggle competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview) host by Optiver.

# Structure
- `data/`: The directory to store the data.
- `checkpoints/`: The directory to store the checkpoints.
- `core/`: The directory to store the feature engineering, models and losses.
- `assets/`: The directory to store the assets.
- `config/`: The configuration files.
- `main.py`: The main file to run the code.

# Usage
We recommend to run the code with python 3.10.
1. Clone the repository
2. Download the data from the [kaggle competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) and put the data into the `data/` directory. You can also download the data from PKU disk:

<blockquote> 
<a href="https://disk.pku.edu.cn/link/AA9318B275E17C49BBB7F1D52B9AFB8FE4">
Download URL
</a>

File Name：optiver-realized-volatility-prediction.zip

Due：2024-09-01 20:30 
</blockquote>

3. Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```
4. Run the following command to run the code:
```bash
python main.py
```

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
