# Fruits Classifier

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different fruit types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Fruits 100 Dataset](https://www.kaggle.com/datasets/marquis03/fruits-100/data). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/sports-classification-pytorch-98-accuracy). The models can be trained using two different frameworks ([PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/Fruits-Classifier.git
```

2. Create conda environment from yml file using the following script:

a) Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n speed python=3.9
```

- Activate the environment using the following command:

```python
conda activate speed
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

b) Create a virtual environment using yml file:

```python
conda env create -f environment.yml
```

Then activate the environment using the following command:
```python
conda activate speed
```

3. Data Visualization

![image](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/05a2256c-0685-4051-a83e-470f95b563ea)

4. Train the AI model using the following script:

a) PyTorch training:

```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0" --train_framework "py"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/d6ef5b40-b792-4654-ae23-f1259a01c7f7)

The training process progress:

![image](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/a5306179-bb94-4477-b380-49b67d37a7c7)

b) PyTorch Lightning training:

```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0" --train_framework "pl"
```

5. Learning curves:
   
Use [DrawLearningCurves](https://github.com/bekhzod-olimov/Fruits-Classifier/blob/a4319e9403d7f3263a08921cfbcbb0acf38d287e/main.py#L91C13-L91C13) class to plot and save learning curves.

* Train and validation loss curves:
  
![loss_learning_curves](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/553b4341-4aae-4d6f-9d04-dc71ba853b79)

* Train and validation accuracy curves:
  
![acc_learning_curves](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/d4d2817f-44f3-44ec-adf9-2ce90846e940)

7. Inference Results (Predictions):

![fruits_preds](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/36ca2f10-fe74-423a-b950-7e33ade9bf44)

8. Inference Results (GradCAM):
   
![fruits_gradcam](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/7c2d910a-8027-475e-9696-dca6962203c3)

9. Run Demo

```python
streamlit run demo.py
```

10. [Demo](http://218.38.14.21:8502/): 
![image](https://github.com/bekhzod-olimov/Fruits-Classifier/assets/50166164/bb81d826-7b18-43cf-a670-3710621c0cd9)
