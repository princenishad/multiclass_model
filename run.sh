export TRAINING_DATA=input/train_fold.csv

#export FOLD=0

#python -m src.train
python -m src.predict
#python -m src.post_processing.py