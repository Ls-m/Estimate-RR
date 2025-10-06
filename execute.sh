rm -rf checkpoints/
tensorboard --logdir logs/
python src/main.py logging.experiment_name="test_v1"