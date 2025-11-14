from model.model import Model
from model.Training import Training
from preprocess.preprocess import PreProcess
from preprocess.ToTensor import ToTensorBatch
from preprocess.DataLoader import DataLoader


def main():
    model = Model()
    model.compile()

    data_dir = r"C:\Users\eitan\Downloads\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train"
    csv_path = r"C:\Users\eitan\Downloads\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train.csv"

    pre = PreProcess(input_dir=data_dir, csv_path=csv_path)
    to_tensor = ToTensorBatch()
    loader = DataLoader(data_dir=data_dir, csv_path=csv_path, batch_size=64, preprocessor=pre, to_tensor=to_tensor)

