import os

from src.utils import Downloader
from src.config import MODEL_NAME

FILE_ID = "1bccsdNB4CsrjRlRVkFjEps_V_G4DMu_J"


if __name__ == '__main__':
    model_dir = os.path.join('src', 'model')
    if not os.path.isdir('model'):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, MODEL_NAME)

    d = Downloader()
    d.download_file_from_google_drive(FILE_ID, model_path)
    print("Model downloaded")
