from torch.utils.data import Dataset, DataLoader
from torch import as_tensor
#from utils.get_new_vectors import preprocess_text, preprocess_english_text
from model.classifier import Model
from sklearn import preprocessing


class TextDataset(Dataset):
    '''
    file_list:  e.g. [./data/text/0_0.txt, ...]
    Return the setence.
    '''
    def __init__(self, txt, labels) -> None:
        self.text = txt
        self.labels = labels


    def __getitem__(self, index):
        label_text = self.labels[index]
        self.le = preprocessing.LabelEncoder()
        targets = self.le.fit_transform([label_text])
        txt = self.text[index]
        sample = {'Text': txt, 'Label': targets}
        return sample # Text = text, Label = label


    def __len__(self):
        return len(self.labels)