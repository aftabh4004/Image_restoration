
from PIL import Image
import torchvision.transforms as transforms


from torch.utils.data import Dataset

class Record(object):
    def __init__(self, masked_img, unmasked_img, b1row, b1col, b2row, b2col) -> None:
        self.masked_img = masked_img
        self.unmasked_img = unmasked_img
        self.b1row = b1row
        self.b1col = b1col
        self.b2row = b2row
        self.b2col = b2col

    
class AnimalDataset(Dataset):
    def __init__(self, data_list, transform=None) -> None:
        super().__init__()
        self.data_list = data_list
        self.records = self._parse_list()
        self.transform = transform

    def _parse_list(self):
        lines = []
        records = []
        with open(self.data_list) as fp:
            lines = fp.readlines()
        
        for line in lines:
            masked_img, unmasked_img, b1row, b1col, b2row, b2col = line.split(',')
            records += [Record(masked_img, unmasked_img, b1row, b1col, b2row, b2col)]
        
        return records
    
    def __getitem__(self, index):
        record = self.records[index]

        masked_img = self._get_image(record.masked_img)
        unmasked_img = self._get_image(record.unmasked_img)
        
        return masked_img, unmasked_img
    
    def _get_image(self, path):
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        return image
    def __len__(self):
        return len(self.records)