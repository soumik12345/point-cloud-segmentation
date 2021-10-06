import unittest

from point_seg import ShapeNetCoreLoaderInMemory, ShapeNetCoreLoader


class DataLoaderTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_memory_data_loader = ShapeNetCoreLoaderInMemory(
            object_category="Airplane"
        )
        self.in_memory_data_loader.load_data()
        self.e2e_data_loader = ShapeNetCoreLoader(object_category="Airplane")

    def test_in_memory_data_loader(self):
        train_dataset, val_dataset = self.in_memory_data_loader.get_datasets()
        x, y = next(iter(train_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 4)
        x, y = next(iter(val_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 4)

    def test_e2e_data_loader(self):
        train_dataset, val_dataset = self.e2e_data_loader.get_datasets()
        x, y = next(iter(train_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 4)
        x, y = next(iter(val_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 4)
