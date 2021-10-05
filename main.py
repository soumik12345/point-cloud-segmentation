from point_seg import ShapeNetCoreLoaderInMemory, ShapeNetCoreLoader


data_loader = ShapeNetCoreLoaderInMemory(object_category="Airplane")

data_loader.load_data()
data_loader.visualize_data_plt(0)
data_loader.visualize_data_plt(300)

train_dataset, val_dataset = data_loader.get_datasets()
print("Train Dataset:", train_dataset.element_spec)
print("Val Dataset:", val_dataset.element_spec)

data_loader = ShapeNetCoreLoader(object_category="Airplane")
train_dataset, val_dataset = data_loader.get_datasets()
x, y = next(iter(train_dataset))
print(x.shape, y.shape)
