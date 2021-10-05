from point_seg import ShapeNetCoreLoader


data_loader = ShapeNetCoreLoader(object_category='Airplane')

data_loader.load_data()
data_loader.visualize_data_plt(0)
data_loader.visualize_data_plt(300)

data_loader.sample_points()
data_loader.visualize_data_plt(0)
data_loader.visualize_data_plt(300)

data_loader.get_datasets()
