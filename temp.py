from dataset import load_dataset as ld

dataloaders, dataset_sizes = ld(bs=1)

for i, (image_tensor, label) in enumerate(dataloaders['test']):
    print(label)