from importlib import import_module

from torch.utils.data.dataloader import default_collate
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from .div2k import DIV2K
from . import benchmark

class ARGS:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def get_module(name):
    if name == "DIV2K":
        return DIV2K
    elif name == "Set5":
        return Set5

def get_loader(split, exp_dict, n_threads, test_only):
    dataset_dict = exp_dict["dataset"]
    args = ARGS(ext=dataset_dict["ext"],
                scale=dataset_dict["scale"],
                data_range=dataset_dict["data_range"],
                dir_data=dataset_dict["data_dir"],
                n_colors=dataset_dict["n_colors"],
                rgb_range=dataset_dict["rgb_range"],
                patch_size=dataset_dict["patch_size"],
                no_augment=dataset_dict["no_augment"],
                data_train=dataset_dict["data_train"],
                data_test=dataset_dict["data_test"],
                lr_type=dataset_dict["lr_type"],
                batch_size=exp_dict["batch_size"],
                test_every=exp_dict["test_every"],
                n_threads=n_threads,
                test_only=test_only)
    dataset_name = dataset_dict["data_%s" %split] 

    if split == "train":
        dataset = DIV2K(args, train=(split=="train"), name=dataset_name)
        loader = dataloader.DataLoader(dataset,
                              batch_size=exp_dict["batch_size"],
                              shuffle=True,
                              pin_memory=True)
        return loader
    else:

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
            module_test = benchmark.Benchmark
            testset = module_test(args, name=args.data_test, train=False)
        else:
            testset = DIV2K(args, train=False, name=dataset_name)
            # module_test = import_module('.' +  args.data_test.lower())
            # testset = getattr(module_test, args.data_test)(args, train=False)

        return dataloader.DataLoader(testset,
            batch_size=1,
            shuffle=False,
            pin_memory=False
        )





# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )