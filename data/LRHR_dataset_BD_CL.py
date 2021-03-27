import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR = None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])

        assert self.paths_HR, '[Error] HR paths are empty.'


    def __getitem__(self, idx):
        lr, hr_x, hr, hr_path = self._load_file(idx)
        if self.train:
            lr, hr_x, hr = self._get_patch(lr, hr_x, hr, True)
            lr_tensor, hr_tensor, hr_x_tensor = common.np2Tensor([lr, hr, hr_x], self.opt['rgb_range'])
            return {'LR': lr_tensor, 'HR': hr_tensor, 'HR_x': hr_x_tensor, 'HR_path': hr_path}
            # print(type(lr_tensor), type(hr_tensor), type(hr_x_tensor))
        else:
            lr, hr = self._get_patch(lr, hr_x, hr, False)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.opt['rgb_range'])
            # print(type(lr_tensor), type(hr_tensor))
            return {'LR': lr_tensor, 'HR': hr_tensor, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        hr_path = self.paths_HR[idx]
        lr, hr_x, hr = common.get_imgs(hr_path)
        return lr, hr_x, hr, hr_path


    def _get_patch(self, lr, hr_x, hr, is_train=True):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr_x, hr = common.get_patch_hrx(
            lr, hr_x, hr, LR_size, self.scale)
        lr, hr_x, hr = common.augment([lr, hr_x, hr])
        if not is_train:
            return lr, hr
        return lr, hr_x, hr
    
