import os
import torch
import torchvision


def torch_cat(prev, next, dim=0):
    if prev is None:
        return next
    else:
        return torch.cat((prev, next), dim=dim)


class FilePrinter(object):
    def __init__(self, file_name, only_stdout=False, file_root="result"):
        super().__init__()

        self.only_stdout = only_stdout
        if not only_stdout:
            if not os.path.isdir(file_root):
                os.makedirs(file_root)

            file_path = file_root + "/" + file_name
            assert not os.path.isfile(file_path)
            self.output_f = open(file_path, "w")

    def print(self, *strs):
        print_str = ""
        for str_i, str in enumerate(strs, 0):
            if str_i != 0:
                print_str += " "
            print_str += str

        print(print_str)
        if not self.only_stdout:
            self.output_f.write(print_str + "\n")
            self.output_f.flush()

    def close(self):
        if not self.only_stdout:
            self.output_f.close()


class TorchSaver(object):
    def __init__(self, dir_name, dir_root="model"):
        super().__init__()

        if not os.path.isdir(dir_root):
            os.makedirs(dir_root)

        dir_path = dir_root + "/" + dir_name
        assert not os.path.isdir(dir_path)
        os.makedirs(dir_path)

        self.dir_path = dir_path

    def save(self, save_dict, file_name, should_not_exist=True):
        file_path = self.dir_path + "/" + file_name
        if should_not_exist:
            assert not os.path.isfile(file_path)
        torch.save(save_dict, file_path)

    def get_saved_path(self, file_name):
        file_path = self.dir_path + "/" + file_name
        return file_path

