import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target, self.next_shape, self.next_name, self.next_target_gt = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_shape = None
            self.next_name = None
            self.next_target_gt = None
            return

        with torch.cuda.stream(self.stream):
            device = torch.device("cuda:0")
            self.next_input = self.next_input.to(device)#.cuda(non_blocking=True)
            self.next_target = self.next_target.to(device)#.cuda(non_blocking=True)
            self.next_target_gt = self.next_target_gt.to(device)#.cuda(non_blocking=True)


            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_target_gt = self.next_target_gt.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        shape = self.next_shape
        name = self.next_name
        target_gt = self.next_target_gt
        self.preload()
        return input, target, shape, name, target_gt
