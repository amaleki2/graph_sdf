from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch


class CustomDataParallel(DataParallel):
    FEATURE = 'x'
    def to_dict(self, outputs):
        out = outputs[0]
        if isinstance(out, Batch):
            # return [output.to_dict() for output in outputs]
            return [{k: o[k] for k in self.FEATURES} for o in out]
        else:
            return outputs

    def gather(self, outputs, output_device):
        # outputs_dict = self.to_dict(outputs)
        out = outputs[0]
        if not isinstance(out, Batch):
            super().gather(outputs, output_device)
        else:
            dx = super().gather([output['x'] for output in outputs], output_device)
            dy = super().gather([output['y'] for output in outputs], output_device)
            d = Batch().from_dict({'x': dx, 'y': dy})
            return d
