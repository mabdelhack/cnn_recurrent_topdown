from torch.nn import Module
from torch.nn import functional as F
from torch import zeros, mean


class CrossEntropyLossPlus(Module):

    def __init__(self, recurrent_flag=True, topdown_flag=True,
                 lambda_recognition=0.5, lambda_recurrent=0.25, lambda_topdown=0.25,
                 reduction='mean'):
        super(CrossEntropyLossPlus, self).__init__()
        self.reduction = reduction
        self.recurrent_flag = recurrent_flag
        self.topdown_flag = topdown_flag
        self.lambda_recognition = lambda_recognition
        self.lambda_recurrent = lambda_recurrent
        self.lambda_topdown = lambda_topdown

    def forward(self, recognition_input, recurrent_input, topdown_input, target):
        recognition_loss = F.cross_entropy(recognition_input, target)
        recurrent_loss = zeros(1)
        topdown_loss = zeros(1)
        if self.recurrent_flag:
            for rec_input in recurrent_input:
                if rec_input is None:
                    continue
                recurrent_loss += mean(rec_input)
        if self.topdown_flag:
            for tpdwn_input in topdown_input:
                if tpdwn_input is None:
                    continue
                topdown_loss += mean(tpdwn_input)
        loss = self.lambda_recognition * recognition_loss + \
               self.lambda_recurrent * recurrent_loss + \
               self.lambda_topdown * topdown_loss
        return loss
