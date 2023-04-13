from typing import Union
from ..hook import HOOK, execute_period

def CkptHOOK(HOOK):
    def __init__(self, save_root: Union[None, str]=None, pretrain: Union[None, str]=None):
        self.save_root = save_root
        self.pretrain = pretrain
        if self.save_root: setattr(self, after_epoch, save_model)

    def get_pretrain_model(self):
        if self.pretrain is None: return
        if not os.path.exists(self.pretrain): 
            raise(ValueError(f"{self.pretrain} is not an existed file or a directory."))
        if os.path.isdir(self.pretrain):
            files = os.listdir(self.pretrain)
            for f in files:
              tmp = f.split('.')
              if tmp[-1] not in ['pt', 'pth']: continue
              tmp = int(tmp[0].split('_')[-1])
              if not isinstance(tmp, int): 
                  raise(ValueError(f"Please set pretrain as the path of file or name the model as *_[epoch].pt"))
              if tmp > runner.start_epoch: 
                pretrain = os.path.join(self.pretrain, f)
        elif os.path.isfile(self.pretrain): 
              pretrain = self.pretrain
        else: raise(ValueError(f"Get unkown type as pretrain. Expect path of file or directory, but get {type(self.pretrain)}"))

        print('====== Load ckpt ======')
        print(f"Loading from {pretrain}")
        checkpoint = torch.load(pretrain)
        return checkpoint

    def before_run(self, runner):
        """
        load pretrain model
        """
        checkpoint = self.get_pretrain_model()
        if runner.is_dist():
          runner.model.module.load_state_dict(checkpoint['state_dict'])
        else:
          runner.model.load_state_dict(checkpoint['state_dict'])
        runner.start_epoch = int(checkpoint['epoch']) + 1
        runner.optimizer_hook.optimizer.load_state_dict(checkpoint['optimizer'])
        runner.best_acc_top1 = float(checkpoint['best_acc_top1'])

    def _save_model(self, model_name: Union[None, str]=None):
        ckpt = {
          'epoch': epoch,
          'state_dict': runner.model.state_dict(),
          'best_acc_top1': runner.best_acc_top1,
          'optimizer' : runner.optimizer_hook.optimizer.state_dict(),
               }
        model_name = 'weight_%d.pt'%epoch if model_name is None else model_name
        save_path = os.path.join(self.save_path, model_name)
        torch.save(ckpt, save_path)

    def save_model(self, runner):
#        model_name = runner.info['current_epoch']
        self._save_model('last.pt')
        if runner.info.get('is_best', False):
            self._save_model('best.pt')
        



