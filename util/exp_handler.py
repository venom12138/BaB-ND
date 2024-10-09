import csv
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import wandb
import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
class ExpHandler:
    _home = Path.home()

    def __init__(self, config=None, en_wandb=False, resume=''):
        exp_name = os.getenv('exp_name', default='default_group')
        run_name = os.getenv('run_name', default='default_name')
        self._exp_id = f'{self._get_exp_id()}_{run_name}'
        self._exp_name = exp_name
        self._run_name = run_name
        self._en_wandb = en_wandb
        if resume != '' and (Path(resume) / 'config.yaml').exists():
            print('----------resuming-----------')
            self._save_dir = Path(resume)
            with open(self._save_dir / 'config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            self._exp_id = config['exp_id']
            self._exp_name = config['exp_name']
            self._run_name = config['run_name']
            if en_wandb:
                self.wandb_run = wandb.init(group=self._exp_name, name=self._run_name, save_code=True,
                                            id=config['wandb_id'], resume='allow')
        else:
            self._save_dir = os.path.join('{}/.exp/{}'.format(self._home, os.getenv('WANDB_PROJECT', default='default_project')),
                                    exp_name, self._exp_id)
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)
            if 'stage' in config.keys():
                os.makedirs(f"{self._save_dir}/stage_{config['stage']}/", exist_ok=True)
            if en_wandb:
                self.wandb_run = wandb.init(group=exp_name, name=run_name, settings=wandb.Settings(start_method="fork"), save_code=True)
                val_step = wandb.define_metric('val_step')
                info_step = wandb.define_metric('info_step')
                wandb.define_metric(name='eval/valid_mse_final_step', step_metric=val_step)
                wandb.define_metric(name='eval/valid_l2_final_step', step_metric=val_step)
                wandb.define_metric(name='eval/valid_reg', step_metric=val_step)
                wandb.define_metric(name='eval/valid_mse', step_metric=val_step)
                wandb.define_metric(name='info/hard_negative_rate', step_metric=info_step)
                wandb.define_metric(name='info/median_error', step_metric=info_step)
                wandb.define_metric(name='info/min_error', step_metric=info_step)
                wandb.define_metric(name='info/threshold', step_metric=info_step)
                wandb.define_metric(name='info/mean_error', step_metric=info_step)
                wandb.define_metric(name='info/max_error', step_metric=info_step)
            self.val_step = 0
            self.info_step = 0
            self.save_config(config)
        # sym_dest = self._get_sym_path('N')
        # os.symlink(self._save_dir, sym_dest)

        self._logger = self._init_logger()
        
            
    def log_image(self, images):
        image = wandb.Image(images, )
        wandb.log({"images": image})
        
    @staticmethod
    def resume_sanity(new_conf, old_conf):
        print('-' * 10, 'Resume sanity check', '-' * 10)
        old_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in old_conf.items()
                                if k not in new_conf['resume_check_exclude_keys']}
        new_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in new_conf.items()
                                if k not in new_conf['resume_check_exclude_keys']}
        print(f'Diff config: {set(old_config_hashable.items()) ^ set(new_config_hashable.items())}')
        assert old_config_hashable == new_config_hashable, 'Resume sanity check failed'

    def _get_sym_path(self, state):
        sym_dir = f'{self._home}/.exp/syms'
        if not os.path.exists(sym_dir):
            os.makedirs(sym_dir)

        sym_dest = os.path.join(sym_dir, '--'.join([self._exp_id, state, self._exp_name]))
        return sym_dest

    @property
    def save_dir(self):
        return self._save_dir

    @staticmethod
    def _get_exp_id():
        with open(f'{ExpHandler._home}/.core/counter', 'r+') as f:
            counter = eval(f.read())
            f.seek(0)
            f.write(str(counter + 1))
        with open(f'{ExpHandler._home}/.core/identifier', 'r+') as f:
            identifier = f.read()[0]
        exp_id = '{}{:04d}'.format(identifier, counter)
        return exp_id

    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self._save_dir, f'{self._exp_id}_log.txt'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def save_config(self, conf):
        # conf = vars(args)
        conf = OmegaConf.to_container(conf, resolve=True)
        conf['exp_id'] = self._exp_id
        conf['exp_name'] = self._exp_name
        conf['run_name'] = self._run_name
        conf['commit'] = os.getenv('commit', default='not_set')
        conf['run_id'] = self._exp_id.split('_')[0]
        if hasattr(self, 'wandb_run'):
            conf['wandb_id'] = self.wandb_run.id
        with open(f'{self._save_dir}/config.yaml', 'w') as f:
            yaml.dump(conf, f)

        if self._en_wandb:
            wandb.config.update(conf,allow_val_change=True)

    def write(self, eval_metrics=None, train_metrics=None, info=None, **kwargs):
        rowd = OrderedDict([(f'all/{k}', v) for k, v in kwargs.items() ])
        if eval_metrics:
            rowd.update([(f'eval/' + k, v) for k, v in eval_metrics.items()])
            rowd.update([('val_step', self.val_step)])
            self.val_step += 1
        if train_metrics:
            rowd.update([(f'train/' + k, v) for k, v in train_metrics.items()])
        if info:
            rowd.update([(f'info/' + k, v) for k, v in info.items()])
            rowd.update([('info_step', self.info_step)])
            self.info_step += 1
        # path = os.path.join(self._save_dir, f'{self._exp_id}_summary.csv')
        # initial = not os.path.exists(path)
        # with open(path, mode='a') as cf:
        #     dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        #     if initial:
        #         dw.writeheader()
        #     dw.writerow(rowd)

        if self._en_wandb:
            wandb.log(rowd)
    

    def log(self, msg):
        self._logger.info(msg)

    def finish(self):
        if self._en_wandb:
            wandb.finish()
        Path(f'{self._save_dir}/finished').touch()
        # os.rename(self._get_sym_path('N'), self._get_sym_path('Y'))
        
    # def write(self, prefix, eval_metrics=None, train_metrics=None, **kwargs):
    #     rowd = OrderedDict([(f'{prefix}/{k}', v) for k, v in kwargs.items() ])
    #     if eval_metrics:
    #         rowd.update([(f'{prefix}/eval_' + k, v) for k, v in eval_metrics.items()])
    #     if train_metrics:
    #         rowd.update([(f'{prefix}/train_' + k, v) for k, v in train_metrics.items()])

    #     path = os.path.join(self._save_dir, f'{self._exp_id}_{prefix}_summary.csv')
    #     initial = not os.path.exists(path)
    #     with open(path, mode='a') as cf:
    #         dw = csv.DictWriter(cf, fieldnames=rowd.keys())
    #         if initial:
    #             dw.writeheader()
    #         dw.writerow(rowd)

    #     if self._en_wandb:
    #         wandb.log(rowd)

# selected_pics_info: {video_key: {gt_path:[gt, gt,...], rgb_path:[rgb, rgb, ...], pred_path: [pred1, pred2, ...]}}
def pair_pics_together(selected_pics_info):
    h, w = [456, 256]
    
    cate_counts = len(selected_pics_info[list(selected_pics_info.keys())[0]].keys()) # mask RGB pred ...图片的类数，有多少行
    rows_counts = len(selected_pics_info[list(selected_pics_info.keys())[0]]['pred_path']) # 有多少列，沿着行方向count

    font = cv2.FONT_HERSHEY_SIMPLEX
    output_imgs = []
    for key, value in selected_pics_info.items():
        output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
        col_cnt = 0 # 沿着列方向count
        skip_this_video = False
        # 选择一个video
        # video内部依次RGB、mask、pred, 也就是v会依次是[RGB],[gt], [pred]
        for k, v in value.items():
            # Default as key value itself
            caption = key + k.split('_')[0]

            # Handles new line character
            dy = 40
            for i, line in enumerate(caption.split('\n')):
                cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                        font, 0.8, (255,255,255), 2, cv2.LINE_AA)
            for row_cnt, img_path in enumerate(v):
                try:
                    if 'rgb' in k:
                        img = np.array(Image.open(img_path))
                    else:
                        img = np.array(Image.open(img_path))
                        img = (img * 255).astype('uint8')
                except:
                    print(f'img not found:{img_path}')
                    skip_this_video = True
                    break
                im_shape = img.shape
                if len(im_shape) == 2:
                    img = img[..., np.newaxis]

                output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                            (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            col_cnt += 1
        if not skip_this_video:
            # plt.imshow(output_image)
            # plt.show()
            # dd
            output_imgs.append(output_image)
    return output_imgs

def consume_prefix_in_state_dict_if_present(
        state_dict, prefix
):
    r"""Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)