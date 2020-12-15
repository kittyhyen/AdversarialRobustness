# https://github.com/yaircarmon/semisup-adv/blob/master/robust_self_training.py가 argparse 깔끔함

# %%
import os
import numpy as np
import argparse
import torch
import torchvision
import torch.nn.functional as F
from lib.iccc_util import wide_attention_pre_act_resnet32
from lib.attack import MyPGDAttack

from lib.paper_exp import MultiRandomStartPGDEval
from lib.cvpr_util import return_loader_adt
from lib.vulnerable_util import TorchSaver, FilePrinter

from lib.loss import SmoothingLoss

# %%
parser = argparse.ArgumentParser('Attention Pairing ADT with Various Aug')
parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])

# training setup
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--epoch', type=int, default=110)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--lr_weight_decay', type=float, default=2e-4)
parser.add_argument('--lr_momentum', type=float, default=0.9)
parser.add_argument('--lr_decay_epoch', type=str, default="100")
parser.add_argument('--lr_decay_gamma', type=float, default=1e-1)

parser.add_argument('--valid_ratio', type=float, default=0.1, help='validation split ratio(for valid)')
parser.add_argument('--random_seed', type=int, default=0, help='random seed for validation split')

# model setup
parser.add_argument('--model_width', type=int, default=4, help='model width for wide resnet')
parser.add_argument('--input_std', type=bool, default=True, help='whether use input std in model')

parser.add_argument('--label_smoothing', type=float, default=1, help='target p for label')

# pgd setup
parser.add_argument('--pgd_epsilon', type=int, default=8)
parser.add_argument('--pgd_num_steps', type=int, default=7)
parser.add_argument('--pgd_step_size', type=int, default=2)

parser.add_argument('--attention_pairing_lambda', type=float, default=1e-1)

parser.add_argument('--augment', type=str, default='rchf', choices=['rchf', 'hf', 'rschf'])

# logging setup
parser.add_argument('--save', action='store_true', help='save model state & eval log')
parser.add_argument('--save_model_term', type=int, default=110)
parser.add_argument('--save_num_best', type=int, default=3, help='number of best model to be saved')

# eval setup
parser.add_argument('--eval_num_restart', type=int, default=4, help='number of restart in pgd in test eval')

args = parser.parse_args()

# %%

gpu_name = "cuda:%d" % args.gpu
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")

# %%

tmp = "pre_act_resnet32_%dx_epochs_%d_batch_%d_label_smoothing_%g_valid_%g_%g_save_%d_%d" % (
        args.model_width,
        args.epoch,
        args.batch_size,
        args.label_smoothing,
        args.valid_ratio,
        args.random_seed,
        args.save_num_best,
        args.eval_num_restart)

root_dir_tail = [
    args.dataset,
    "pgd_%d_num_steps_%d_step_size_%d" % (args.pgd_epsilon, args.pgd_num_steps, args.pgd_step_size),
    tmp + "_input_std" if args.input_std else tmp
]

save_name = "attention_pairing_%g_aug_%s" % (
    args.attention_pairing_lambda,
    args.augment
)

# %%

logger = FilePrinter(file_name=save_name, file_root=os.path.join(*(['result'] + root_dir_tail)),
                     only_stdout=not args.save)

if args.save:
    torch_saver = TorchSaver(dir_name=save_name, dir_root=os.path.join(*(['model'] + root_dir_tail)))

# %%

if args.dataset in ['cifar10']:
    img_size = 32
    num_classes = 10
elif args.dataset in ['cifar100']:
    img_size = 32
    num_classes = 100
elif args.dataset in ['tiny']:
    img_size = 64
    num_classes = 200
elif args.dataset in ['stl10']:
    img_size = 96
    num_classes = 10
else:
    raise NotImplementedError

# %%

if args.augment == 'rchf':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(img_size, padding=int(0.125 * img_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
elif args.augment == 'rschf':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
elif args.augment == 'hf':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

# %%

(num_train, train), (num_valid, valid), (num_test, test) = return_loader_adt(dataset=args.dataset,
                                                                             batch_size=args.batch_size,
                                                                             test_batch_size=args.batch_size//args.eval_num_restart,
                                                                             train_transform=train_transform,
                                                                             valid_ratio=args.valid_ratio,
                                                                             random_seed=args.random_seed)

logger.print("Train: %d     Valid: %d       Test: %d" % (num_train, num_valid, num_test))

# %%

def main():
    model = wide_attention_pre_act_resnet32(width=args.model_width, input_standardize=args.input_std, num_classes=num_classes)
    model = model.to(device)
    ce_criterion = torch.nn.CrossEntropyLoss()
    pairing_criterion = torch.nn.MSELoss()
    smoothing_criterion = SmoothingLoss(p_for_true=args.label_smoothing, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.lr_momentum,
                                weight_decay=args.lr_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=list(
                                                            map(lambda i: int(i), args.lr_decay_epoch.split(','))),
                                                        gamma=args.lr_decay_gamma)

    pgd = MyPGDAttack(model, ce_criterion,
                      epsilon=args.pgd_epsilon / 255,
                      num_steps=args.pgd_num_steps,
                      step_size=args.pgd_step_size / 255)
    multi_rs_pgd_eval = MultiRandomStartPGDEval(model,
                                                ce_criterion,
                                                epsilon=args.pgd_epsilon / 255,
                                                num_steps=args.pgd_num_steps,
                                                step_size=args.pgd_step_size / 255)

    best_eval_dict = {
        'ValidAcc': [0. for i in range(args.save_num_best)],
        'WhiteValidAcc': [0. for i in range(args.save_num_best)]
    }

    ###################################################################################################################
    for epoch in range(args.epoch):

        eval_dict = {}
        ### 1. train ##############################################################
        model.train()

        total_iter = 0
        total_ce_loss = 0.
        total_attention_pairing_loss = 0.

        correct = 0
        white_correct = 0
        for data in train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            white_inputs = pgd.generate(inputs, y=labels)

            conv = model.forward_img_to_conv(inputs)
            outputs = model.forward_conv_to_logit(conv)

            white_conv = model.forward_img_to_conv(white_inputs)
            white_outputs = model.forward_conv_to_logit(white_conv)

            # loss         #########################################################
            ce_loss = smoothing_criterion(white_outputs, labels)
            attention_pairing_loss = pairing_criterion(model.forward_conv_to_attention(white_conv, normalize_type=None),
                                                       model.forward_conv_to_attention(conv, normalize_type=None))
            total_loss = ce_loss + args.attention_pairing_lambda * attention_pairing_loss

            # test_str += "%.8f  " % proposed_loss.item()
            print(ce_loss.item(), attention_pairing_loss.item())

            # print(test_str)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_iter += 1
            total_ce_loss += ce_loss.item()
            total_attention_pairing_loss += attention_pairing_loss.item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            white_correct += white_outputs.max(1)[1].eq(labels).sum().item()

        eval_dict['TrainCELoss'] = total_ce_loss / total_iter
        eval_dict['TrainAttentionLoss'] = total_attention_pairing_loss / total_iter
        eval_dict['TrainAcc'] = correct / num_train
        eval_dict['WhiteTrainAcc'] = white_correct / num_train

        lr_scheduler.step()
        ### 2. eval ##############################################################
        model.eval()

        correct = 0
        white_correct = 0
        for data in valid:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            white_inputs = pgd.generate(inputs, y=labels)

            correct += model(inputs).max(1)[1].eq(labels).sum().item()
            white_correct += model(white_inputs).max(1)[1].eq(labels).sum().item()

        eval_dict['ValidAcc'] = correct / num_valid
        eval_dict['WhiteValidAcc'] = white_correct / num_valid

        print_str = "[Epoch %3d] CELoss: %.8f  AttentionLoss: %.8f  " % (
            epoch + 1, eval_dict['TrainCELoss'], eval_dict['TrainAttentionLoss']
        )
        print_str += "Acc: %.2f%%  WhiteAcc: %.2f%%  ValidAcc: %.2f%%  WhiteValidAcc: %.2f%%" % (
            100 * eval_dict['TrainAcc'], 100 * eval_dict['WhiteTrainAcc'],
            100 * eval_dict['ValidAcc'], 100 * eval_dict['WhiteValidAcc']
        )

        logger.print(print_str)

        ### 3. save ##############################################################
        if args.save:
            save_eval_dict = {
                'epoch': epoch,
                'eval_dict': eval_dict
            }
            torch_saver.save(save_dict=save_eval_dict, file_name="Eval_Epoch_%d.tar" % (epoch + 1),
                             should_not_exist=True)

            save_state_dict = {
                'epoch': epoch,
                'eval_dict': eval_dict,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }
            for measure in best_eval_dict.keys():
                worst_in_best_idx = np.argmin(np.array(best_eval_dict[measure]))
                if best_eval_dict[measure][worst_in_best_idx] < eval_dict[measure]:
                    best_eval_dict[measure][worst_in_best_idx] = eval_dict[measure]
                    torch_saver.save(save_dict=save_state_dict, file_name="Best_%s_%d.tar" % (measure, worst_in_best_idx),
                                     should_not_exist=False)

            if (epoch + 1) % args.save_model_term == 0:
                torch_saver.save(save_dict=save_state_dict, file_name="State_Epoch_%d.tar" % (epoch + 1),
                                 should_not_exist=True)

    ###################################################################################################################
    logger.print(" ")
    logger.print("=================== Performance at BestWhiteValidAcc ===================")
    for idx in range(args.save_num_best):
        best_white_valid_path = torch_saver.get_saved_path("Best_WhiteValidAcc_%d.tar" % idx)
        cp = torch.load(best_white_valid_path, map_location=gpu_name)
        model.load_state_dict(cp['model_state_dict'])

        model.eval()

        correct = 0
        white_correct = 0
        for data in test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            correct += model(inputs).max(1)[1].eq(labels).sum().item()
            white_correct += multi_rs_pgd_eval.eval(inputs, y=labels, multiple=args.eval_num_restart).sum().item()

        test_acc = correct / num_test
        white_test_acc = white_correct / num_test

        logger.print("[BestWhiteValid %d] TestAcc: %.2f%%  WhiteTestAcc: %.2f%%" % (
            idx, 100 * test_acc, 100 * white_test_acc,
        ))


# %%

if __name__ == '__main__':
    main()
    logger.close()

# %%

