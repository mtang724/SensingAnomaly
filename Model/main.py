from DGAD import DGAD
from DGAD_rnn import DGAD_rnn
from LOF import LOF
from GAE import GAD
import argparse
from utils import *
import os
#from torch.backends import cudnn
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Data_dict = {'sensor': [],
             'reddit_data': [150, 50, 3199, 2411, 16, 300, 64, 300],
             'har': [7352, 2947, 18, 18, 16, 79, 64, 79],
             'har_clean': [6366, 2947, 18, 18, 16, 79, 64, 79],
             'nnn': [95, 80, 10, 10, 10, 10, 64, 10]}  # TODO: Modify

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of 3dgraphconv"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test or smooth_test')
    parser.add_argument('--dataset', type=str, default='nnn', help='dataset_name: reddit_data/har/har_clean')
    parser.add_argument('--model', type=str, default='CNN', help='CNN/LOF/GAE/RNN')
    parser.add_argument('--dataset_setting', type=dict, default=Data_dict,
                        help='0 train_len, 1 test_len, 2 number of trn node, 3 number of dev node,' +
                             '4 sensor embedding dim, 5 input channel, 6 conv_channel, 7 Original dim')

    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

    parser.add_argument('--decay_flag', type=bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=1, help='decay epoch')

    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    # parser.add_argument('--iteration', type=int, default=2000, help='The number of training iterations')##
    parser.add_argument('--new_start', type=bool, default=True, help='new_start')

    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')

    parser.add_argument('--rx_w', type=float, default=0.5, help='weight of Graph reconstruction error')
    parser.add_argument('--nx_w', type=float, default=0.5, help='weight of masked nodes reconstruction error')
    parser.add_argument('--loss_function', type=str, default='l2_loss', help='loss function type [l1_loss / l2_loss/ cross_entropy]')
    parser.add_argument('--edge_corr', type=bool, default=False,
                        help='If False, learning an embedding; otherwise, claculate average RV correlation as edge.')

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=5, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=2000, help='The number of ckpt_save_freq')
    parser.add_argument('--print_net', type=bool, default=False, help='print_net')
    parser.add_argument('--use_tensorboard', type=bool, default=False, help='use_tensorboard')

    parser.add_argument('--num_clips', type=int, default=3, help='The size of clip')
    parser.add_argument('--conv_ch', type=int, default=0, help='The base channel number per layer')
    parser.add_argument('--head', type=int, default=4, help='Number of head for attention')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main(**setting):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    #cudnn.benchmark = True
    if args.model == 'CNN':
        gae = DGAD(args)
        print('Model: {}'.format(args.model))
    elif args.model == 'RNN':
        gae = DGAD_rnn(args)
        print('Model: {}'.format(args.model))
    elif args.model == 'LOF':
        gae = LOF(args)
    elif args.model == 'GAE':
        gae = GAD(args)
        print('Model: {}'.format(args.model))

    torch.backends.cudnn.benchmark = True

    if args.phase == 'train':
        if args.model != 'LOF':
            # launch the graph in a session
            gae.train()
            print(" [*] Training finished!")
            print("\n\n\n")
        gae.test()
        print(" [*] Test finished!")

    if args.phase == 'test':
        gae.test()
        print(" [*] Test finished!")

    if args.phase == 'smooth_test':
        gae.test_other()
        print(" [*] Test finished!")


if __name__ == '__main__':
    #main()
    main(phase='train', model='GAD', resume_iters=0, lr=0.005, epoch=10, num_clips=3, conv_ch=32, denoising=0.2)