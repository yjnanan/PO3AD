import argparse

# config para
def get_parser():

    # # Train config
    parser = argparse.ArgumentParser(description='3D anomaly detection')
    parser.add_argument('--task', type=str, default='train', help='task: train or test')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=1001, help='Total epoch')
    parser.add_argument('--num_works', type=int, default=12, help='num_works for dataset')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=1, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str, default='./log/ashtray0/', help='path to save logs')
    parser.add_argument('--validation', type=bool, default=False, help='Whether to verify the validation set')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', help='datasets')
    parser.add_argument('--category', type=str, default='ashtray0', help='categories for each class')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for single GPU')
    parser.add_argument('--data_repeat', type=int, default=100, help='repeat the date for each epoch')
    parser.add_argument('--mask_num', type=int, default=64)

    # #Adjust learning rate
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--step_epoch', type=int, default=10, help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.5, help='Learning rate decay: lr = lr * multiplier')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for SGD')

    # #model parameter
    parser.add_argument('--voxel_size', type=float, default=0.03, help='voxel size')
    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--out_channels', type=int, default=32, help='backbone feat channels')


    args = parser.parse_args()
    return args