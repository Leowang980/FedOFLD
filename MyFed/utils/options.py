import argparse
def args_parser():
    parser=argparse.ArgumentParser()
    
    #global parameters
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--communication_round', type=int, default=200)
    parser.add_argument('--all_client', type=int, default=20)
    parser.add_argument('--each_client', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5, help='smaller, more non-iid')
    parser.add_argument('--method', type=str, default='FedAvg')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument("--resume", type=bool, default=False)

    #local parameters
    parser.add_argument('--local_epoch', type=int, default=20)
    parser.add_argument('--local_optimizer', type=str, default='sgd')
    parser.add_argument('--local_learning_rate', type=float, default=0.005)
    parser.add_argument('--local_momentum', type=float, default=0.9)
    parser.add_argument('--local_weight_decay', type=float, default=1e-5)
    parser.add_argument('--local_batch_size', type=int, default=64)

    #distill parameters
    parser.add_argument('--warmup_round', type=int, default=10, help='must be smaller than communication_round')
    parser.add_argument('--distill_epoch', type=int, default=10)
    parser.add_argument('--distill_batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--distill_optimizer', type=str, default='sgd')
    parser.add_argument('--distill_learning_rate', type=float, default=0.005)
    parser.add_argument('--distill_momentum', type=float, default=0.9)
    parser.add_argument('--distill_weight_decay', type=float, default=1e-5)
    parser.add_argument('--distill_scheduler', type=str, default='CosineAnnealingLR')

    #feature_distill parameters
    parser.add_argument('--beta', type=float, default=0.5, help='bigger, more feature distill')
    parser.add_argument('--gamma', type=float, default=0.5, help='bigger, more logit distill')

    #Hetero parameters
    parser.add_argument('--model_num', type=int, default=3)
    parser.add_argument('--model_level', type=list, default=[1, 0.7, 0.4])
    parser.add_argument('--model_proportion', type=list, default=[0.4, 0.3, 0.3])

    args=parser.parse_args()
    return args
