import yaml
import configargparse

def get_train_config():
    parser = configargparse.ArgumentParser()
    parser.add('--conf', help='training config')
    parser.add('--train', help='training data')
    parser.add('--dev', help='development data')
    parser.add('--resume', help='loading file to keep training')
    parser.add('--features', default='exp', help='experiment directory')
    parser.add('--exp', default='exp', help='experiment directory')
    parser.add('--device', default='cpu', help='use cpu or cuda')
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    return args, conf

def get_test_config():
    parser = configargparse.ArgumentParser()
    parser.add('--conf', help='training config')
    parser.add('--test', help='test data')
    parser.add('--features', default='exp', help='experiment directory')
    parser.add('--exp', default='exp', help='experiment directory')
    parser.add('--device', default='cpu', help='use cpu or cuda')
    parser.add('--start', default='cpu', help='start testing epoch')
    parser.add('--end', default='cpu', help='end testing epoch')
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    return args, conf
