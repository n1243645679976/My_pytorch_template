import yaml
import configargparse

def get_config():
    parser = configargparse.ArgumentParser()
    parser.add('--conf', help='training config')
    parser.add('--train', help='training data')
    parser.add('--dev', help='development data')
    parser.add('--resume', help='loading file to keep training')
    parser.add('--exp', default='exp', help='experiment directory')
    parser.add('--device', default='cpu', help='use cpu or cuda')
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    return args, conf