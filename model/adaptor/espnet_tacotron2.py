import yaml
import configargparse
import torch
from utils.dynamic_import import dynamic_import
from .base import baseAdaptor
from espnet.utils.cli_utils import strtobool



def get_parser():
    """Get parser of decoding arguments."""
    parser = configargparse.ArgumentParser(
        description="Synthesize speech from text using a TTS model on one CPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="second config file path that overwrites the settings in `--config`.",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="third config file path that overwrites "
        "the settings in `--config` and `--config2`.",
    )

    # decoding related
    parser.add_argument(
        "--maxlenratio", type=float, default=5, help="Maximum length ratio in decoding"
    )
    parser.add_argument(
        "--minlenratio", type=float, default=0, help="Minimum length ratio in decoding"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold value in decoding"
    )
    parser.add_argument(
        "--use-att-constraint",
        type=strtobool,
        default=False,
        help="Whether to use the attention constraint",
    )
    parser.add_argument(
        "--backward-window",
        type=int,
        default=1,
        help="Backward window size in the attention constraint",
    )
    parser.add_argument(
        "--forward-window",
        type=int,
        default=3,
        help="Forward window size in the attention constraint",
    )
    parser.add_argument(
        "--fastspeech-alpha",
        type=float,
        default=1.0,
        help="Alpha to change the speed for FastSpeech",
    )
    # save related
    parser.add_argument(
        "--save-durations",
        default=False,
        type=strtobool,
        help="Whether to save durations converted from attentions",
    )
    parser.add_argument(
        "--save-focus-rates",
        default=False,
        type=strtobool,
        help="Whether to save focus rates of attentions",
    )
    return parser

class espnetTacotron2Adaptor(torch.nn.Module):
    def __init__(self, conf):
        super(espnetTacotron2Adaptor, self).__init__()
        parser = configargparse.ArgumentParser()
        parser.add("--config", is_config_file=True, help="config file path")
        parser.add("--idim", help='input_dimension')
        parser.add("--odim", help='output_dimension')
        parser.add('--inference_conf', help='inference config file')

        if 'config_file' in conf:
            with open(conf['config_file']) as f:
                loaded_conf = yaml.safe_load(f)
            module_class_conf.update(loaded_conf)
        parser = model_class.add_argument(parser)
        inference_parser = get_parser()

        args = parser.parser(['--config', conf_file])
        self.model = model_class(args.idim, args.odim)
        self.inference_args = inference_parser.parser(['--config', args.inference_conf])
        
    def forward(self, packed_data):
        xs = packed_data[0].data
        ilens = packed_data[0].len.reshape[-1]
        ys = packed_data[1].data
        olens = packed_data[1].len.reshape[-1]
        loss = self.model(xs, ilens, ys, olens)

        packed_data['_loss']['overall_loss'] = loss
        packed_data['_loss']['update_loss'] = loss / self.accum_grad

        return packed_data

    def inference(self, packed_data):
        xs = packed_data[0].data
        outs, probs, att_ws = self.model(xs, self.inference_args)
        return outs
        
class espnetTacotron2Adaptor(baseAdaptor):
    def __init__(self, conf_file):
        super(baseAdaptor, self).__init__()
        parser = configargparse.ArgumentParser()
        parser.add("--config", is_config_file=True, help="config file path")
        parser.add("--idim", type=int, help='input_dimension')
        parser.add("--odim", type=int, help='output_dimension')
        parser.add('--inference_conf', help='inference config file')

        model_class = dynamic_import(conf_file['model_class'])
        parser = model_class.add_arguments(parser)
        args = parser.parse_args(['--config', conf_file['config_file']])
        self.model = model_class(args.idim, args.odim)
        inference_parser = get_parser()
        self.inference_args = inference_parser.parse_args(['--config', args.inference_conf])
        
    def forward(self, packed_data):
        xs = packed_data[0].data.long()
        ilens = packed_data[0].len
        ys = packed_data[1].data
        olens = packed_data[1].len
        sortlist = [(x, ilen, y, olen) for x, ilen, y, olen in zip(xs, ilens, ys, olens)]
        sortlist = sorted(sortlist, key=lambda x:-x[1])
        xs = torch.cat([s[0].unsqueeze(0) for s in sortlist], dim=0)
        ilens = torch.cat([s[1].unsqueeze(0) for s in sortlist], dim=0)
        ys = torch.cat([s[2].unsqueeze(0) for s in sortlist], dim=0)
        olens = torch.cat([s[3].unsqueeze(0) for s in sortlist], dim=0)

        labels = torch.zeros(ys.shape[0], ys.shape[1]).to(ys.device)
        for i, l in enumerate(olens):
            labels[i, l-1:] = 1.0
        loss = self.model(xs=xs, ilens=ilens, ys=ys, labels=labels, olens=olens)

        loss = {'overall_loss':loss}
        return [loss]

    def inference(self, packed_data):
        xs = packed_data[0].data.long().reshape(-1)
        outs, probs, att_ws = self.model.inference(xs, self.inference_args)
        print(outs.shape)
        return [outs.unsqueeze(0)]

class espnetTacotron2AdaptorSpkemb(baseAdaptor):
    def __init__(self, conf):
        super(espnetTacotron2Adaptor, self).__init__()
        parser = configargparse.ArgumentParser()
        parser.add("--config", is_config_file=True, help="config file path")
        parser.add("--idim", help='input_dimension')
        parser.add("--odim", help='output_dimension')
        parser.add('--inference_conf', help='inference config file')

        
        parser = model_class.add_argument(parser)
        args = parser.parser(['--config', conf_file])
        self.model = model_class(args.idim, args.odim)
        inference_parser = get_parser()
        self.inference_args = inference_parser.parser(['--config', args.inference_conf])
        
    def forward(self, batchxs):
        xs = batchxs[0].data
        ilens = batchxs[0].len
        ys = batchxs[1].data
        olens = batchxs[1].len
        spkemb = batchxs[2].data
        loss = self.model(xs, ilens, ys, olens, spkemb=spkemb)

        _loss = {'overall_loss': loss}
        return _loss

    def inference(self, packed_data):
        xs = packed_data[0].data
        spkemb = packed_data[1].data
        outs, probs, att_ws = self.model(xs, self.inference_args, spkemb-spkemb)
        return outs