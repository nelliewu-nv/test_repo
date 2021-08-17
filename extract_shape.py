import argparse
import logging
import re
import copy
import pprint
import pickle

import torch
import torch.nn

logger = logging.getLogger(__name__)

BATCH_SIZE = 512

GEMM_template = {"instance": {"M": 0, "K": 0, "N": int(BATCH_SIZE),
                           "densities":
                               {
                               "A":
                                   {"distribution": "fixed-structured",
                                     "density": 0.5
                                   }
                               }
                           }
                 }

def convert_to_timeloop_dict(model):
    timeloop_specs = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            timeloop_repr = copy.deepcopy(GEMM_template)
            M = m.out_features
            K = m.in_features
            timeloop_repr["instance"]["M"] = M
            timeloop_repr["instance"]["K"] = K
            timeloop_specs[name] = timeloop_repr
    return timeloop_specs

def summarize_representative_layers(timeloop_specs):
    repr_layer_shapes = {}
    for layer_name, layer_spec in timeloop_specs.items():
        layer_key = "M" + str(layer_spec["instance"]["M"]) + "-K" + str(layer_spec["instance"]["K"]) + \
        "-N" + str(layer_spec["instance"]["N"])
        if layer_key in repr_layer_shapes:
            repr_layer_shapes[layer_key]["count"]  = repr_layer_shapes[layer_key]["count"] + 1
            repr_layer_shapes[layer_key]["layer_names"].append(layer_name)
        else:
            repr_layer_shapes[layer_key] = {}
            repr_layer_shapes[layer_key]["count"] = 1
            repr_layer_shapes[layer_key]["layer_names"] = [layer_name]
            repr_layer_shapes[layer_key]["timeloop_spec"] = copy.deepcopy(layer_spec)
    return repr_layer_shapes

def main():
    networks = ['bert-base',
                'bert-large',
                'transformer',
                'transformer-xl',
                'resnet50']
    schemes  = ['prune',
                'mn',
                'block',
                'ell']
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme',
                        choices=schemes,
                        type=str.lower,
                        help='Compression scheme')
    parser.add_argument('--sparsity',
                        type=float,
                        default=0.5,
                        help='Target sparsity ratio')
    parser.add_argument('--blocksize',
                        type=str.lower,
                        default='2x2',
                        help='Block size for block sparsity')
    parser.add_argument('--mn',
                        default='2:4',
                        help='M:N sparsity sizes')
    parser.add_argument('--pretrained',
                        action='store_true',
                        help='Load pretrained weights (default: False)')
    parser.add_argument('--gprune',
                        action='store_true',
                        help='Prune based on global threshold (default: False)')
    parser.add_argument('--network',
                        choices=networks,
                        type=str.lower,
                        required=True,
                        help='Network')
    parser.add_argument('--out',
                        type=str,
                        default='compressed.pt',
                        help='Compressed model')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Suppress verbose logging messages')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.WARNING if args.quiet else logging.INFO)

    try:
        from transformers import BertModel, BertConfig
        from transformers import TransfoXLModel, TransfoXLConfig
        import torchvision.models as models
    except ImportError:
        print('Please install Torchvision and HuggingFace Transformers libraries')

    hfnets = {
        'bert-base': ['bert-base-uncased', BertConfig, BertModel],
        'bert-large': ['bert-large-uncased', BertConfig, BertModel],
        'transformer-xl': ['transfo-xl-wt103', TransfoXLConfig, TransfoXLModel]
    }

    if args.network == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained, progress=True)
    elif args.network == 'transformer':
        logger.info('Loading pretrained Transformer model from Torch Hub')
        torch.hub.list('pytorch/fairseq')
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
        # Disable dropout
        en2de.eval()
        model = en2de.models[0]
    elif args.network in ['bert-base', 'bert-large', 'transformer-xl']:
        _name, _config, _model = hfnets[args.network]
        if args.pretrained:
            model = _model.from_pretrained(_name)
        else:
            config = _config.from_pretrained(_name)
            model = _model(config)
    else:
        raise RuntimeError(f'Unknown network: {args.network}')

    logger.info('Finished loading model')
    #print(model)

    named_modules = []
    if args.network in ['bert-base', 'bert-large']:
        #rx_int = 'encoder\.layer\.[0-9]+\.intermediate\.dense.weight'
        #rx_att = 'encoder\.layer\.[0-9]+\.attention\.output\.dense.weight'
        #rx_qkv = 'encoder\.layer\.[0-9]+\.attention\.self\.(query|key|value)\.weight'
        #rx_out = 'encoder\.layer\.[0-9]+\.output\.dense.weight'

        #for name, w in model.named_parameters():
        #    if (re.match(rx_int, name) or
        #        re.match(rx_att, name) or
        #        re.match(rx_out, name) or
        #        re.match(rx_qkv, name)):
        #            print(name)

        for name, m in model.named_modules():
            if name == 'pooler.dense':
                continue
            if isinstance(m, torch.nn.Linear):
                named_modules.append((name, m))
    elif args.network == 'transformer-xl':
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                named_modules.append((name, m))
    elif args.network == 'resnet50':
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                named_modules.append((name, m))
    elif args.network == 'transformer':
        for name, m in model.named_modules():
            if name == 'decoder.output_projection':
                continue
            if isinstance(m, torch.nn.Linear):
                named_modules.append((name, m))
    else:
        raise RuntimeError(f'Unknown network: {args.network}')

    timeloop_specs = convert_to_timeloop_dict(model)
    repr_layer_shapes = summarize_representative_layers(timeloop_specs)
    pprint.pprint(repr_layer_shapes)
    pickle.dump(timeloop_specs, open(args.network+"_specs.pkl", "wb"))
    logger.info('Per layer Timeloop specs saved to disk: ' + args.network + "_specs.pkl")
    pickle.dump(repr_layer_shapes, open(args.network + "_repr_specs.pkl", "wb"))
    logger.info('Repr layer Timeloop specs saved to disk: ' + args.network + "_repr_specs.pkl")

if __name__ == '__main__':
    main()
