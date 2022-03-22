"""
If given diction_path
"""

import os
import argparse
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--token', help='token file e.g. data/train/text')
parser.add_argument('--tokenid', help='tokenid file e.g. data/train/tokenid')
parser.add_argument('--diction_path', default=None, help='diction path e.g. data/train/diction')
parser.add_argument('--save_diction_path', help='save diction path, do not save when --diction_path is set e.g. data/train/diction')
parser.add_argument('--token_type', choices=['char', 'token'], help='token type to generate tokenid, supported: char and token')
parser.add_argument('--space', action='store_true', help='change space to space token, this will be effect when --token_type is token')
parser.add_argument('--eos', action='store_true', help='add eos to the end of token list')
args = parser.parse_args()

_token2tokenid = defaultdict(int) # this will return 0 (OOV) when not exist
_token2tokenid = {'<OOV>': '0', '<EOS>': '1', '<SPACE>': '2'}
# 0~15 is reserved
max_tokenid = 15
OOVs = 0

diction_exist = os.path.isfile(args.diction_path) if args.diction_path else False
def token2tokenid(x):
    global OOVs
    if x == ' ':
        return _token2tokenid['<SPACE>']
    if x not in _token2tokenid:
        OOVs += 1
        return _token2tokenid['<OOV>']
    return _token2tokenid[x]

def initial_diction():
    if diction_exist:
        print(f'diction exist: {args.diction_path}, load it')
        with open(args.diction_path) as f:
            for line in f.read().splitlines():
                token, tokenid = line.split()
                _token2tokenid[token] = tokenid

    else:
        _tokens = set()
        global max_tokenid
        with open(args.token) as token_fd:
            for line in token_fd.read().splitlines():
                key, tokens = line.split(' ', maxsplit=1)
                if args.token_type == 'char':
                    token_list = list(tokens)
                elif args.token_type == 'token':
                    token_list = tokens.split()
                _tokens |= set(token_list)
        
            for token in sorted(_tokens):
                if token == ' ':
                    continue
                if token not in _token2tokenid:
                    max_tokenid += 1
                    _token2tokenid[token] = str(max_tokenid)
                else:
                    print(f'WARNING: {token} is initialed but shown in {args.token}')

def write_tokenid():
    with open(args.token) as token_fd, open(args.tokenid, 'w+') as tokenid_fd:
        for line in token_fd.read().splitlines():
            key, tokens = line.strip().split(' ', maxsplit=1)
            if args.token_type == 'char':
                token_list = list(tokens)
            elif args.token_type == 'token':
                if args.space:
                    tokens = tokens.replace(' ', ' <SPACE> ')
                token_list = tokens.split()

            if args.eos:
                if args.space:
                    token_list.append('<SPACE>')
                token_list.append('<EOS>')
            tokenid_list = list(map(token2tokenid, token_list))
            tokenids = ' '.join(tokenid_list)
            tokenid_fd.write(f'{key} {tokenids}\n')

def write_diction():
    print(f'write to diction: {args.save_diction_path}')
    print(_token2tokenid)
    with open(args.save_diction_path, 'w+') as save_diction_path_fd:
        for key, value in sorted(_token2tokenid.items(), key=lambda x: int(x[1])):
            save_diction_path_fd.write(f'{key} {value}\n')

def main():
    initial_diction()
    write_tokenid()
    if OOVs > 0:
        print(f'WARNING: find {OOVs} OOVs')
    if not diction_exist:
        write_diction()

if __name__ == '__main__':
    main()