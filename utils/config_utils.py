import yaml

def load_config(args, method):
    if args.config == 'none':
        conf = vars(args)
    elif 'config_paper' in args.config:
        # TODO:
        with open('config_paper.yaml') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            conf = yml[f'{method}_{args.model}']
    elif 'config/' in args.config:
        env = args.config.split('/')[1]
        with open('config_paper.yaml') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            conf = yml[env]
    else:
        raise ValueError(f"No format for config file '{args.config}'")
    
    return conf