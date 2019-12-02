def init_model(dataset, pointrcnn_filename, logger):
    """
    """
    model = PointRCNN(num_classes=dataset.num_classes, use_xyz=True, mode='TEST')
    model.cuda()
    
    if not os.path.isfile(pointrcnn_filename):
        raise FileNotFoundError(pointrcnn_filename)

    logger.info("==> Loading PointRCNN from '{}'".format(pointrcnn_filename))
    checkpoint = torch.load(pointrcnn_filename)
    model.load_state_dict(checkpoint['model_state'])
    total_keys = len(model.state_dict().keys())
    
    model.eval()
    return model

