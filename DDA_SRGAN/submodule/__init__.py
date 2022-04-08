from submodule import generator, discriminator, feature_dis

def load_module(model_arc, model_params):   
    
    if model_arc == 'generator':
        model = generator.build_generator(**model_params)
            
    elif model_arc == 'discriminator':
        model = discriminator.build_discriminator(**model_params)
            
    elif model_arc == 'feature_dis':
        model = feature_dis.build_feature_dis(**model_params)
            
    else:
        raise Exception('Wrong model architecture! It should be generator, discriminator, feature_dis, style_trfs.')
    return model 