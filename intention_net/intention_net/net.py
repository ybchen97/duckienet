from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2
from keras.layers import (
        Input,
        Flatten,
        Dense,
        Dropout,
        Lambda,
        concatenate,
        BatchNormalization,
        )
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras import backend as K

INIT='he_normal'
L2=1e-5
DROPOUT=0.3

def filter_control(args):
    outs, intention = args[:-1], args[-1]
    outs = K.concatenate(outs, axis=0)
    batch_size = K.shape(intention)[0]
    intention_idx = K.cast(K.argmax(intention), 'int32') * batch_size + K.arange(0, batch_size)
    #return outs[intention_idx, :]
    return K.gather(outs, intention_idx)

def FCModel(input_length):
    input = Input(shape=(input_length, ))
    x = Dense(64, kernel_initializer=INIT, kernel_regularizer=l2(L2), activation='relu')(input)
    x = Dropout(DROPOUT)(x)
    model = Model(inputs=input, outputs=x)
    return model

def FeatModel():
    feat_model = ResNet50(weights='imagenet')
    layer_dict = dict([(l.name, l) for l in feat_model.layers])
    inp = feat_model.layers[0].input
    oup = layer_dict['avg_pool'].output
    return Model(inputs=inp, outputs=oup)

def IntentionNet(mode, input_frame, num_control, num_intentions=-1,use_side_model=True):
    print (f'Intention Mode {mode} Input frame {input_frame}')
    # model
    feat_model = FeatModel()
    
    # Input for intention net
    if input_frame != 'MULTI':
        rgb_input = Input(shape=(224, 224, 3))
        rgb_feat = [feat_model(rgb_input)]
    else:
        rgbl_input = Input(shape=(224, 224, 3))
        rgbm_input = Input(shape=(224, 224, 3))
        rgbr_input = Input(shape=(224, 224, 3))
        
        if use_side_model: #use different model for side view
            side_feat_model = FeatModel()
            rgbl_feat = side_feat_model(rgbl_input)
            rgbm_feat = feat_model(rgbm_input)
            rgbr_feat = side_feat_model(rgbr_input)
        else:
            rgbl_feat = feat_model(rgbl_input)
            rgbm_feat = feat_model(rgbm_input)
            rgbr_feat = feat_model(rgbr_input)

        rgbl_feat = Dropout(DROPOUT)(rgbl_feat)
        rgbl_feat = Dense(512,kernel_initializer=INIT,kernel_regularizer=l2(L2),activation='relu')(rgbl_feat)

        rgbm_feat = Dropout(DROPOUT)(rgbm_feat)
        rgbm_feat = Dense(1024,kernel_initializer=INIT,kernel_regularizer=l2(L2),activation='relu')(rgbm_feat)
        
        rgbr_feat = Dropout(DROPOUT)(rgbr_feat)
        rgbr_feat = Dense(512,kernel_initializer=INIT,kernel_regularizer=l2(L2),activation='relu')(rgbr_feat)

        rgb_feat = [rgbl_feat, rgbm_feat, rgbr_feat]

    if mode == 'DLM':
        assert (num_intentions != -1), "Number of intentions must be bigger than one"
        intention_input = Input(shape=(num_intentions,))
        intention_feat = FCModel(num_intentions)(intention_input)
        #speed_input = Input(shape=(1,))
        #speed_feat = FCModel(1)(speed_input)
        #feat = concatenate(rgb_feat + [intention_feat, speed_feat])
        feat = concatenate(rgb_feat + [intention_feat])
        # controls
        outs = []
        for i in range(num_intentions):
            out = Dropout(DROPOUT)(feat)
            #out = Dense(1024, kernel_initializer=INIT, kernel_regularizer=l2(L2), activation='relu')(out)
            #out = Dropout(DROPOUT)(out)
            out = Dense(num_control, kernel_initializer=INIT, kernel_regularizer=l2(L2))(out)
            outs.append(out)
        outs.append(intention_input)
        print ('num_intentions', num_intentions)
        control = Lambda(filter_control, output_shape=(num_control, ))(outs)

        if input_frame != 'MULTI':
            #model = Model(inputs=[rgb_input, intention_input, speed_input], outputs=control)
            model = Model(inputs=[rgb_input, intention_input], outputs=control)
        else:
            model = Model(inputs=[rgbl_input, rgbm_input, rgbr_input, intention_input], outputs=control)

    else:
        if mode == 'LPE_SIAMESE':
            lpe_input = Input(shape=(224, 224, 3))
            lpe_feat = feat_model(lpe_input)
        else:
            assert (mode == 'LPE_NO_SIAMESE'), "LPE WITHOUT SIAMESE ARCHITECTURE"
            lpe_input = Input(shape=(224, 224, 3))
            lpe_feat = FeatModel()(lpe_input)
        speed_input = Input(shape=(1,))
        speed_feat = FCModel(1)(speed_input)
        feat = concatenate(rgb_feat + [lpe_feat, speed_feat])
        out = Dropout(DROPOUT)(feat)
        out = Dense(2048, kernel_initializer=INIT, kernel_regularizer=l2(L2), activation='relu')(out)
        out = Dropout(DROPOUT)(out)
        control = Dense(num_control, kernel_initializer=INIT, kernel_regularizer=l2(L2))(out)

        if input_frame != 'MULTI':
            model = Model(inputs=[rgb_input, lpe_input, speed_input], outputs=control)
        else:
            model = Model(inputs=[rgbl_input, rgbm_input, rgbr_input, lpe_input, speed_input], outputs=control)

    return model

def test():
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.resnet50 import preprocess_input
    from keras.utils import to_categorical
    import numpy as np
    feat_model = FeatModel()

    img = load_img('/home/gaowei/dog.jpeg', target_size=(224, 224))
    img = preprocess_input(img_to_array(img))
    img = np.expand_dims(img, axis=0)

    net = IntentionNet('DLM', 2, 4)
    print (net.summary())
    control = net.predict([img, to_categorical([1], num_classes=4), np.array([[1.]])])
    print (control)

#test()
