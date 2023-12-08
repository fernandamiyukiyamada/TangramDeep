
def cae(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x1 = tf.keras.layers.Conv2D(48, kernel_size=(11, 11), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(input_layer)
    x1 = tf.keras.layers.BatchNormalization(axis=-1)(x1)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)

    x2 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x1)
    x2 = tf.keras.layers.Dropout(.2)(x2)
    x2 = tf.keras.layers.BatchNormalization(axis=-1)(x2)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)

    x2_0 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2)
    x2_0 = tf.keras.layers.Dropout(.2)(x2_0)
    x2_0 = tf.keras.layers.BatchNormalization(axis=-1)(x2_0)
    x2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2_0)

    x3 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2_0)
    x3 = tf.keras.layers.Dropout(.2)(x3)
    x3 = tf.keras.layers.BatchNormalization(axis=-1)(x3)
    x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)

    x3_0 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3)
    x3_0 = tf.keras.layers.Dropout(.2)(x3_0)
    x3_0 = tf.keras.layers.BatchNormalization(axis=-1)(x3_0)
    x3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3_0)

    x4 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3_0)
    x4 = tf.keras.layers.Dropout(.2)(x4)
    x4 = tf.keras.layers.BatchNormalization(axis=-1)(x4)
    x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)

    x4_0 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4)
    x4_0 = tf.keras.layers.Dropout(.2)(x4_0)
    x4_0 = tf.keras.layers.BatchNormalization(axis=-1)(x4_0)
    x4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4_0)

    x5 = tf.keras.layers.Conv2D(48, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4_0)
    x5 = tf.keras.layers.BatchNormalization(axis=-1)(x5)
    x5 = tf.keras.layers.LeakyReLU(alpha=0.2)(x5)

    tr1 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x5)
    tr1 = tf.keras.layers.BatchNormalization(axis=-1)(tr1)
    tr1 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr1)

    tr2 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr1, x4_0]))
    tr2 = tf.keras.layers.Dropout(.2)(tr2)
    tr2 = tf.keras.layers.BatchNormalization(axis=-1)(tr2)
    tr2 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2)

    tr2_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2, x4]))
    tr2_0 = tf.keras.layers.Dropout(.2)(tr2_0)
    tr2_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr2_0)
    tr2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2_0)

    tr3 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2_0, x3_0]))
    tr3 = tf.keras.layers.Dropout(.2)(tr3)
    tr3 = tf.keras.layers.BatchNormalization(axis=-1)(tr3)
    tr3 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3)

    tr3_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3, x3]))
    tr3_0 = tf.keras.layers.Dropout(.2)(tr3_0)
    tr3_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr3_0)
    tr3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3_0)

    tr4 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3_0, x2_0]))
    tr4 = tf.keras.layers.Dropout(.2)(tr4)
    tr4 = tf.keras.layers.BatchNormalization(axis=-1)(tr4)
    tr4 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4)

    tr4_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4, x2]))
    tr4_0 = tf.keras.layers.Dropout(.2)(tr4_0)
    tr4_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr4_0)
    tr4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4_0)

    output = tf.keras.layers.Conv2D(input_shape[2], kernel_size=(1, 1), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4_0, x1]))
    output = tf.keras.layers.BatchNormalization(axis=-1)(output)
    output = tf.keras.layers.Activation("softplus")(output)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])
    return model

def vae(input_shape):

    latent_dim = 2

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x1 = tf.keras.layers.Conv2D(48, kernel_size=(11, 11), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(input_layer)
    x1 = tf.keras.layers.BatchNormalization(axis=-1)(x1)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)

    x2 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x1)
    x2 = tf.keras.layers.Dropout(.2)(x2)
    x2 = tf.keras.layers.BatchNormalization(axis=-1)(x2)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)

    x2_0 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2)
    x2_0 = tf.keras.layers.Dropout(.2)(x2_0)
    x2_0 = tf.keras.layers.BatchNormalization(axis=-1)(x2_0)
    x2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2_0)

    x3 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2_0)
    x3 = tf.keras.layers.Dropout(.2)(x3)
    x3 = tf.keras.layers.BatchNormalization(axis=-1)(x3)
    x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)

    x3_0 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3)
    x3_0 = tf.keras.layers.Dropout(.2)(x3_0)
    x3_0 = tf.keras.layers.BatchNormalization(axis=-1)(x3_0)
    x3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3_0)

    x4 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3_0)
    x4 = tf.keras.layers.Dropout(.2)(x4)
    x4 = tf.keras.layers.BatchNormalization(axis=-1)(x4)
    x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)

    x4_0 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4)
    x4_0 = tf.keras.layers.Dropout(.2)(x4_0)
    x4_0 = tf.keras.layers.BatchNormalization(axis=-1)(x4_0)
    x4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4_0)

    x5 = tf.keras.layers.Conv2D(48, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4_0)
    x5 = tf.keras.layers.BatchNormalization(axis=-1)(x5)
    x5 = tf.keras.layers.LeakyReLU(alpha=0.2)(x5)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x5)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x5)
    z = Sampling()([z_mean, z_log_var])

    tr1 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(z)
    tr1 = tf.keras.layers.BatchNormalization(axis=-1)(tr1)
    tr1 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr1)

    tr2 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr1, x4_0]))
    tr2 = tf.keras.layers.Dropout(.2)(tr2)
    tr2 = tf.keras.layers.BatchNormalization(axis=-1)(tr2)
    tr2 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2)

    tr2_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2, x4]))
    tr2_0 = tf.keras.layers.Dropout(.2)(tr2_0)
    tr2_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr2_0)
    tr2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2_0)

    tr3 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2_0, x3_0]))
    tr3 = tf.keras.layers.Dropout(.2)(tr3)
    tr3 = tf.keras.layers.BatchNormalization(axis=-1)(tr3)
    tr3 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3)

    tr3_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3, x3]))
    tr3_0 = tf.keras.layers.Dropout(.2)(tr3_0)
    tr3_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr3_0)
    tr3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3_0)

    tr4 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3_0, x2_0]))
    tr4 = tf.keras.layers.Dropout(.2)(tr4)
    tr4 = tf.keras.layers.BatchNormalization(axis=-1)(tr4)
    tr4 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4)

    tr4_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4, x2]))
    tr4_0 = tf.keras.layers.Dropout(.2)(tr4_0)
    tr4_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr4_0)
    tr4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4_0)

    output = tf.keras.layers.Conv2D(input_shape[2], kernel_size=(1, 1), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4_0, x1]))
    output = tf.keras.layers.BatchNormalization(axis=-1)(output)
    output = tf.keras.layers.Activation("softplus")(output)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])
    return model

def unet(input_shape):
    inputs = Input(input_shape)
    
    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Expansive path
    up5 = UpSampling2D(size=(2, 2))(conv4)
    concat5 = Concatenate()([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(concat5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    concat6 = Concatenate()([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(concat6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    concat7 = Concatenate()([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(concat7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = keras.layers.Input(shape=image_shape)
    in_target_image = keras.layers.Input(shape=image_shape)
    merged = keras.layers.Concatenate()([in_src_image, in_target_image])

    d = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)

    d = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = keras.layers.Activation('sigmoid')(d)

    model = keras.models.Model([in_src_image, in_target_image], patch_out)
    return model