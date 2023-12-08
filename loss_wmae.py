def wmae_loss():
    def loss(y_true, y_pred):
        coeff = 5
        y_true1, y_pred1 = (255 - y_true) / 255, (255 - y_pred) / 255
        dif = y_true1 - y_pred1
        temp1 = K.l2_normalize(K.cast(K.greater(dif, 0),"float32")*y_true1,axis=-1) * coeff
        temp2 = K.cast(K.less(dif, 0),"float32")
        weight = temp1 + temp2  
        loss = K.abs(weight*dif)
        return K.mean(loss) 
    return loss
