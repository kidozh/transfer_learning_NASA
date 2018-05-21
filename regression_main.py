from numpy.core.multiarray import ndarray
from model import build_residual_model
from data import DataSet
import time
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# --------------- CONF -------------------------
PREDICT = False
LOG_DIR = './regression_log/'
DROPOUT = 0.2

# --------------- CONF -------------------------


data = DataSet()
y = data.vb_value
signal_input,catalog_input,number_input = data.signal_value,data.material_type,data.number_value  # type: (ndarray, ndarray, None)

print(y.shape)
for i in y:
    print('*',i)

plt.plot(y,label="y")
plt.show()

for depth in [20,15,10]:
    local_time = "%Y-%m-%d %H:%M:%S", time.localtime()
    train_name = 'regression_depth_%s_%s_%s'%(depth,DROPOUT,local_time)
    model_name = '%s.kerasmodel' % (train_name)
    weight_name = '%s.keras_weight'%(train_name)

    if not PREDICT:
        tb_cb = TensorBoard(log_dir=LOG_DIR+train_name)
        model = build_residual_model(signal_input.shape[1],signal_input.shape[2],catalog_input.shape[1],number_input.shape[1],y.shape[1],block_number=depth,dropout=DROPOUT)

        model.fit([signal_input,catalog_input,number_input],y,callbacks=[tb_cb],batch_size=4,epochs=1000,validation_split=0.2)
        model.save(model_name)
        model.save_weights(weight_name)

    else:
        from keras.models import load_model
        model = load_model(model_name)
        y_pred = model.predict([signal_input,catalog_input,number_input])
        plt.plot(y,label="y")
        plt.plot(y_pred,label="pred")
        plt.show()