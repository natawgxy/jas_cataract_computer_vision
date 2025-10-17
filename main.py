import streamlit as st
import numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl
from PIL import Image
import os
import matplotlib.pyplot as plt
try:
    from keras.applications.efficientnet import preprocess_input as _eff_preprocess
except Exception:
    _eff_preprocess = None

epsilon = 1e-8
def load_model_safely(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        keras.config.enable_unsafe_deserialization()
    except:
        pass 
    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    print("Input shape of the loaded model:", model.input_shape)
    return model

def get_img_array(img_path, size=(128, 128)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0) # делаем [batch_size=1(сама картинка), h, w, channels=3(rgb)]
    return x

def preprocess_input(img_arr):
    if _eff_preprocess is not None: # доступен effnet preprocess
        return _eff_preprocess(img_arr)
    else:
        return img_arr / 255.0 # нормализируем в [0,1]

def print_model_layers(model, prefix=""):
    for layer in getattr(model, "layers", []):
        print(prefix + layer.name, type(layer).__name__)
        if hasattr(layer, "layers"):
            print_model_layers(layer, prefix + " ")

# img_tensor [1, h, w, 3]
def grad_cam(img_tensor, model, last_conv_layer_name="top_conv", predicted_class_index=None):
    assert img_tensor is not None 
    if isinstance(img_tensor, np.ndarray):  
        assert img_tensor.ndim == 4 and img_tensor.shape[-1] == 3
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32) # теперь точно тензор
    aug_layer = None # если есть слой аугментации (если к sequential модели добавлены random flip/zoom)
    try:
        aug_layer = model.get_layer("sequential")
    except Exception:
        pass

    eff_net_base = model.get_layer("efficientnetb0") # получаем саму efficientnet
    feature_layer = eff_net_base.get_layer(last_conv_layer_name) # последний conv слой
    feature_extractor_model = keras.Model(inputs=eff_net_base.input, outputs=feature_layer.output) # получаем feature map
    # вытягиваем голову модели (єти слои идут сразу же после effnet)
    gl_av_p = model.get_layer("global_average_pooling2d")
    dense0 = model.get_layer("dense")
    dropout = model.get_layer("dropout")
    dense1 = model.get_layer("dense_1")
    prev_mode = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True) # включили режим
    try:
        with tf.GradientTape() as tape:
            if aug_layer is not None:# пропускаем через аугментацию если есть
                x = aug_layer(img_tensor, training=False)
            else:
                x = img_tensor

            conv_out = feature_extractor_model(x, training=False) # активация последней conv карты
            # вручную прогоняем голову послойно
            # граф модели: [Input] -> sequential (Augmentations) -> efficientnetb0 -> [ top_conv -> top_bn -> top_activation ] -> Global Average Pooling 2D -> Dense -> Dropout -> Dense_1 (predictions)
            y = gl_av_p(conv_out, training=False)
            y = dense0(y, training=False)
            y = dropout(y, training=False)
            preds = dense1(y, training=False) # (batch_size=1, вероятности классов)

            if predicted_class_index is None: 
                predicted_class_index = tf.argmax(preds[0])
            loss = preds[:, predicted_class_index] 

        gradients = tape.gradient(loss, conv_out) 
        if gradients is None:
            raise ValueError("error in gradients")
        weights = tf.reduce_mean(gradients, axis=(1,2)) # для каждого канала считаем средний вклад по пикселям
        grad_cam = tf.reduce_sum(conv_out[0] * weights[0], axis=-1) #канал*вес, суммируем
        grad_cam = tf.nn.relu(grad_cam) #находит только положительные влияния. max(feature, 0)
        grad_cam = grad_cam / (tf.reduce_max(grad_cam) + epsilon) # нормализация в [0,1]
        grad_cam = tf.image.resize(grad_cam[..., None], (img_tensor.shape[1], img_tensor.shape[2]))
        return tf.squeeze(grad_cam, -1).numpy()
    finally:
        tf.config.run_functions_eagerly(prev_mode)


def overlay_heatmap(heatmap, img_tensor, transparency=0.4):
    hm = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm_color = mpl.get_cmap("jet")(hm)[:,:,:3] # берём только r,g,b (h,w,3)
    res_img = img_tensor[0].astype(np.float32)
    mn, mx = res_img.min(), res_img.max()
    if mx > 1.0 or mn < 0.0: # не нормализованая картинка
        res_img = (res_img - mn) / (mx - mn + epsilon) # в [0,1]
    overlay = (1 - transparency) * res_img + transparency * hm_color # 45% - hm, 55% - original
    return np.clip(res_img, 0.0, 1.0)

#===============================================================
model_stage_path = "stage_hough_preprocess.keras"
model_locs_path = "best_model.keras"
model_stage = load_model_safely(model_stage_path)
model_locs = load_model_safely(model_locs_path)

def predict_stage(imgs):
    all_preds = []
    class_names = ['immature', 'mature', 'normal']
    for i in imgs:
        x = get_img_array(i)
        x = preprocess_input(x)
        preds = model_stage.predict(x)
        label = np.argmax(preds, axis=1)[0]
        all_preds.append(class_names[label])
    return max(set(all_preds), key=all_preds.count)

def predict_locs(imgs):
    all_preds = []
    class_names = ['IOL', 'NO1NC1', 'NO2NC2', 'NO3NC3', 'NO4NC4', 'NO5NC5', 'NO6NC6']
    for i in imgs:
        x = get_img_array(i)
        x = preprocess_input(x)
        preds = model_locs.predict(x)
        label = np.argmax(preds, axis=1)[0]
        all_preds.append(class_names[label])
    return max(set(all_preds), key=all_preds.count)

last_conv = "top_activation"
def grad_cam_for_some_imgs(imgs, model):
    all_grad_cams = []
    for i in imgs:
        x = get_img_array(i)
        x = preprocess_input(x)
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        _ = model(x_tensor, training=False)
        preds = model.predict(x_tensor, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds_un = preds[0]
        else: 
            preds_un = preds
        predicted_class = int(np.argmax(preds_un[0]))
        heatmap = grad_cam(x, model, last_conv_layer_name=last_conv, predicted_class_index=predicted_class)
        overlay = overlay_heatmap(heatmap, x)
        all_grad_cams.append(overlay)
    return all_grad_cams

st.write("TensorFlow version:", tf.__version__)
st.title("Demo version of the cataract classifier")
st.markdown("Add photos of eyes taken on your phone")
st.markdown("If the model determines the cataract class to be immature, add slit-lamp images of eyes")

st.header("Left eye")
left_photos = st.file_uploader("Upload photos", accept_multiple_files=True, key="left_uploader")

st.header("Right eye")
right_photos = st.file_uploader("Upload photos", accept_multiple_files=True, key="right_uploader")

if st.button("Classify"):
    if left_photos:
        left_res = predict_stage(left_photos)
        # gc1 = grad_cam_for_some_imgs(left_photos, model_stage, last_conv_layer_name_stage)
        # for i in gc1:
        #     st.image(i)
        # if left_res == "immature":
        #     left_slit_lamp = st.file_uploader("Upload photos taken with slit-lamp", accept_multiple_files=True, key="left_slit_lamp_uploader")
        #     if left_slit_lamp:
        #         left_locs_res = predict_locs(left_slit_lamp)
        #         st.subheader(f"Classification result by LOCS III: {left_locs_res}")
        #         gc2 = grad_cam_for_some_imgs(left_slit_lamp, model_locs, last_conv_layer_name_locs)
        #         for j in gc2:
        #             st.image(j)
        # else:
        st.subheader(f"Result for left eye: {left_res}")
        gc = grad_cam_for_some_imgs(left_photos, model_stage)
        for i in gc:
            st.image(i)

    else:
        st.warning("Please upload a photo of the left eye.")

    if right_photos:
        right_res = predict_stage(right_photos)
        gc1 = grad_cam_for_some_imgs(right_photos, model_stage)
        # for i in gc1:
        #     st.image(i)
        # if right_res == "immature":
        #     right_slit_lamp = st.file_uploader("Upload photos taken with slit-lamp", accept_multiple_files=True, key="right_slit_lamp_uploader")
        #     if right_slit_lamp:
        #         right_locs_res = predict_locs(right_slit_lamp)
        #         st.subheader(f"Classification result by LOCS III: {right_locs_res}")
        #         gc2 = grad_cam_for_some_imgs(right_slit_lamp, model_locs)
        #         for j in gc2:
        #             st.image(j)
        # else:
        st.subheader(f"Result for right eye: {right_res}")
        gc = grad_cam_for_some_imgs(right_photos, model_stage)
        for i in gc:
            st.image(i)
    else:
        st.warning("Please upload a photo of the right eye.")